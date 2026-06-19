"""
BARseq2 gene-panel design from the Allen 2020 reference (Yao et al., 2021,
"A taxonomy of transcriptomic cell types across the isocortex and hippocampal
formation").

This module is intentionally **self-contained** (numpy / scipy / pandas / sklearn /
h5py / matplotlib + stdlib only). It deliberately does NOT import ``iss_analysis.io``
or ``iss_analysis.pick_genes`` because those pull in the ``iss_preprocess`` /
``brainglobe_atlasapi`` chain, which is not needed here and is often not installed.
Load it as a top-level module to bypass the package ``__init__``::

    import sys; sys.path.insert(0, ".../iss-analysis/iss_analysis")
    import panel_design as pd_

Pipeline (see plan ``i-want-to-select-encapsulated-comet.md``):

    Stage 0  load_allen2020_streaming  -> cluster/subclass means, gene stats, cell subsample
    Stage 1  filter_candidate_genes    -> biophysical filter (+ forced conventional markers)
    Stage 2  marginal_usefulness       -> per-gene mutual information under BARseq noise
    Stage 3  build_neighbor_graph      -> reference manifold kNN (continuous structure)
    Stage 4  greedy_select x2 + fuse   -> accuracy ranking, overlap ranking, fused panel
    Stage 5  evaluate / outputs

Design notes
------------
* The reference expression matrix is stored *dense* (genes x cells, ~145 GB, gzip
  chunked 1000x10000). Only chunk-aligned column-block streaming is efficient
  (per-gene row reads are ~12 s). Stage 0 makes a single pass.
* Naive-Bayes classification uses a negative-binomial likelihood with r=2 (as in the
  original ``pick_genes``). The ``log(k+1)`` term is cluster-independent so it cancels
  in argmax-over-clusters; classification therefore reduces to a matrix product
  ``X @ A.T + offset`` (A,B below), avoiding the infeasible cells x genes x clusters
  array.
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ss


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Canonical cell-type markers that are *always* force-included in the panel, regardless
# of the data-driven selection. User-editable. (pan-neuronal / NT identity, GABAergic
# subclasses, glutamatergic layer / projection markers.)
CONVENTIONAL_MARKERS = [
    # excitatory / inhibitory identity
    "Slc17a7", "Slc17a6", "Slc17a8", "Gad1", "Gad2", "Slc32a1",
    # GABAergic subclasses + classic interneuron markers
    "Pvalb", "Sst", "Vip", "Lamp5", "Sncg", "Chodl",
    "Calb1", "Calb2", "Cck", "Npy", "Reln", "Cplx3", "Lhx6", "Adarb2",
    # glutamatergic layer / projection identity
    "Cux2", "Rorb", "Fezf2", "Bcl11b", "Foxp2", "Tle4", "Ntsr1", "Tshz2",
    "Satb2", "Tbr1", "Sla", "Rspo1", "Syt6", "Car3", "Nr4a2", "Ccn2",
]

# Immediate-early / activity-dependent genes — expression reflects activity state, not
# cell type. Kept by default; pass drop_ieg=True to remove them.
IEG_GENES = [
    "Fos", "Fosb", "Fosl2", "Arc", "Egr1", "Egr2", "Junb", "Jun", "Npas4",
    "Nr4a1", "Nr4a3", "Bdnf", "Homer1", "Dusp1", "Per1",
]


# --------------------------------------------------------------------------------------
# Small helpers (adapted from iss_analysis.io.filter_genes and pick_genes; inlined to
# keep this module import-safe).
# --------------------------------------------------------------------------------------

def gene_artifact_mask(gene_names, drop_ieg=False):
    """Boolean mask: True = KEEP. Drops gene models / predicted genes / mito / ribo /
    sex / (optionally) immediate-early genes.

    Mirrors ``iss_analysis.io.filter_genes`` and extends it. IEGs are kept by default
    (pass ``drop_ieg=True`` to remove them).
    """
    gene_names = np.asarray(gene_names)

    def matches(pattern):
        return np.array([re.search(pattern, s) is not None for s in gene_names])

    drop = (
        matches(r"Rik$")            # Riken cDNA clones
        | matches(r"Gm\d")          # predicted gene models
        | matches(r"LOC\d")
        | matches(r"^[A-Z]{2}\d*$")  # accession-style names
        | matches(r"^mt-")          # mitochondrial
        | matches(r"^Rp[ls]\d")     # ribosomal proteins
        | matches(r"^(Xist|Tsix|Uty|Ddx3y|Eif2s3y|Kdm5d)$")  # sex
    )
    if drop_ieg:
        drop |= np.isin(gene_names, IEG_GENES)
    return ~drop


def resample_counts(matrix, efficiency, rng=None):
    """Binomial down-sampling of integer counts to simulate ISS sensitivity.

    (Same model as ``pick_genes.resample_counts`` but acts only on the count matrix.)
    """
    assert 0 < efficiency <= 1
    rng = np.random.default_rng() if rng is None else rng
    return rng.binomial(n=matrix.astype("int32"), p=efficiency)


def tau_specificity(mean_expr):
    """Tau tissue-specificity index per gene across groups (rows=groups, cols=genes).

    tau in [0, 1]: 0 = uniformly expressed across all groups, 1 = specific to a single
    group. Computed on the per-group mean expression (log1p of mean counts).
    """
    x = np.log1p(np.asarray(mean_expr, dtype=float))  # (n_groups, n_genes)
    xmax = x.max(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xhat = x / xmax[None, :]                       # normalise each gene to its max
        tau = (1.0 - xhat).sum(axis=0) / (x.shape[0] - 1)
    tau[~np.isfinite(tau)] = 0.0                        # genes with all-zero expression
    return tau


def normalize_library(X, target=None):
    """Per-cell library-size normalisation: rescale each cell so its total counts (over the
    given genes) equal ``target`` (default: the median total). Removes technical per-cell
    sequencing-depth variation, which is otherwise a confound for the NB classifier (cell
    depth is correlated with cell type in the reference). Returns int16 counts.

    Compute size factors on the FULL gene set (pass the whole cells x genes matrix), then
    subset to the panel afterwards.
    """
    X = np.asarray(X)
    tot = X.sum(1).astype(np.float64)
    target = float(np.median(tot)) if target is None else float(target)
    sf = (target / np.maximum(tot, 1.0)).astype(np.float32)        # per-cell scale factor
    out = np.empty(X.shape, dtype=np.int16)                        # chunked to bound memory
    for s in range(0, X.shape[0], 20000):
        e = min(s + 20000, X.shape[0])
        blk = X[s:e].astype(np.float32) * sf[s:e, None]
        np.rint(blk, out=blk)
        out[s:e] = np.minimum(blk, 32767).astype(np.int16)
    return out


def normalize_bundle(bundle):
    """Return a copy of a Stage-0 bundle with the cell subsample library-size normalised and
    the cluster/subclass means + expressing-fractions recomputed from the normalised counts.
    """
    Xn = normalize_library(bundle["subsample_X"])
    sub = bundle["sub_subclass"].astype(int)
    clu = bundle["sub_cluster"].astype(int)
    nsub, nclu = len(bundle["subclass_labels"]), len(bundle["cluster_labels"])

    def gmean(ids, n):
        M = np.zeros((n, Xn.shape[1])); c = np.zeros(n)
        np.add.at(M, ids, Xn.astype(np.float64)); np.add.at(c, ids, 1.0)
        return M / np.maximum(c, 1)[:, None]

    frac = np.zeros((nsub, Xn.shape[1])); fc = np.zeros(nsub)
    np.add.at(frac, sub, (Xn > 0).astype(np.float64)); np.add.at(fc, sub, 1.0)
    nb = dict(bundle)
    nb["subsample_X"] = Xn
    nb["cluster_means"] = gmean(clu, nclu)
    nb["subclass_means"] = gmean(sub, nsub)
    nb["subclass_frac"] = frac / np.maximum(fc, 1)[:, None]
    return nb


# --------------------------------------------------------------------------------------
# Stage 0 - scalable streaming loader
# --------------------------------------------------------------------------------------

def load_allen2020_streaming(
    datapath,
    subsample_n=150000,
    block_size=10000,
    min_cluster_cells=25,
    drop_regions=("HIP", "PAR-POST-PRE-SUB-ProS", "ENT"),
    drop_neighborhoods=("DG/SUB/CA", "Other"),
    drop_subclass_regex=r"(ENT|PPP|RHP|HPF|HATA|TPE|\bSUB\b)",
    include_classes=("Glutamatergic", "GABAergic"),
    include_regions=None,
    seed=0,
    max_blocks=None,
    verbose=True,
):
    """Single streaming pass over the dense ``data/counts`` matrix.

    Returns a dict with per-cluster and per-subclass mean expression, per-gene stats,
    and a random in-memory subsample of cells (raw counts) for the downstream
    information-theoretic / greedy / manifold stages.

    Args:
        datapath: folder containing ``expression_matrix.hdf5`` and ``metadata.csv``.
        subsample_n: number of neocortical-neuron cells to keep in memory.
        block_size: column-block width for streaming (10000 = chunk aligned).
        min_cluster_cells: drop clusters with fewer kept cells than this.
        drop_regions / drop_neighborhoods: metadata values to exclude (non-isocortex).
        drop_subclass_regex: drop whole subclasses whose name matches (allocortical /
            transition types, e.g. ENT/PPP/TPE/SUB); set to None to disable.
        include_classes: cell ``class_label`` values to keep (neurons only by default).
        seed: RNG seed for the subsample.
        max_blocks: if set, process only this many blocks (for quick testing).
        verbose: print progress.

    Returns:
        dict (see keys assembled at the end of the function).
    """
    import h5py

    datapath = Path(datapath)
    rng = np.random.default_rng(seed)
    h5 = h5py.File(datapath / "expression_matrix.hdf5", "r")
    counts = h5["data/counts"]  # (n_genes, n_cells) dense int32
    n_genes, n_cells = counts.shape
    gene_names = np.array([g.decode() for g in h5["data/gene"][:]])
    samples = np.array([s.decode() for s in h5["data/samples"][:]])

    # --- align metadata to the HDF5 column order ---
    meta = pd.read_csv(
        datapath / "metadata.csv",
        usecols=[
            "sample_name", "class_label", "region_label",
            "neighborhood_label", "subclass_label", "cluster_label",
        ],
        low_memory=False,
    ).set_index("sample_name")
    meta = meta.reindex(samples)  # rows now in HDF5 column order; missing -> NaN

    # --- cell mask: neuron class + neocortex region/neighborhood ---
    mask = meta["class_label"].isin(include_classes).to_numpy()
    if include_regions:                                   # keep ONLY these regions (e.g. VISp)
        mask &= meta["region_label"].isin(include_regions).to_numpy()
    if drop_regions:
        mask &= ~meta["region_label"].isin(drop_regions).to_numpy()
    if drop_neighborhoods:
        mask &= ~meta["neighborhood_label"].isin(drop_neighborhoods).to_numpy()

    # --- drop allocortical / transition subclasses by name (neocortex focus) ---
    dropped_subclasses = set()
    if drop_subclass_regex:
        rx_s = re.compile(drop_subclass_regex)
        sub_str = meta["subclass_label"].fillna("")
        sub_drop = sub_str.map(lambda s: rx_s.search(s) is not None).to_numpy()
        dropped_subclasses = set(sub_str[mask & sub_drop].unique())
        mask &= ~sub_drop

    # --- decide which clusters to keep (min size), from metadata ---
    clu_series = meta["cluster_label"]
    sizes = clu_series[mask].value_counts()
    keep_clusters = set(sizes.index[sizes >= min_cluster_cells])
    dropped_small = set(sizes.index) - keep_clusters
    mask &= clu_series.isin(keep_clusters).to_numpy()

    kept_idx = np.nonzero(mask)[0]          # HDF5 column indices of kept cells (sorted)
    n_kept = kept_idx.size
    if verbose:
        print(f"[load] {n_kept} kept neocortical neurons "
              f"({len(keep_clusters)} clusters); dropped {len(dropped_small)} small "
              f"clusters + {len(dropped_subclasses)} allocortical subclasses", flush=True)
        if dropped_subclasses:
            print(f"[load] dropped subclasses: {sorted(dropped_subclasses)}", flush=True)

    # --- label encodings over kept cells ---
    meta_k = meta.iloc[kept_idx]
    subclass_labels = np.array(sorted(meta_k["subclass_label"].unique()))
    cluster_labels = np.array(sorted(meta_k["cluster_label"].unique()))
    sub_to_i = {s: i for i, s in enumerate(subclass_labels)}
    clu_to_i = {c: i for i, c in enumerate(cluster_labels)}
    n_sub, n_clu = len(subclass_labels), len(cluster_labels)

    cell_clu = meta_k["cluster_label"].map(clu_to_i).to_numpy()
    cell_sub = meta_k["subclass_label"].map(sub_to_i).to_numpy()
    cell_region = meta_k["region_label"].to_numpy()

    # cluster -> subclass map
    cluster_to_subclass = np.empty(n_clu, dtype=int)
    pairs = meta_k[["cluster_label", "subclass_label"]].drop_duplicates()
    for _, r in pairs.iterrows():
        cluster_to_subclass[clu_to_i[r["cluster_label"]]] = sub_to_i[r["subclass_label"]]

    # --- choose the subsample (indices into kept cells) ---
    n_sub_cells = min(subsample_n, n_kept)
    in_sub = np.zeros(n_kept, dtype=bool)
    in_sub[rng.choice(n_kept, n_sub_cells, replace=False)] = True
    sub_row = np.full(n_kept, -1, dtype=np.int64)
    sub_row[in_sub] = np.arange(n_sub_cells)

    # --- accumulators ---
    cluster_sum = np.zeros((n_clu, n_genes), dtype=np.float64)
    cluster_cnt = np.zeros(n_clu, dtype=np.float64)
    subclass_sum = np.zeros((n_sub, n_genes), dtype=np.float64)
    subclass_nnz = np.zeros((n_sub, n_genes), dtype=np.float64)
    subclass_cnt = np.zeros(n_sub, dtype=np.float64)
    subsample_X = np.zeros((n_sub_cells, n_genes), dtype=np.int16)

    # --- stream column blocks ---
    starts = list(range(0, n_cells, block_size))
    if max_blocks is not None:
        starts = starts[:max_blocks]
    kept_seen = 0
    t0 = time.time()
    for bi, start in enumerate(starts):
        end = min(start + block_size, n_cells)
        blk_mask = mask[start:end]
        if not blk_mask.any():
            continue
        local = np.nonzero(blk_mask)[0]
        block = counts[:, start:end]                      # (n_genes, block) int32
        block_kept = block[:, local].T                    # (nkept_blk, n_genes)
        nkb = block_kept.shape[0]

        gk = kept_seen + np.arange(nkb)                    # global kept indices
        clu_ids = cell_clu[gk]
        sub_ids = cell_sub[gk]

        # one-hot (groups x cells) sparse @ dense for fast group sums
        oh_c = ss.csr_matrix(
            (np.ones(nkb), (clu_ids, np.arange(nkb))), shape=(n_clu, nkb)
        )
        oh_s = ss.csr_matrix(
            (np.ones(nkb), (sub_ids, np.arange(nkb))), shape=(n_sub, nkb)
        )
        bk = block_kept.astype(np.float64)
        cluster_sum += oh_c @ bk
        np.add.at(cluster_cnt, clu_ids, 1.0)
        subclass_sum += oh_s @ bk
        subclass_nnz += oh_s @ (bk > 0)
        np.add.at(subclass_cnt, sub_ids, 1.0)

        # collect subsample rows
        sel = in_sub[gk]
        if sel.any():
            rows = sub_row[gk[sel]]
            subsample_X[rows] = np.minimum(block_kept[sel], 32767).astype(np.int16)

        kept_seen += nkb
        if verbose and (bi % 20 == 0 or bi == len(starts) - 1):
            print(f"[load] block {bi+1}/{len(starts)} "
                  f"({kept_seen} kept) {time.time()-t0:.0f}s", flush=True)

    h5.close()

    cluster_means = cluster_sum / np.maximum(cluster_cnt, 1)[:, None]
    subclass_means = subclass_sum / np.maximum(subclass_cnt, 1)[:, None]
    subclass_frac = subclass_nnz / np.maximum(subclass_cnt, 1)[:, None]

    return dict(
        gene_names=gene_names,
        cluster_means=cluster_means,            # (n_clu, n_genes)
        cluster_cnt=cluster_cnt,
        cluster_labels=cluster_labels,
        subclass_means=subclass_means,          # (n_sub, n_genes)
        subclass_frac=subclass_frac,            # fraction of cells expressing per subclass
        subclass_cnt=subclass_cnt,
        subclass_labels=subclass_labels,
        cluster_to_subclass=cluster_to_subclass,
        subsample_X=subsample_X[:kept_seen if max_blocks else n_sub_cells],
        sub_cluster=cell_clu[in_sub],
        sub_subclass=cell_sub[in_sub],
        sub_region=cell_region[in_sub],
        n_kept=int(kept_seen),
    )


def save_bundle(bundle, cache_dir):
    """Persist a Stage-0 bundle so the expensive streaming pass runs only once."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "subsample_X.npy", bundle["subsample_X"])
    arrays = {k: v for k, v in bundle.items() if k != "subsample_X"}
    np.savez(cache_dir / "bundle.npz", **arrays)
    print(f"[save] wrote bundle to {cache_dir}", flush=True)


def load_bundle(cache_dir):
    cache_dir = Path(cache_dir)
    out = {}
    with np.load(cache_dir / "bundle.npz", allow_pickle=True) as z:
        for k in z.files:
            out[k] = z[k]
    out["subsample_X"] = np.load(cache_dir / "subsample_X.npy")
    for k in ("n_kept",):
        out[k] = int(out[k])
    return out


# --------------------------------------------------------------------------------------
# Stage 1 - biophysical candidate filter (+ forced conventional markers)
# --------------------------------------------------------------------------------------

def filter_candidate_genes(
    bundle,
    min_max_subclass_mean=1.0,
    min_tau=0.3,
    drop_ieg=False,
    conventional=CONVENTIONAL_MARKERS,
    verbose=True,
):
    """Reduce ~31k genes to a biophysically sensible candidate pool, encoding the two
    BARseq constraints, and always retain the conventional markers.

    This is a permissive *gate*: it removes clearly-unusable genes (very low expression,
    ubiquitous, artifacts). The quantitative "highly expressed" preference is enforced
    downstream by the BARseq noise model in Stages 2/4 (low-expression genes down-sample
    to mostly zeros, giving low MI / classification gain).

    * "relatively highly expressed"  -> max over subclass means >= ``min_max_subclass_mean``
    * "not broadly expressed"        -> tau specificity >= ``min_tau``
    * not a gene-model / mito / ribo / sex (/ IEG) artifact

    Returns a DataFrame (one row per gene) with the metrics and the boolean ``candidate``
    column; conventional markers are flagged and force-kept even if they fail the filters.
    """
    gn = np.asarray(bundle["gene_names"])
    sub_means = bundle["subclass_means"]                       # (n_sub, n_genes)
    max_sub_mean = sub_means.max(axis=0)
    top_subclass = bundle["subclass_labels"][sub_means.argmax(axis=0)]
    tau = tau_specificity(sub_means)
    frac = bundle["subclass_frac"]                              # (n_sub, n_genes)
    frac_top = frac[sub_means.argmax(axis=0), np.arange(len(gn))]
    keep_artifact = gene_artifact_mask(gn, drop_ieg=drop_ieg)

    passes = keep_artifact & (max_sub_mean >= min_max_subclass_mean) & (tau >= min_tau)
    conv = np.isin(gn, conventional)
    candidate = passes | conv

    df = pd.DataFrame({
        "gene": gn,
        "idx": np.arange(len(gn)),
        "max_subclass_mean": max_sub_mean,
        "top_subclass": top_subclass,
        "frac_expr_top": frac_top,
        "tau": tau,
        "conventional": conv,
        "passes_biophysical": passes,
        "candidate": candidate,
    })
    if verbose:
        missing = sorted(set(conventional) - set(gn))
        print(f"[filter] candidates: {int(candidate.sum())} "
              f"({int(passes.sum())} pass biophysical + {int((conv & ~passes).sum())} "
              f"forced conventional)", flush=True)
        if missing:
            print(f"[filter] conventional markers not in dataset: {missing}", flush=True)
    return df


# --------------------------------------------------------------------------------------
# Stage 2 - marginal usefulness via mutual information under BARseq noise
# --------------------------------------------------------------------------------------

def _mi_discrete(X, y, max_count=3):
    """Mutual information (nats) between each discrete column of X (values 0..max_count)
    and label vector y. Vectorised over genes. Returns array of length n_genes."""
    X = np.minimum(X, max_count).astype(np.int64)
    n, g = X.shape
    L = int(y.max()) + 1
    onehot = np.zeros((n, L), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0
    K = int(X.max()) + 1
    joint = np.empty((g, K, L))
    for a in range(K):
        joint[:, a, :] = (X == a).astype(np.float64).T @ onehot
    joint /= n
    px = joint.sum(axis=2)                                  # (g, K)
    py = joint.sum(axis=1)                                  # (g, L)
    denom = px[:, :, None] * py[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0, joint / denom, 1.0)
        mi = (joint * np.log(ratio)).sum(axis=(1, 2))
    return mi


def marginal_usefulness(bundle, cand_idx, efficiency=0.01, lam=0.5, max_count=3, seed=0):
    """Per-candidate-gene marginal usefulness = MI(subclass) + lam*MI(cluster), measured
    on BARseq-down-sampled, discretised counts. This is the fast first usefulness ranking.
    """
    rng = np.random.default_rng(seed)
    X = bundle["subsample_X"][:, cand_idx].astype("int32")
    Xds = resample_counts(X, efficiency, rng=rng)
    mi_sub = _mi_discrete(Xds, bundle["sub_subclass"].astype(int), max_count)
    mi_clu = _mi_discrete(Xds, bundle["sub_cluster"].astype(int), max_count)
    return dict(mi_subclass=mi_sub, mi_cluster=mi_clu, marginal=mi_sub + lam * mi_clu)


# --------------------------------------------------------------------------------------
# Stage 3 - reference manifold (HVG -> PCA -> kNN) capturing continuous + discrete structure
# --------------------------------------------------------------------------------------

def _normalize_log(X):
    """Library-size normalise to the median depth, then log1p (float32)."""
    X = X.astype("float32")
    lib = X.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * np.median(lib))


def build_neighbor_graph(bundle, overlap_cells=6000, n_hvg=2000, n_pcs=50, k=15, seed=0):
    """Build the full-transcriptome reference kNN graph on a subset of cells. This graph
    is the "ground truth" the panel must reproduce (Stage 4b); it encodes discrete
    clusters *and* continuous gradients.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n = bundle["subsample_X"].shape[0]
    ov = np.sort(rng.choice(n, min(overlap_cells, n), replace=False))
    Xn = _normalize_log(bundle["subsample_X"][ov])
    keep = gene_artifact_mask(bundle["gene_names"])
    var = np.where(keep, Xn.var(0), -1.0)
    hvg = np.argsort(var)[::-1][:n_hvg]
    Z = Xn[:, hvg]
    Z = Z - Z.mean(0)
    pcs = PCA(n_components=n_pcs, random_state=seed).fit_transform(Z)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pcs)
    _, idx = nn.kneighbors(pcs)
    ref_knn = idx[:, 1:]                              # (n_ov, k), self removed
    return dict(overlap_idx=ov, ref_knn=ref_knn, k=k)


# --------------------------------------------------------------------------------------
# Stage 4 - scalable Naive-Bayes classification + two greedy selections + rank fusion
# --------------------------------------------------------------------------------------

def nb_coeffs(cluster_means, efficiency, nu=0.001, r=2):
    """Negative-binomial (r=2) coefficients for the linearised log-likelihood.

    For cell counts x and down-sampled cluster mean mu, the cluster-dependent part of the
    log-likelihood is ``x*A + B`` with A,B below; the ``log(x+1)`` term is dropped because
    it is constant across clusters and cancels in argmax. Classification is then the matrix
    product ``X @ A.T + B.sum`` -- no cells x genes x clusters array needed.
    """
    mu = cluster_means * efficiency + nu
    A = np.log(mu / (mu + r)).astype("float32")        # (n_clu, n_genes), coeff on counts
    B = (r * np.log(r / (mu + r))).astype("float32")   # (n_clu, n_genes), offset
    return A, B


def _cell_loglik(Xf, A, B, sel):
    """(N, C) cluster log-likelihood (up to const) for the selected gene columns ``sel``.

    ``X @ A.T`` gives (N, n_clu); the offset is the per-cluster sum of B over the
    selected genes, i.e. ``B[:, sel].sum(axis=1)`` -> (n_clu,).
    """
    if len(sel) == 0:
        return np.zeros((Xf.shape[0], A.shape[0]), dtype="float32")
    return Xf[:, sel] @ A[:, sel].T + B[:, sel].sum(axis=1)[None, :]


def _softmax_rows(L):
    """Row-wise softmax with max-subtraction for numerical stability."""
    M = L - L.max(axis=1, keepdims=True)
    np.exp(M, out=M)
    M /= M.sum(axis=1, keepdims=True)
    return M


def _softmax_axis1(L):
    """Softmax along axis 1 of a 3D array (N, K, b), max-subtracted for stability."""
    M = L - L.max(axis=1, keepdims=True)
    np.exp(M, out=M)
    M /= M.sum(axis=1, keepdims=True)
    return M


def _posterior_objective(L, true_clu, true_sub, c2s, c2s_onehot, lam, kind="soft_acc",
                         w_sub=1.0):
    """Smooth NB-posterior objective for the hierarchical greedy.

    ``L`` is the (N, C) cluster log-likelihood up to a *per-cell* additive constant; that
    constant cancels in the softmax, so ``softmax(L)`` is exactly the NB posterior under a
    uniform cluster prior (the same prior the argmax classifier uses). The subclass
    posterior is the cluster posterior grouped by ``c2s_onehot`` (C, S).

    ``w_sub``/``lam`` weight the subclass and cluster terms: (w_sub=1, lam=0) is a
    subclass-only objective, (w_sub=0, lam=1) a cluster-only objective, and the default
    (w_sub=1, lam=0.5) the balanced objective.

    kind:
      "soft_acc" (default): w_sub*mean P(true_sub) + lam*mean P(true_clu) -- smooth relaxation
          of the 0/1 accuracy that is still reported downstream. Every gene moves the score for
          every cell, so candidates are cleanly discriminated (no zero-gain ties).
      "logpost": w_sub*mean log P(true_sub) + lam*mean log P(true_clu) -- cross-entropy surrogate.
      "accuracy": legacy hard 0/1 accuracy of the argmax (w_sub*sub_acc + lam*clu_acc).
    """
    if kind == "accuracy":
        pred = L.argmax(1)
        sub = float((c2s[pred] == true_sub).mean())
        clu = float((pred == true_clu).mean())
        return w_sub * sub + lam * clu
    ar = np.arange(L.shape[0])
    P = _softmax_rows(L)
    p_clu = P[ar, true_clu]
    p_sub = (P @ c2s_onehot)[ar, true_sub]
    if kind == "logpost":
        return (w_sub * float(np.log(p_sub + 1e-12).mean())
                + lam * float(np.log(p_clu + 1e-12).mean()))
    return w_sub * float(p_sub.mean()) + lam * float(p_clu.mean())


def _posterior_objective_single(L, true, kind="soft_acc"):
    """Single-level analogue of ``_posterior_objective`` (subclass- or cluster-level).

    kind="soft_acc": mean P(true); "logpost": mean log P(true); "accuracy": mean(argmax==true).
    """
    if kind == "accuracy":
        return float((L.argmax(1) == true).mean())
    p = _softmax_rows(L)[np.arange(L.shape[0]), true]
    if kind == "logpost":
        return float(np.log(p + 1e-12).mean())
    return float(p.mean())


def greedy_accuracy(
    Xf, A, B, true_clu, true_sub, c2s, seed_sel, n_select,
    lam=0.5, w_sub=1.0, objective="soft_acc", verbose=True,
):
    """Lazy (CELF) forward greedy maximising a hierarchical classification objective
    ``w_sub*subclass + lam*cluster``.

    ``w_sub``/``lam`` weight the two levels: (w_sub=1, lam=0) is a subclass-only objective,
    (w_sub=0, lam=1) a cluster-only objective; the default (w_sub=1, lam=0.5) is balanced.

    ``objective`` (see ``_posterior_objective``) selects the greedy score:
      "soft_acc" (default): smooth NB-posterior relaxation of ``w_sub*sub_acc + lam*cluster_acc``;
      "logpost": cross-entropy surrogate; "accuracy": legacy hard 0/1 accuracy.
    The smooth objectives give every candidate a non-zero marginal gain at every step, so
    the greedy is not stuck breaking arbitrary ties among genes that flip no argmax. The
    reported ``hist`` is always hard ``(sub_acc, clu_acc)`` for interpretability.

    Uses cached marginal gains in a max-heap: because marginal gains diminish as the panel
    grows, stale gains are upper bounds, so each step only re-evaluates the few top
    candidates instead of all of them. Each evaluation is a cheap (N, n_clu) softmax/argmax.

    Xf: (N, G) float32 down-sampled counts on candidate genes.
    A, B: (n_clu, G) NB coefficients on candidate genes.
    seed_sel: candidate-space indices to force-include first (conventional markers).
    Returns: order (list of candidate indices), history of (sub_acc, clu_acc).
    """
    import heapq

    N, G = Xf.shape
    Cn = A.shape[0]
    S = int(c2s.max()) + 1
    c2s_onehot = np.zeros((Cn, S), dtype="float32")
    c2s_onehot[np.arange(Cn), c2s] = 1.0
    selected = list(dict.fromkeys(seed_sel))
    sel_mask = np.zeros(G, dtype=bool)
    sel_mask[selected] = True
    Lc = _cell_loglik(Xf, A, B, selected).copy()

    def score_with(g):
        return _posterior_objective(
            Lc + np.outer(Xf[:, g], A[:, g]) + B[:, g],
            true_clu, true_sub, c2s, c2s_onehot, lam, objective, w_sub=w_sub)

    def cur_score():
        return _posterior_objective(
            Lc, true_clu, true_sub, c2s, c2s_onehot, lam, objective, w_sub=w_sub)

    def hard_acc():                                    # interpretable accuracy for reporting
        pred = Lc.argmax(1)
        return float((c2s[pred] == true_sub).mean()), float((pred == true_clu).mean())

    base = cur_score()
    heap = []                                          # (-marginal_gain, g)
    for g in np.nonzero(~sel_mask)[0]:
        heapq.heappush(heap, (-(score_with(int(g)) - base), int(g)))

    order, hist = list(selected), []
    while len(selected) < min(n_select, G) and heap:
        base = cur_score()
        while True:                                    # lazy re-evaluation
            neg_gain, g = heapq.heappop(heap)
            if sel_mask[g]:
                continue
            gain = score_with(g) - base
            if not heap or gain >= -heap[0][0]:
                break
            heapq.heappush(heap, (-gain, g))
        selected.append(g)
        sel_mask[g] = True
        order.append(g)
        Lc += np.outer(Xf[:, g], A[:, g]) + B[:, g]
        sub, clu = hard_acc()
        hist.append((sub, clu))
        if verbose and len(selected) % 25 == 0:
            print(f"[greedy-acc] {len(selected)} genes  sub={sub:.3f} clu={clu:.3f}",
                  flush=True)
    return order, hist


def greedy_accuracy_single(Xf, A, B, true, seed_sel, n_select, objective="soft_acc",
                           verbose=True):
    """Lazy (CELF) greedy maximising a single-level classification objective. ``A,B`` are the
    NB coefficients for the target grouping (subclass means -> 16-way, or cluster means ->
    215-way) and ``true`` the matching labels. ``objective`` (see
    ``_posterior_objective_single``): "soft_acc" (default, mean true-class posterior),
    "logpost" (cross-entropy), or "accuracy" (legacy hard 0/1). The smooth objectives avoid
    zero-gain ties among genes that flip no argmax; ``hist`` is always hard accuracy."""
    import heapq

    N, G = Xf.shape
    selected = list(dict.fromkeys(seed_sel))
    sel_mask = np.zeros(G, dtype=bool)
    sel_mask[selected] = True
    Lc = _cell_loglik(Xf, A, B, selected).copy()

    def score_with(g):
        return _posterior_objective_single(
            Lc + np.outer(Xf[:, g], A[:, g]) + B[:, g], true, objective)

    def cur_score():
        return _posterior_objective_single(Lc, true, objective)

    base = cur_score()
    heap = []
    for g in np.nonzero(~sel_mask)[0]:
        heapq.heappush(heap, (-(score_with(int(g)) - base), int(g)))

    order, hist = list(selected), []
    while len(selected) < min(n_select, G) and heap:
        base = cur_score()
        while True:
            neg_gain, g = heapq.heappop(heap)
            if sel_mask[g]:
                continue
            gain = score_with(g) - base
            if not heap or gain >= -heap[0][0]:
                break
            heapq.heappush(heap, (-gain, g))
        selected.append(g); sel_mask[g] = True; order.append(g)
        Lc += np.outer(Xf[:, g], A[:, g]) + B[:, g]
        a = float((Lc.argmax(1) == true).mean())       # interpretable accuracy for reporting
        hist.append(a)
        if verbose and len(selected) % 25 == 0:
            print(f"[greedy-acc] {len(selected)} genes  acc={a:.3f}", flush=True)
    return order, hist


def _mean_knn_overlap(D, R, k):
    """Mean fraction of each cell's k panel-neighbours that are also reference neighbours.
    D: (N,N) panel squared-distances; R: (N,N) bool reference adjacency (self=False)."""
    N = D.shape[0]
    part = np.argpartition(D, k, axis=1)[:, :k + 1]     # k+1 includes self (dist 0)
    gathered = R[np.arange(N)[:, None], part]           # self contributes 0 (R[i,i]=False)
    return gathered.sum(1).mean() / k


def greedy_overlap(Xf_ov, ref_knn, seed_sel, n_select, k=15, verbose=True):
    """Lazy (CELF) forward greedy maximising reference kNN-graph overlap (geneBasis-style
    continuous objective). The panel squared-distance matrix ``D`` is maintained
    incrementally; cached marginal overlap gains (which diminish as the panel grows) let
    each step re-evaluate only a few candidates. Returns order (candidate indices) and
    overlap history.
    """
    import heapq

    t = np.log1p(Xf_ov)                                 # (N, G) BARseq-like transform
    N, G = t.shape
    R = np.zeros((N, N), dtype=bool)
    R[np.arange(N)[:, None], ref_knn] = True
    D = np.zeros((N, N), dtype="float32")
    selected = list(dict.fromkeys(seed_sel))
    sel_mask = np.zeros(G, dtype=bool)
    sel_mask[selected] = True
    for g in selected:
        d = t[:, g]
        D += (d[:, None] - d[None, :]) ** 2

    def overlap_with(g):
        d = t[:, g]
        return _mean_knn_overlap(D + (d[:, None] - d[None, :]) ** 2, R, k)

    base = _mean_knn_overlap(D, R, k)
    heap = []
    for g in np.nonzero(~sel_mask)[0]:
        heapq.heappush(heap, (-(overlap_with(int(g)) - base), int(g)))

    order, hist = list(selected), []
    while len(selected) < min(n_select, G) and heap:
        base = _mean_knn_overlap(D, R, k)
        while True:
            neg_gain, g = heapq.heappop(heap)
            if sel_mask[g]:
                continue
            ov = overlap_with(g)
            gain = ov - base
            if not heap or gain >= -heap[0][0]:
                break
            heapq.heappush(heap, (-gain, g))
        selected.append(g)
        sel_mask[g] = True
        order.append(g)
        d = t[:, g]
        D += (d[:, None] - d[None, :]) ** 2
        hist.append(ov)
        if verbose and len(selected) % 25 == 0:
            print(f"[greedy-ovl] {len(selected)} genes  overlap={ov:.3f}", flush=True)
    return order, hist


def reciprocal_rank_fusion(order_acc, order_ovl, conv_mask_cand, kappa=60):
    """Fuse two candidate-index rankings by RRF; conventional markers pinned on top.
    Returns a DataFrame indexed by candidate index with rank columns + fused rank."""
    ra = {g: i for i, g in enumerate(order_acc)}
    ro = {g: i for i, g in enumerate(order_ovl)}
    genes = sorted(set(order_acc) | set(order_ovl))
    rows = []
    for g in genes:
        a, o = ra.get(g), ro.get(g)
        score = (1.0 / (kappa + a) if a is not None else 0.0) + \
                (1.0 / (kappa + o) if o is not None else 0.0)
        rows.append((g, a, o, score, bool(conv_mask_cand[g])))
    df = pd.DataFrame(rows, columns=["cand", "rank_accuracy", "rank_overlap",
                                     "rrf_score", "conventional"])
    df = df.sort_values(["conventional", "rrf_score"], ascending=[False, False])
    df = df.reset_index(drop=True)
    df["fused_rank"] = np.arange(len(df))
    return df


# --------------------------------------------------------------------------------------
# Stage 4d - plug-and-play optimizers (swappable objective x search strategy)
# --------------------------------------------------------------------------------------
#
# Selection is factored into two interchangeable pieces so new ideas drop in cheaply:
#
#   * an OBJECTIVE  - *what* makes a panel good (classification accuracy, manifold
#     overlap, ...). It exposes a small *incremental* interface so a search strategy can
#     grow a panel one (or a few) genes at a time without recomputing from scratch. Gene
#     indices are positions in the candidate pool (0..G-1).
#
#   * a SEARCH STRATEGY - *how* the panel is grown (greedy / stochastic random walks /
#     stochastic-greedy / ...). A strategy only ever talks to the objective through the
#     interface below, so any strategy composes with any objective.
#
# ``greedy_accuracy`` / ``greedy_overlap`` above are the original hand-fused (objective +
# CELF greedy) implementations and remain the default. The classes/strategies here express
# the same maths through the generic interface and add the stochastic optimizers. Register
# a new strategy in ``SELECTORS`` (or subclass ``SelectionObjective``) and it is usable
# from ``run_selection(acc_strategy=...)`` immediately.


class SelectionObjective:
    """Incremental forward-selection objective. ``state`` is opaque to the strategy; only
    these methods are used (gene indices are candidate-pool positions 0..n_genes-1):

    ===================  =====================================================
    ``init(seed)``       -> state            new panel, force-including ``seed``
    ``add(state, genes)``-> state            commit gene(s) (mutates + returns state)
    ``clone(state)``     -> state            independent copy (restarts / branching)
    ``selected(state)``  -> list[int]        genes chosen so far, in selection order
    ``score(state)``     -> float            objective value (higher = better)
    ``gain_one(s, g)``   -> float            exact marginal gain of adding gene ``g``
    ``gain_all(state)``  -> (G,) np.ndarray  marginal gain of every gene (-inf if chosen)
    ===================  =====================================================
    """

    n_genes = 0

    def init(self, seed=()):
        raise NotImplementedError

    def add(self, state, genes):
        raise NotImplementedError

    def clone(self, state):
        raise NotImplementedError

    def selected(self, state):
        return list(state["order"])

    def score(self, state):
        raise NotImplementedError

    def gain_one(self, state, g, base=None):
        raise NotImplementedError

    def gain_all(self, state):
        raise NotImplementedError


class AccuracyObjective(SelectionObjective):
    """Naive-Bayes accuracy objective for forward selection, matching the greedy design
    (``greedy_accuracy`` / ``greedy_accuracy_single`` / ``_posterior_objective``).

    ``score = w_sub * <subclass term> + lam * <cluster term>``, where each term is a function
    (``objective``) of the per-cell NB posterior (``softmax`` of the NB log-likelihood = the
    posterior under a uniform prior):

      - ``"soft_acc"`` (default): mean P(true) -- smooth relaxation of 0/1 accuracy that
        avoids zero-gain argmax ties (every gene moves every cell's score).
      - ``"logpost"``: mean log P(true) -- cross-entropy surrogate.
      - ``"accuracy"``: legacy hard 0/1 accuracy of the argmax.

    The CLUSTER term uses the cluster classifier (``A``, ``B`` from cluster means -> ``Lc``).
    The SUBCLASS term uses the DIRECT subclass classifier (``Asub``, ``Bsub`` from subclass
    means -> ``Ls``) when those are given, else the cluster->subclass rollup ``c2s``. Marginal
    gains use rank-1 updates of the cached log-likelihoods; ``gain_all`` evaluates every
    candidate in gene blocks sized so the transient ``(N, n_clu, block)`` array stays under
    ``mem_elements``.
    """

    def __init__(self, Xf, A, B, true_clu, true_sub, c2s, Asub=None, Bsub=None,
                 w_sub=1.0, lam=0.5, objective="soft_acc", mem_elements=1e8):
        self.Xf = np.ascontiguousarray(Xf, dtype="float32")
        self.A = np.asarray(A, dtype="float32")
        self.B = np.asarray(B, dtype="float32")
        self.true_clu = np.asarray(true_clu)
        self.true_sub = np.asarray(true_sub)
        self.c2s = np.asarray(c2s)
        self.direct = Asub is not None and Bsub is not None     # direct subclass classifier?
        self.Asub = np.asarray(Asub, dtype="float32") if self.direct else None
        self.Bsub = np.asarray(Bsub, dtype="float32") if self.direct else None
        self.w_sub, self.lam, self.objective = float(w_sub), float(lam), objective
        self.N, self.n_genes = self.Xf.shape
        self.C = self.A.shape[0]
        self.ar = np.arange(self.N)
        if not self.direct:                                     # rollup one-hot (C, S)
            S = int(self.c2s.max()) + 1
            self.c2s_onehot = np.zeros((self.C, S), dtype="float32")
            self.c2s_onehot[np.arange(self.C), self.c2s] = 1.0
        self.block = max(1, int(mem_elements // max(1, self.N * self.C)))

    def init(self, seed=()):
        seed = list(dict.fromkeys(int(g) for g in seed))
        st = {"Lc": _cell_loglik(self.Xf, self.A, self.B, seed).astype("float32"),
              "mask": np.zeros(self.n_genes, dtype=bool), "order": list(seed)}
        st["mask"][seed] = True
        if self.direct:
            st["Ls"] = _cell_loglik(self.Xf, self.Asub, self.Bsub, seed).astype("float32")
        return st

    def clone(self, state):
        s = {"Lc": state["Lc"].copy(), "mask": state["mask"].copy(),
             "order": list(state["order"])}
        if self.direct:
            s["Ls"] = state["Ls"].copy()
        return s

    def add(self, state, genes):
        if np.isscalar(genes):
            genes = [genes]
        for g in genes:
            g = int(g)
            if state["mask"][g]:
                continue
            state["Lc"] += np.outer(self.Xf[:, g], self.A[:, g]) + self.B[:, g]
            if self.direct:
                state["Ls"] += np.outer(self.Xf[:, g], self.Asub[:, g]) + self.Bsub[:, g]
            state["mask"][g] = True
            state["order"].append(g)
        return state

    def _terms(self, Lc, Ls):
        """(subclass_term, cluster_term) under self.objective from cached (2D) log-liks."""
        if self.objective == "accuracy":
            clu = float((Lc.argmax(1) == self.true_clu).mean())
            sub = (float((Ls.argmax(1) == self.true_sub).mean()) if self.direct
                   else float((self.c2s[Lc.argmax(1)] == self.true_sub).mean()))
            return sub, clu
        Pc = _softmax_rows(Lc)
        p_clu = Pc[self.ar, self.true_clu]
        p_sub = (_softmax_rows(Ls)[self.ar, self.true_sub] if self.direct
                 else (Pc @ self.c2s_onehot)[self.ar, self.true_sub])
        if self.objective == "logpost":
            return float(np.log(p_sub + 1e-12).mean()), float(np.log(p_clu + 1e-12).mean())
        return float(p_sub.mean()), float(p_clu.mean())

    def detail(self, state):
        """Hard (subclass_acc, cluster_acc) of the current panel (for reporting)."""
        clu = float((state["Lc"].argmax(1) == self.true_clu).mean())
        sub = (float((state["Ls"].argmax(1) == self.true_sub).mean()) if self.direct
               else float((self.c2s[state["Lc"].argmax(1)] == self.true_sub).mean()))
        return sub, clu

    def score(self, state):
        sub, clu = self._terms(state["Lc"], state.get("Ls"))
        return self.w_sub * sub + self.lam * clu

    def gain_one(self, state, g, base=None):
        if base is None:
            base = self.score(state)
        Lc2 = state["Lc"] + (np.outer(self.Xf[:, g], self.A[:, g]) + self.B[:, g])
        Ls2 = (state["Ls"] + (np.outer(self.Xf[:, g], self.Asub[:, g]) + self.Bsub[:, g])
               if self.direct else None)
        sub, clu = self._terms(Lc2, Ls2)
        return (self.w_sub * sub + self.lam * clu) - base

    def gain_all(self, state):
        base = self.score(state)
        ar = self.ar
        gains = np.full(self.n_genes, -np.inf, dtype="float32")
        todo = np.nonzero(~state["mask"])[0]
        for s in range(0, todo.size, self.block):
            gg = todo[s:s + self.block]
            candC = (state["Lc"][:, :, None]
                     + self.Xf[:, gg][:, None, :] * self.A[:, gg][None, :, :]
                     + self.B[:, gg][None, :, :])               # (N, C, b)
            candS = None
            if self.direct:
                candS = (state["Ls"][:, :, None]
                         + self.Xf[:, gg][:, None, :] * self.Asub[:, gg][None, :, :]
                         + self.Bsub[:, gg][None, :, :])         # (N, S, b)
            if self.objective == "accuracy":
                predc = candC.argmax(1)                          # (N, b)
                clu = (predc == self.true_clu[:, None]).mean(0)
                sub = ((candS.argmax(1) == self.true_sub[:, None]).mean(0) if self.direct
                       else (self.c2s[predc] == self.true_sub[:, None]).mean(0))
                gains[gg] = (self.w_sub * sub + self.lam * clu) - base
                continue
            Pc = _softmax_axis1(candC)
            p_clu = Pc[ar, self.true_clu, :]                     # (N, b)
            if self.direct:
                p_sub = _softmax_axis1(candS)[ar, self.true_sub, :]
            else:
                p_sub = np.einsum("ncb,cs->nsb", Pc, self.c2s_onehot)[ar, self.true_sub, :]
            if self.objective == "logpost":
                gains[gg] = (self.w_sub * np.log(p_sub + 1e-12).mean(0)
                             + self.lam * np.log(p_clu + 1e-12).mean(0)) - base
            else:
                gains[gg] = (self.w_sub * p_sub.mean(0) + self.lam * p_clu.mean(0)) - base
        return gains


class OverlapObjective(SelectionObjective):
    """geneBasis-style manifold-preservation objective (same maths as ``greedy_overlap``):
    the mean fraction of each cell's ``k`` panel-space neighbours that are also neighbours
    in the full-transcriptome reference kNN graph. The panel squared-distance matrix ``D``
    is maintained incrementally. ``gain_all`` is O(G * N^2), so prefer small ``overlap``
    cell counts when pairing this objective with the stochastic strategies.
    """

    def __init__(self, Xf_ov, ref_knn, k=15):
        self.t = np.log1p(np.asarray(Xf_ov, dtype="float32"))   # (N, G)
        self.N, self.n_genes = self.t.shape
        self.k = int(k)
        R = np.zeros((self.N, self.N), dtype=bool)
        R[np.arange(self.N)[:, None], ref_knn] = True
        self.R = R

    def init(self, seed=()):
        seed = list(dict.fromkeys(int(g) for g in seed))
        D = np.zeros((self.N, self.N), dtype="float32")
        for g in seed:
            d = self.t[:, g]
            D += (d[:, None] - d[None, :]) ** 2
        mask = np.zeros(self.n_genes, dtype=bool)
        mask[seed] = True
        return {"D": D, "mask": mask, "order": list(seed)}

    def clone(self, state):
        return {"D": state["D"].copy(), "mask": state["mask"].copy(),
                "order": list(state["order"])}

    def add(self, state, genes):
        if np.isscalar(genes):
            genes = [genes]
        for g in genes:
            g = int(g)
            if state["mask"][g]:
                continue
            d = self.t[:, g]
            state["D"] += (d[:, None] - d[None, :]) ** 2
            state["mask"][g] = True
            state["order"].append(g)
        return state

    def score(self, state):
        return _mean_knn_overlap(state["D"], self.R, self.k)

    def gain_one(self, state, g, base=None):
        if base is None:
            base = self.score(state)
        d = self.t[:, g]
        return _mean_knn_overlap(state["D"] + (d[:, None] - d[None, :]) ** 2,
                                 self.R, self.k) - base

    def gain_all(self, state):
        base = self.score(state)
        gains = np.full(self.n_genes, -np.inf, dtype="float32")
        for g in np.nonzero(~state["mask"])[0]:
            gains[g] = self.gain_one(state, int(g), base)
        return gains


# ---- search strategies (objective-agnostic) -----------------------------------------

def select_greedy(obj, n_select, seed_sel=(), rng=None, verbose=False,
                  label="greedy", log_every=25):
    """Lazy (CELF) forward greedy: repeatedly add the single highest-marginal-gain gene.
    The generic equivalent of ``greedy_accuracy`` / ``greedy_overlap``. ``rng`` is accepted
    for a uniform strategy signature but unused (greedy is deterministic).
    Returns (order, hist) where ``hist`` are objective scores after each addition.
    """
    import heapq

    state = obj.init(seed_sel)
    gains = obj.gain_all(state)
    heap = [(-float(gains[g]), int(g)) for g in np.nonzero(np.isfinite(gains))[0]]
    heapq.heapify(heap)
    target, hist = min(n_select, obj.n_genes), []
    while len(state["order"]) < target and heap:
        base = obj.score(state)
        while True:                                          # lazy re-evaluation
            neg, g = heapq.heappop(heap)
            if state["mask"][g]:
                continue
            gain = obj.gain_one(state, g, base)
            if not heap or gain >= -heap[0][0]:
                break
            heapq.heappush(heap, (-gain, g))
        obj.add(state, g)
        hist.append(obj.score(state))
        if verbose and len(state["order"]) % log_every == 0:
            print(f"[{label}] {len(state['order'])} genes  score={hist[-1]:.4f}",
                  flush=True)
    return list(state["order"]), hist


def _lazy_topk(obj, state, heap, top_k, base):
    """CELF generalised to the top-``k`` genes: lazily re-evaluate cached upper-bound gains
    (valid under diminishing-returns) until the ``top_k`` genes with the largest *true*
    current marginal gain are confirmed. Mutates ``heap`` (a heapified list of
    ``(-gain, gene)``); returns the confirmed ``[(gene, true_gain), ...]``, best first.
    """
    import heapq

    confirmed = []
    while len(confirmed) < top_k and heap:
        neg, g = heapq.heappop(heap)
        if state["mask"][g]:
            continue
        gain = obj.gain_one(state, g, base)
        if not heap or gain >= -heap[0][0]:
            confirmed.append((g, gain))
        else:
            heapq.heappush(heap, (-gain, g))
    return confirmed


def _limit_worker_threads():
    """ProcessPoolExecutor initializer: pin BLAS/OpenMP to one thread per worker so the
    process-level parallelism over walks doesn't oversubscribe cores (best-effort; for full
    effect also export OMP_NUM_THREADS=1 in the parent before importing numpy)."""
    import os
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def available_cpus():
    """CPUs actually usable by this process -- respects cgroup/SLURM affinity (so it returns
    the cores allotted to the job, not the machine total that ``os.cpu_count()`` reports)."""
    import os
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                       # not on Linux
        return max(1, os.cpu_count() or 1)


def _walk_once(obj, n_select, seed_sel, top_k, step_choices, start_heap, seed):
    """Run ONE random walk from a copy of the shared starting heap. Module-level (picklable)
    so it can run in a worker process. Returns ``(final_score, order, hist)``.

    Each cycle ranks the unselected genes by current marginal gain (lazy top-``k``), then
    adds ``n_add`` of the ``top_k`` chosen uniformly at random, with ``n_add`` itself drawn
    uniformly from ``step_choices`` (1, 2 or 3 picks per cycle).
    """
    import heapq

    rng = np.random.default_rng(seed)
    step_choices = np.asarray(step_choices)
    target = min(n_select, obj.n_genes)
    state = obj.init(seed_sel)
    heap = list(start_heap)                                  # copy; tuples are immutable
    hist = []
    while len(state["order"]) < target and heap:
        base = obj.score(state)
        confirmed = _lazy_topk(obj, state, heap, top_k, base)
        if not confirmed:
            break
        cands = [g for g, _ in confirmed]
        remaining = target - len(state["order"])
        n_add = int(min(rng.choice(step_choices), remaining, len(cands)))   # 1, 2 or 3
        pick = set(rng.choice(len(cands), size=n_add, replace=False).tolist())
        # return the un-chosen top-k genes to the heap with their fresh (exact) gains
        for i, (g, gain) in enumerate(confirmed):
            if i not in pick:
                heapq.heappush(heap, (-gain, g))
        obj.add(state, [cands[i] for i in pick])
        hist.append(obj.score(state))
    return obj.score(state), list(state["order"]), hist


def _walk_batch(obj, n_select, seed_sel, top_k, step_choices, start_heap, seeds):
    """Run a batch of walks in one worker (amortises the per-task pickling of ``obj``)."""
    return [_walk_once(obj, n_select, seed_sel, top_k, step_choices, start_heap, s)
            for s in seeds]


def select_stochastic_walk(obj, n_select, seed_sel=(), rng=None, n_walks=8, top_k=15,
                           step_choices=(1, 2, 3), n_jobs=1, verbose=False, label="walk"):
    """Randomized-greedy panel search with restarts. Each *walk* repeatedly:

      1. ranks all not-yet-selected genes by their marginal gain given the current panel,
      2. takes the ``top_k`` most informative, and
      3. adds ``n_add`` of them chosen uniformly at random, where ``n_add`` is drawn
         uniformly from ``step_choices`` (default 1, 2 or 3 picks per cycle),

    until the panel reaches ``n_select``; ``n_walks`` independent walks are run and the one
    with the best final objective score is returned. The randomness lets the search leave
    the single trajectory pure greedy is locked into and probe correlated-gene trade-offs
    greedy never revisits, while the ``top_k`` gate keeps every step near-optimal.

    The top-``k`` per step is found with a CELF-style lazy heap (``_lazy_topk``); the
    starting per-gene gains are identical across walks, so they are computed once and the
    heap is copied per walk. Walks are independent, so ``n_jobs > 1`` (or ``-1`` for all
    cores) runs them across processes via a ``ProcessPoolExecutor`` -- results are identical
    to the sequential run because every walk is seeded deterministically from ``rng``.
    (For best multi-core scaling set ``OMP_NUM_THREADS=1`` so BLAS doesn't oversubscribe.)

    Returns ``(order, hist, info)`` -- ``order`` is the best walk's selection order, ``hist``
    its per-cycle scores, and ``info`` carries every walk's final score and selection order.
    """
    rng = np.random.default_rng() if rng is None else rng

    # gains at the shared start are identical for every walk -> compute the heap once
    seed_state = obj.init(seed_sel)
    base_gains = obj.gain_all(seed_state)
    start_heap = sorted((-float(base_gains[g]), int(g))
                        for g in np.nonzero(np.isfinite(base_gains))[0])
    # deterministic, independent per-walk seeds (so n_jobs never changes the result)
    walk_seeds = [int(s) for s in rng.integers(0, 2**63 - 1, size=n_walks)]

    if n_jobs == 1:
        results = [_walk_once(obj, n_select, seed_sel, top_k, step_choices, start_heap, s)
                   for s in walk_seeds]
    else:
        from concurrent.futures import ProcessPoolExecutor
        n_jobs = available_cpus() if n_jobs in (-1, None) else int(n_jobs)
        n_jobs = max(1, min(n_jobs, n_walks))
        idx_chunks = [list(range(i, n_walks, n_jobs)) for i in range(n_jobs)]  # round-robin
        results = [None] * n_walks
        with ProcessPoolExecutor(max_workers=n_jobs,
                                 initializer=_limit_worker_threads) as ex:
            futs = [(idxs, ex.submit(_walk_batch, obj, n_select, seed_sel, top_k,
                                     step_choices, start_heap,
                                     [walk_seeds[j] for j in idxs]))
                    for idxs in idx_chunks if idxs]
            for idxs, f in futs:                              # scatter back into walk order
                for j, r in zip(idxs, f.result()):
                    results[j] = r

    walk_finals = np.array([r[0] for r in results])
    best_i = int(walk_finals.argmax())
    info = {"walk_finals": walk_finals, "walk_orders": [r[1] for r in results],
            "best_final": float(walk_finals[best_i]), "best_walk": best_i}
    if verbose:
        print(f"[{label}] {n_walks} walks (n_jobs={n_jobs}): best={walk_finals[best_i]:.4f} "
              f"mean={walk_finals.mean():.4f} worst={walk_finals.min():.4f}", flush=True)
    return results[best_i][1], results[best_i][2], info


def select_stochastic_greedy(obj, n_select, seed_sel=(), rng=None, eps=0.05,
                             sample_size=None, verbose=False, label="sgreedy",
                             log_every=25):
    """Lazier-than-greedy selection (Mirzasoleiman et al., 2015). Each step scores only a
    random subset of the unselected genes of size ``sample_size`` (default
    ``ceil(G/k * ln(1/eps))``) and adds the best of that subset. Retains a
    ``(1 - 1/e - eps)`` expected-quality guarantee at a fraction of greedy's evaluations,
    while injecting mild diversity. Returns (order, hist).
    """
    rng = np.random.default_rng() if rng is None else rng
    state = obj.init(seed_sel)
    target = min(n_select, obj.n_genes)
    if sample_size is None:
        k = max(1, target - len(state["order"]))
        sample_size = int(np.ceil(obj.n_genes / k * np.log(1.0 / eps)))
    hist = []
    while len(state["order"]) < target:
        avail = np.nonzero(~state["mask"])[0]
        if avail.size == 0:
            break
        subset = rng.choice(avail, size=min(sample_size, avail.size), replace=False)
        base = obj.score(state)
        gains = np.array([obj.gain_one(state, int(g), base) for g in subset])
        obj.add(state, int(subset[int(gains.argmax())]))
        hist.append(obj.score(state))
        if verbose and len(state["order"]) % log_every == 0:
            print(f"[{label}] {len(state['order'])} genes  score={hist[-1]:.4f}",
                  flush=True)
    return list(state["order"]), hist


# Strategy registry: name -> callable(obj, n_select, seed_sel=, rng=, verbose=, **kw).
# Each returns (order, hist) or (order, hist, info); ``select_panel`` normalises that.
SELECTORS = {
    "greedy": select_greedy,
    "stochastic_walk": select_stochastic_walk,
    "stochastic_greedy": select_stochastic_greedy,
}


def select_panel(objective, n_select, seed_sel=(), strategy="greedy", rng=None,
                 verbose=False, **kwargs):
    """Plug-and-play dispatcher: run ``strategy`` (a key of ``SELECTORS`` or any callable
    with the same signature) against ``objective``. Returns ``(order, hist, info)`` where
    ``info`` is ``{}`` for strategies that don't produce one.
    """
    fn = strategy if callable(strategy) else SELECTORS[strategy]
    out = fn(objective, n_select, seed_sel=seed_sel, rng=rng, verbose=verbose, **kwargs)
    if len(out) == 3:
        return out
    order, hist = out
    return order, hist, {}


def _restart_batch(strategy, obj, n_select, seed_sel, kwargs, seeds):
    """Worker: run a single-shot `strategy` once per seed; return (train_final, order) each."""
    fn = strategy if callable(strategy) else SELECTORS[strategy]
    out = []
    for s in seeds:
        res = fn(obj, n_select, seed_sel=seed_sel, rng=np.random.default_rng(int(s)), **kwargs)
        order, hist = res[0], res[1]
        final = float(hist[-1]) if len(hist) else float(obj.score(obj.init(seed_sel)))
        out.append((final, list(order)))
    return out


def best_of_restarts(strategy, obj, n_select, seed_sel=(), n_restarts=100, rng=None,
                     n_jobs=1, strategy_kwargs=None, verbose=False, label=None):
    """Run ``n_restarts`` independent runs of a single-shot stochastic ``strategy`` (e.g.
    ``"stochastic_greedy"``), parallelised over processes, and keep the run with the best
    final TRAIN objective -- the external-restart analogue of the internal restarts that
    ``stochastic_walk`` already does. Returns ``(best_order, finals, orders)`` (all runs'
    final scores and orders, for distribution / train-vs-val analysis). Determinism is
    independent of ``n_jobs`` (per-restart seeds are derived from ``rng``).
    """
    rng = np.random.default_rng() if rng is None else rng
    kwargs = dict(strategy_kwargs or {})
    seeds = [int(s) for s in rng.integers(0, 2**63 - 1, size=n_restarts)]
    label = label or (strategy if isinstance(strategy, str) else "restart")
    if n_jobs == 1:
        results = _restart_batch(strategy, obj, n_select, seed_sel, kwargs, seeds)
    else:
        from concurrent.futures import ProcessPoolExecutor
        n_jobs = available_cpus() if n_jobs in (-1, None) else int(n_jobs)
        n_jobs = max(1, min(n_jobs, n_restarts))
        idx_chunks = [list(range(i, n_restarts, n_jobs)) for i in range(n_jobs)]
        results = [None] * n_restarts
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_limit_worker_threads) as ex:
            futs = [(idxs, ex.submit(_restart_batch, strategy, obj, n_select, seed_sel,
                                     kwargs, [seeds[j] for j in idxs]))
                    for idxs in idx_chunks if idxs]
            for idxs, f in futs:
                for j, r in zip(idxs, f.result()):
                    results[j] = r
    finals = np.array([r[0] for r in results])
    orders = [r[1] for r in results]
    bi = int(np.nanargmax(finals))
    if verbose:
        print(f"[{label}] {n_restarts} restarts (n_jobs={n_jobs}): best={finals[bi]:.4f} "
              f"mean={np.nanmean(finals):.4f} worst={np.nanmin(finals):.4f}", flush=True)
    return orders[bi], finals, orders


# --------------------------------------------------------------------------------------
# Stage 5 - evaluation
# --------------------------------------------------------------------------------------

def classify_accuracy(Xf, A, B, sel, true_clu, true_sub, c2s, mask=None):
    """Hierarchical accuracy (subclass, cluster) of the NB classifier for gene set ``sel``."""
    pred = _cell_loglik(Xf, A, B, list(sel)).argmax(1)
    if mask is not None:
        pred, true_clu, true_sub = pred[mask], true_clu[mask], true_sub[mask]
    return float((c2s[pred] == true_sub).mean()), float((pred == true_clu).mean())


def classify_subsample(bundle, gene_list, efficiency=0.01, level="subclass", seed=0):
    """Classify the cached subsample cells with a named gene panel. Convenience wrapper
    (e.g. for confusion matrices in a notebook). Returns (pred_labels, true_labels, labels)
    as integer-coded arrays plus the label names for the requested ``level``.
    """
    gn = list(np.asarray(bundle["gene_names"]))
    idx = [gn.index(g) for g in gene_list if g in gn]
    rng = np.random.default_rng(seed)
    X = resample_counts(bundle["subsample_X"][:, idx].astype("int32"), efficiency, rng).astype("float32")
    A, B = nb_coeffs(bundle["cluster_means"][:, idx], efficiency)
    pred_clu = _cell_loglik(X, A, B, list(range(len(idx)))).argmax(1)
    if level == "subclass":
        c2s = bundle["cluster_to_subclass"].astype(int)
        return c2s[pred_clu], bundle["sub_subclass"].astype(int), bundle["subclass_labels"]
    return pred_clu, bundle["sub_cluster"].astype(int), bundle["cluster_labels"]


def evaluate_curve(Xf, A, B, ordered_cand, true_clu, true_sub, c2s, sizes, visp_mask=None):
    """Accuracy vs panel-size curve (overall + VISp held-out) for a candidate-index order."""
    rows = []
    for n in sizes:
        sel = ordered_cand[:n]
        sub_a, clu_a = classify_accuracy(Xf, A, B, sel, true_clu, true_sub, c2s)
        row = {"n_genes": n, "subclass_acc": sub_a, "cluster_acc": clu_a}
        if visp_mask is not None and visp_mask.any():
            sub_v, clu_v = classify_accuracy(Xf, A, B, sel, true_clu, true_sub, c2s, visp_mask)
            row["subclass_acc_visp"], row["cluster_acc_visp"] = sub_v, clu_v
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_curve_pure(Xf, Asub, Bsub, true_sub, Aclu, Bclu, true_clu, ordered_cand,
                        sizes, visp_mask=None):
    """Accuracy vs panel size using pure-level classifiers: 16-way subclass (Asub,Bsub) and
    215-way cluster (Aclu,Bclu), reported side by side (overall + VISp)."""
    rows = []
    for n in sizes:
        sel = list(ordered_cand[:n])
        ps = _cell_loglik(Xf, Asub, Bsub, sel).argmax(1)
        pc = _cell_loglik(Xf, Aclu, Bclu, sel).argmax(1)
        row = {"n_genes": n, "subclass_acc": float((ps == true_sub).mean()),
               "cluster_acc": float((pc == true_clu).mean())}
        if visp_mask is not None and visp_mask.any():
            row["subclass_acc_visp"] = float((ps[visp_mask] == true_sub[visp_mask]).mean())
            row["cluster_acc_visp"] = float((pc[visp_mask] == true_clu[visp_mask]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_selection(
    bundle,
    out_dir,
    efficiency=0.1,
    n_select=400,
    lam=0.5,
    w_sub=1.0,
    level="hierarchical",
    accuracy_objective="soft_acc",
    min_max_subclass_mean=1.0,
    min_tau=0.3,
    drop_ieg=False,
    max_candidates=2000,
    n_accuracy_cells=20000,
    overlap_cells=6000,
    n_hvg=2000,
    n_pcs=50,
    k=15,
    seed=0,
    normalize=True,
    acc_strategy="greedy",
    acc_strategy_kwargs=None,
    ovl_strategy="greedy",
    ovl_strategy_kwargs=None,
    verbose=True,
):
    """Full Stage 1-5 pipeline. Writes ``gene_ranking.csv``, ``accuracy_curve.csv`` and
    ``selection_meta.npz`` to ``out_dir`` and returns the in-memory results.

    ``max_candidates`` caps the greedy search pool to the top genes by marginal usefulness
    (conventional markers always kept). This keeps both greedy loops tractable; genes with
    low marginal MI are essentially never picked, so the cap barely affects the result.

    ``level`` sets the accuracy objective for the classification greedy (Stage 4a):
      - "hierarchical": cluster classifier, objective = w_sub*subclass_rollup + lam*cluster (default)
      - "subclass": pure 16-way subclass classifier, objective = subclass accuracy
      - "cluster":  pure 215-way cluster classifier, objective = cluster accuracy
    The candidate-pool cap and the saved curve adapt to ``level``; the curve always reports
    both the pure subclass (16-way) and pure cluster accuracies for reference.

    ``accuracy_objective`` sets the *score the Stage-4a greedy optimises* (the reported curve
    stays hard accuracy regardless): "soft_acc" (default, smooth NB-posterior relaxation of
    accuracy), "logpost" (cross-entropy), or "accuracy" (legacy hard 0/1). The smooth
    objectives avoid arbitrary tie-breaking among genes that flip no argmax at a given step.

    ``w_sub``/``lam`` weight the subclass and cluster terms of the hierarchical objective:
    (w_sub=1, lam=0) optimises subclass only, (w_sub=0, lam=1) cluster only, and the default
    (w_sub=1, lam=0.5) is balanced. (For ``level`` in {"subclass", "cluster"} the pure
    single-level classifier is used and ``w_sub`` is ignored.)

    ``acc_strategy`` / ``ovl_strategy`` independently choose the optimizer for the accuracy
    ranking (Stage 4a) and the manifold-overlap ranking (Stage 4b): ``"greedy"`` (default,
    the lazy CELF greedy ``greedy_accuracy`` / ``greedy_overlap``), ``"stochastic_walk"``
    (multiple randomized random-walk restarts; see ``select_stochastic_walk``) or
    ``"stochastic_greedy"`` -- or any callable / key registered in ``SELECTORS``. The
    ``*_strategy_kwargs`` dicts are passed through to the chosen strategy (e.g.
    ``{"n_walks": 12, "top_k": 15}``). The fused panel combines the two rankings as before.
    Note the overlap objective's gains are O(N^2) per candidate, so stochastic overlap
    strategies are markedly slower than greedy -- keep ``overlap_cells`` modest. The
    pluggable accuracy strategies use the hierarchical ``AccuracyObjective`` with ``w_sub``/
    ``lam`` derived from ``level`` (so ``level`` is honoured on that path too, except that
    "subclass" there means the cluster classifier rolled up to subclass rather than the pure
    16-way classifier).
    """
    acc_strategy_kwargs = dict(acc_strategy_kwargs or {})
    ovl_strategy_kwargs = dict(ovl_strategy_kwargs or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if normalize:
        if verbose:
            print("[run] library-size normalising the reference (per-cell depth)", flush=True)
        bundle = normalize_bundle(bundle)
    gn = np.asarray(bundle["gene_names"])

    # --- Stage 1: candidate filter ---
    cand_df = filter_candidate_genes(
        bundle, min_max_subclass_mean, min_tau, drop_ieg, verbose=verbose)
    cand_idx = cand_df.loc[cand_df["candidate"], "idx"].to_numpy()
    cand_genes = gn[cand_idx]
    conv_mask_cand = np.isin(cand_genes, CONVENTIONAL_MARKERS)

    # --- Stage 2: marginal MI ranking ---
    mu = marginal_usefulness(bundle, cand_idx, efficiency=efficiency, lam=lam, seed=seed)

    # marginal score used for the pool cap (matches the optimisation level)
    cap_key = {"subclass": "mi_subclass", "cluster": "mi_cluster"}.get(level, "marginal")

    # --- cap greedy search pool to top genes by marginal usefulness (keep conventional) ---
    if max_candidates and len(cand_idx) > max_candidates:
        top = np.argsort(mu[cap_key])[::-1][:max_candidates]
        keep = np.array(sorted(set(top.tolist()) | set(np.nonzero(conv_mask_cand)[0].tolist())))
        cand_idx = cand_idx[keep]
        cand_genes = cand_genes[keep]
        conv_mask_cand = conv_mask_cand[keep]
        mu = {key: val[keep] for key, val in mu.items()}
        if verbose:
            print(f"[run] capped greedy pool to {len(cand_idx)} candidates "
                  f"(top {max_candidates} by marginal + conventional)", flush=True)

    # --- Stage 3: reference manifold ---
    graph = build_neighbor_graph(bundle, overlap_cells, n_hvg, n_pcs, k, seed)

    # --- shared inputs for greedy ---
    Xcand = bundle["subsample_X"][:, cand_idx].astype("int32")
    Xds = resample_counts(Xcand, efficiency, rng=rng).astype("float32")
    Aclu, Bclu = nb_coeffs(bundle["cluster_means"][:, cand_idx], efficiency)
    Asub, Bsub = nb_coeffs(bundle["subclass_means"][:, cand_idx], efficiency)
    c2s = bundle["cluster_to_subclass"].astype(int)
    true_clu = bundle["sub_cluster"].astype(int)
    true_sub = bundle["sub_subclass"].astype(int)
    seed_sel = np.nonzero(conv_mask_cand)[0].tolist()

    # --- Stage 4a: accuracy selection (greedy by default, or a pluggable strategy) ---
    acc_cells = np.sort(rng.choice(
        Xds.shape[0], min(n_accuracy_cells, Xds.shape[0]), replace=False))
    if verbose:
        print(f"[run] accuracy selection (level={level}, strategy={acc_strategy}) on "
              f"{acc_cells.size} cells, {cand_idx.size} candidates", flush=True)
    acc_info = {}
    if acc_strategy == "greedy" and not acc_strategy_kwargs:
        # deterministic CELF greedy on the chosen accuracy level + objective (default path)
        if level == "subclass":
            order_acc, hist_acc = greedy_accuracy_single(
                Xds[acc_cells], Asub, Bsub, true_sub[acc_cells], seed_sel, n_select,
                objective=accuracy_objective, verbose=verbose)
        elif level == "cluster":
            order_acc, hist_acc = greedy_accuracy_single(
                Xds[acc_cells], Aclu, Bclu, true_clu[acc_cells], seed_sel, n_select,
                objective=accuracy_objective, verbose=verbose)
        else:  # hierarchical
            order_acc, hist_acc = greedy_accuracy(
                Xds[acc_cells], Aclu, Bclu, true_clu[acc_cells], true_sub[acc_cells], c2s,
                seed_sel, n_select, lam=lam, w_sub=w_sub, objective=accuracy_objective,
                verbose=verbose)
        hist_acc = np.array(hist_acc)
    else:
        # pluggable stochastic strategy on the hierarchical NB-accuracy objective
        acc_obj = AccuracyObjective(
            Xds[acc_cells], Aclu, Bclu, true_clu[acc_cells], true_sub[acc_cells], c2s,
            w_sub=w_sub, lam=lam)
        order_acc, hist_acc, acc_info = select_panel(
            acc_obj, n_select, seed_sel=seed_sel, strategy=acc_strategy, rng=rng,
            verbose=verbose, **acc_strategy_kwargs)
        hist_acc = np.array(hist_acc)

    # --- Stage 4b: overlap selection (greedy by default, or a pluggable strategy) ---
    ov = graph["overlap_idx"]
    if verbose:
        print(f"[run] overlap selection ({ovl_strategy}) on {ov.size} cells, "
              f"{cand_idx.size} candidates, k={k}", flush=True)
    ovl_info = {}
    if ovl_strategy == "greedy" and not ovl_strategy_kwargs:
        # original hand-fused path: identical output + per-gene overlap history
        order_ovl, hist_ovl = greedy_overlap(
            Xds[ov], graph["ref_knn"], seed_sel, n_select, k=k, verbose=verbose)
        hist_ovl = np.array(hist_ovl)
    else:
        ovl_obj = OverlapObjective(Xds[ov], graph["ref_knn"], k=k)
        order_ovl, hist_ovl, ovl_info = select_panel(
            ovl_obj, n_select, seed_sel=seed_sel, strategy=ovl_strategy, rng=rng,
            verbose=verbose, **ovl_strategy_kwargs)
        hist_ovl = np.array(hist_ovl)

    # --- Stage 4c: fuse ---
    fused = reciprocal_rank_fusion(order_acc, order_ovl, conv_mask_cand)
    ci = fused["cand"].to_numpy()
    fused["gene"] = cand_genes[ci]
    fused["gene_idx"] = cand_idx[ci]
    for col in ("mi_subclass", "mi_cluster", "marginal"):
        fused[col] = mu[col][ci]
    bm = cand_df.set_index("idx")
    gi = fused["gene_idx"].to_numpy()
    fused["max_subclass_mean"] = bm.loc[gi, "max_subclass_mean"].to_numpy()
    fused["tau"] = bm.loc[gi, "tau"].to_numpy()
    fused["top_subclass"] = bm.loc[gi, "top_subclass"].to_numpy()

    # --- Stage 5: evaluation curve (fused order) ---
    fused_order = fused.sort_values("fused_rank")["cand"].to_numpy()
    sizes = [s for s in (50, 100, 150, 200, 250, 300, 350, 400) if s <= len(fused_order)]
    visp = (bundle["sub_region"] == "VISp")
    curve = evaluate_curve_pure(Xds, Asub, Bsub, true_sub, Aclu, Bclu, true_clu,
                                fused_order, sizes, visp)
    if verbose:
        print("[run] accuracy curve:\n" + curve.to_string(index=False), flush=True)

    # --- save ---
    ranking = fused.sort_values("fused_rank")[[
        "fused_rank", "gene", "conventional", "rank_accuracy", "rank_overlap",
        "rrf_score", "mi_subclass", "mi_cluster", "marginal",
        "max_subclass_mean", "tau", "top_subclass"]]
    ranking.to_csv(out_dir / "gene_ranking.csv", index=False)
    curve.to_csv(out_dir / "accuracy_curve.csv", index=False)
    meta = dict(
        order_acc=cand_genes[np.array(order_acc, dtype=int)],
        order_ovl=cand_genes[np.array(order_ovl, dtype=int)],
        hist_acc=np.array(hist_acc), hist_ovl=np.array(hist_ovl),
        cand_genes=cand_genes, efficiency=efficiency, n_select=n_select,
        acc_strategy=acc_strategy, ovl_strategy=ovl_strategy,
    )
    if "walk_finals" in acc_info:                 # stochastic-walk diagnostics
        meta["walk_finals"] = acc_info["walk_finals"]
    if "walk_finals" in ovl_info:
        meta["walk_finals_ovl"] = ovl_info["walk_finals"]
    np.savez(out_dir / "selection_meta.npz", **meta)
    if verbose:
        print(f"[run] wrote outputs to {out_dir}", flush=True)
    return dict(ranking=ranking, curve=curve, cand_df=cand_df, fused=fused,
                order_acc=order_acc, order_ovl=order_ovl, graph=graph, mu=mu,
                acc_info=acc_info, ovl_info=ovl_info)


def main():
    import argparse
    p = argparse.ArgumentParser(description="BARseq2 gene-panel design (Allen 2020 reference).")
    p.add_argument("--datapath", default="/camp/home/znamenp/home/shared/resources/allen2020/")
    p.add_argument("--out_dir", default="results/panel")
    p.add_argument("--cache_dir", default="results/panel_cache",
                   help="streaming-pass cache; reused if present, else built")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--subsample_n", type=int, default=150000)
    p.add_argument("--efficiency", type=float, default=0.1,
                   help="simulated BARseq detection efficiency vs scRNAseq; key tunable "
                        "(0.01 is unusably low, ~0.1-0.2 realistic)")
    p.add_argument("--n_select", type=int, default=400)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--level", default="hierarchical",
                   choices=["hierarchical", "subclass", "cluster", "manifold"],
                   help="selection objective: pure subclass/cluster accuracy, hierarchical, "
                        "or manifold (overlap-only, geneBasis-style; skips the accuracy greedy)")
    p.add_argument("--accuracy_objective", default="soft_acc",
                   choices=["soft_acc", "logpost", "accuracy"],
                   help="Stage-4a greedy score: smooth NB-posterior soft accuracy (default), "
                        "cross-entropy (logpost), or legacy hard 0/1 accuracy")
    p.add_argument("--min_max_subclass_mean", type=float, default=1.0)
    p.add_argument("--min_tau", type=float, default=0.3)
    p.add_argument("--drop_ieg", action="store_true")
    p.add_argument("--no_normalize", action="store_true",
                   help="disable per-cell library-size normalisation of the reference")
    p.add_argument("--max_candidates", type=int, default=2000)
    p.add_argument("--n_accuracy_cells", type=int, default=20000)
    p.add_argument("--overlap_cells", type=int, default=6000)
    p.add_argument("--acc_strategy", default="greedy", choices=sorted(SELECTORS),
                   help="optimizer for the accuracy ranking (Stage 4a)")
    p.add_argument("--ovl_strategy", default="greedy", choices=sorted(SELECTORS),
                   help="optimizer for the manifold-overlap ranking (Stage 4b); the "
                   "stochastic variants are slow here (O(N^2) gains) -- keep overlap_cells low")
    p.add_argument("--n_walks", type=int, default=8,
                   help="stochastic_walk: number of independent random-walk restarts")
    p.add_argument("--walk_top_k", type=int, default=15,
                   help="stochastic_walk: pool of most-informative genes sampled each step")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="stochastic_walk: parallel worker processes for the walks "
                   "(-1 = all cores; set OMP_NUM_THREADS=1 for best scaling)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    def _walk_kwargs(strategy):
        return {"n_walks": args.n_walks, "top_k": args.walk_top_k, "n_jobs": args.n_jobs} \
            if strategy == "stochastic_walk" else {}
    acc_kwargs = _walk_kwargs(args.acc_strategy)
    ovl_kwargs = _walk_kwargs(args.ovl_strategy)

    cache = Path(args.cache_dir)
    if (cache / "bundle.npz").exists() and not args.rebuild_cache:
        print(f"[main] loading cached bundle from {cache}", flush=True)
        bundle = load_bundle(cache)
    else:
        print("[main] streaming reference (one-time pass)...", flush=True)
        bundle = load_allen2020_streaming(args.datapath, subsample_n=args.subsample_n)
        save_bundle(bundle, cache)

    run_selection(
        bundle, args.out_dir, efficiency=args.efficiency, n_select=args.n_select,
        lam=args.lam, level=args.level, accuracy_objective=args.accuracy_objective,
        min_max_subclass_mean=args.min_max_subclass_mean,
        min_tau=args.min_tau, drop_ieg=args.drop_ieg, max_candidates=args.max_candidates,
        n_accuracy_cells=args.n_accuracy_cells, overlap_cells=args.overlap_cells,
        seed=args.seed, normalize=not args.no_normalize,
        acc_strategy=args.acc_strategy, acc_strategy_kwargs=acc_kwargs,
        ovl_strategy=args.ovl_strategy, ovl_strategy_kwargs=ovl_kwargs)


if __name__ == "__main__":
    main()
