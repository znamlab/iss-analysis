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
    tot = X.sum(1, keepdims=True).astype(np.float64)
    target = float(np.median(tot)) if target is None else float(target)
    tot[tot == 0] = 1.0
    return np.minimum(np.rint(X * (target / tot)), 32767).astype(np.int16)


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


def greedy_accuracy(
    Xf, A, B, true_clu, true_sub, c2s, seed_sel, n_select,
    lam=0.5, verbose=True,
):
    """Lazy (CELF) forward greedy maximising hierarchical accuracy
    ``subclass_acc + lam*cluster_acc``.

    Uses cached marginal gains in a max-heap: because marginal accuracy gains diminish as
    the panel grows, stale gains are upper bounds, so each step only re-evaluates the few
    top candidates instead of all of them. Each evaluation is a cheap (N, n_clu) argmax.

    Xf: (N, G) float32 down-sampled counts on candidate genes.
    A, B: (n_clu, G) NB coefficients on candidate genes.
    seed_sel: candidate-space indices to force-include first (conventional markers).
    Returns: order (list of candidate indices), history of (sub_acc, clu_acc).
    """
    import heapq

    N, G = Xf.shape
    selected = list(dict.fromkeys(seed_sel))
    sel_mask = np.zeros(G, dtype=bool)
    sel_mask[selected] = True
    Lc = _cell_loglik(Xf, A, B, selected).copy()

    def score_with(g):
        pred = (Lc + np.outer(Xf[:, g], A[:, g]) + B[:, g]).argmax(1)
        sub = float((c2s[pred] == true_sub).mean())
        clu = float((pred == true_clu).mean())
        return sub + lam * clu, sub, clu

    def cur_score():
        pred = Lc.argmax(1)
        return float((c2s[pred] == true_sub).mean()) + lam * float((pred == true_clu).mean())

    base = cur_score()
    heap = []                                          # (-marginal_gain, g, sub, clu)
    for g in np.nonzero(~sel_mask)[0]:
        s, sub, clu = score_with(int(g))
        heapq.heappush(heap, (-(s - base), int(g), sub, clu))

    order, hist = list(selected), []
    while len(selected) < min(n_select, G) and heap:
        base = cur_score()
        while True:                                    # lazy re-evaluation
            neg_gain, g, sub, clu = heapq.heappop(heap)
            if sel_mask[g]:
                continue
            s, sub, clu = score_with(g)
            gain = s - base
            if not heap or gain >= -heap[0][0]:
                break
            heapq.heappush(heap, (-gain, g, sub, clu))
        selected.append(g)
        sel_mask[g] = True
        order.append(g)
        Lc += np.outer(Xf[:, g], A[:, g]) + B[:, g]
        hist.append((sub, clu))
        if verbose and len(selected) % 25 == 0:
            print(f"[greedy-acc] {len(selected)} genes  sub={sub:.3f} clu={clu:.3f}",
                  flush=True)
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


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_selection(
    bundle,
    out_dir,
    efficiency=0.1,
    n_select=400,
    lam=0.5,
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
    verbose=True,
):
    """Full Stage 1-5 pipeline. Writes ``gene_ranking.csv``, ``accuracy_curve.csv`` and
    ``selection_meta.npz`` to ``out_dir`` and returns the in-memory results.

    ``max_candidates`` caps the greedy search pool to the top genes by marginal usefulness
    (conventional markers always kept). This keeps both greedy loops tractable; genes with
    low marginal MI are essentially never picked, so the cap barely affects the result.
    """
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

    # --- cap greedy search pool to top genes by marginal usefulness (keep conventional) ---
    if max_candidates and len(cand_idx) > max_candidates:
        top = np.argsort(mu["marginal"])[::-1][:max_candidates]
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
    A, B = nb_coeffs(bundle["cluster_means"][:, cand_idx], efficiency)
    c2s = bundle["cluster_to_subclass"].astype(int)
    true_clu = bundle["sub_cluster"].astype(int)
    true_sub = bundle["sub_subclass"].astype(int)
    seed_sel = np.nonzero(conv_mask_cand)[0].tolist()

    # --- Stage 4a: accuracy greedy (on a cell subsample) ---
    acc_cells = np.sort(rng.choice(
        Xds.shape[0], min(n_accuracy_cells, Xds.shape[0]), replace=False))
    if verbose:
        print(f"[run] greedy accuracy on {acc_cells.size} cells, "
              f"{cand_idx.size} candidates, {A.shape[0]} clusters", flush=True)
    order_acc, hist_acc = greedy_accuracy(
        Xds[acc_cells], A, B, true_clu[acc_cells], true_sub[acc_cells], c2s,
        seed_sel, n_select, lam=lam, verbose=verbose)

    # --- Stage 4b: overlap greedy (on overlap cells) ---
    ov = graph["overlap_idx"]
    order_ovl, hist_ovl = greedy_overlap(
        Xds[ov], graph["ref_knn"], seed_sel, n_select, k=k, verbose=verbose)

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
    curve = evaluate_curve(Xds, A, B, fused_order, true_clu, true_sub, c2s, sizes, visp)
    if verbose:
        print("[run] accuracy curve:\n" + curve.to_string(index=False), flush=True)

    # --- save ---
    ranking = fused.sort_values("fused_rank")[[
        "fused_rank", "gene", "conventional", "rank_accuracy", "rank_overlap",
        "rrf_score", "mi_subclass", "mi_cluster", "marginal",
        "max_subclass_mean", "tau", "top_subclass"]]
    ranking.to_csv(out_dir / "gene_ranking.csv", index=False)
    curve.to_csv(out_dir / "accuracy_curve.csv", index=False)
    np.savez(
        out_dir / "selection_meta.npz",
        order_acc=cand_genes[np.array(order_acc, dtype=int)],
        order_ovl=cand_genes[np.array(order_ovl, dtype=int)],
        hist_acc=np.array(hist_acc), hist_ovl=np.array(hist_ovl),
        cand_genes=cand_genes, efficiency=efficiency, n_select=n_select,
    )
    if verbose:
        print(f"[run] wrote outputs to {out_dir}", flush=True)
    return dict(ranking=ranking, curve=curve, cand_df=cand_df, fused=fused,
                order_acc=order_acc, order_ovl=order_ovl, graph=graph, mu=mu)


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
    p.add_argument("--min_max_subclass_mean", type=float, default=1.0)
    p.add_argument("--min_tau", type=float, default=0.3)
    p.add_argument("--drop_ieg", action="store_true")
    p.add_argument("--no_normalize", action="store_true",
                   help="disable per-cell library-size normalisation of the reference")
    p.add_argument("--max_candidates", type=int, default=2000)
    p.add_argument("--n_accuracy_cells", type=int, default=20000)
    p.add_argument("--overlap_cells", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

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
        lam=args.lam, min_max_subclass_mean=args.min_max_subclass_mean,
        min_tau=args.min_tau, drop_ieg=args.drop_ieg, max_candidates=args.max_candidates,
        n_accuracy_cells=args.n_accuracy_cells, overlap_cells=args.overlap_cells,
        seed=args.seed, normalize=not args.no_normalize)


if __name__ == "__main__":
    main()
