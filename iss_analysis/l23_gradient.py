"""Evaluate how well a gene panel reconstructs the L2/3 transcriptomic gradient in V1.

Ground truth: the continuous transcriptomic continuum of L2/3 IT neurons in primary visual
cortex (VISp / V1), as defined by the **286 L2/3 cell-type-identity genes** of Xie et al.
2025 (*PNAS* 122(7):e2421022122; identity set originally Cheng et al. 2022, *Cell*
185:311-327). Reference cells = the 3,426 VISp ``L2/3 IT CTX`` neurons from Allen 2020,
extracted once into ``results/l23_reference/visp_l23_cells.npz`` (raw counts, all genes).

The reference manifold is the identity-gene PCA (PC1 = the dominant A->B->C continuum axis,
PC1-PC2 = the triangular archetype plane, plus a root-free diffusion component). Each panel
is scored by how well its **BARseq-down-sampled (eff=0.1) measurements** recover that
manifold:

  * coordinate recovery  - cross-validated Ridge regression of the reference coordinate on
                           the panel PCs (Spearman rho, R^2): "can the panel place each cell
                           on the continuum?"
  * kNN-graph overlap    - geneBasis-style local-neighbourhood preservation (reuses
                           ``panel_design._mean_knn_overlap``)
  * Procrustes (2D)      - shape match of the panel vs reference triangle plane
  * trustworthiness /    - false- vs missed-neighbour rates (sklearn.manifold)
    continuity
  * distance Spearman    - global geometry: pairwise-distance rank correlation (panel vs ref)

Loaded as a standalone module (the ``iss_analysis`` package ``__init__`` does not import in
this env): ``sys.path.insert(0, ".../iss_analysis"); import l23_gradient``.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_design as pdsn  # resample_counts, _mean_knn_overlap, filter_candidate_genes

REF_DIR = Path("results/l23_reference")
CELLS_NPZ = REF_DIR / "visp_l23_cells.npz"
IDENTITY_CSV = REF_DIR / "identity_genes_286.csv"


# --------------------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------------------
def load_reference_cells(path=CELLS_NPZ):
    """Load the cached VISp L2/3 IT cells (raw counts, all Allen genes)."""
    z = np.load(path, allow_pickle=True)
    gene_names = np.array([str(g) for g in z["gene_names"]])
    return dict(
        X=z["X"],  # (n_cells, n_genes) raw UMI counts
        gene_names=gene_names,
        gindex={g: i for i, g in enumerate(gene_names)},
        cluster_label=np.array([str(c) for c in z["cluster_label"]]),
        sample_name=np.array([str(s) for s in z["sample_name"]]),
    )


def load_identity_genes(ref, path=IDENTITY_CSV):
    """Return (present_genes, full_dataframe). ``present`` = identity genes in the vocab."""
    df = pd.read_csv(path)
    present = [g for g in df["gene"].tolist() if g in ref["gindex"]]
    return present, df


def random_pool(min_expr=0.85, min_tau=0.3, cache="results/panel_cache/bundle.npz"):
    """Biophysically-filtered candidate pool (same gate as the selections), drawn from the
    small Stage-0 ``bundle.npz`` only (no need for the 7.4 GB subsample matrix)."""
    z = np.load(cache, allow_pickle=True)
    b = {k: z[k] for k in ("gene_names", "subclass_means", "subclass_frac", "subclass_labels")}
    cand = pdsn.filter_candidate_genes(b, min_max_subclass_mean=min_expr, min_tau=min_tau,
                                       verbose=False)
    return cand.loc[cand["passes_biophysical"], "gene"].tolist()


# --------------------------------------------------------------------------------------
# embedding (mirrors panel_plots.embed preprocessing, minus UMAP)
# --------------------------------------------------------------------------------------
def _preprocess(counts):
    """normalize_total(1e4) -> log1p -> per-gene z-score (clip +/-10). counts: cells x genes."""
    X = counts.astype("float32")
    tot = X.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    X = np.log1p(X / tot * 1e4)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return np.clip((X - mu) / sd, -10.0, 10.0)


def embed_pca(ref, genes, eff=0.1, n_pcs=30, seed=0):
    """PCA embedding of (optionally BARseq-down-sampled) panel-gene expression.

    Returns (pcs[n_cells, n_comp], n_present)."""
    from sklearn.decomposition import PCA

    cols = [ref["gindex"][g] for g in genes if g in ref["gindex"]]
    n_present = len(cols)
    if n_present < 2:
        raise ValueError(f"panel has <2 genes present in the reference vocab ({n_present})")
    Xs = ref["X"][:, cols].astype("int32")
    if eff < 1.0:
        Xs = pdsn.resample_counts(Xs, eff, np.random.default_rng(seed))
    Xp = _preprocess(Xs)
    n_comp = int(min(n_pcs, n_present - 1, Xp.shape[0] - 1))
    pcs = PCA(n_components=n_comp, random_state=seed).fit_transform(Xp)
    return pcs, n_present


# --------------------------------------------------------------------------------------
# reference gradient
# --------------------------------------------------------------------------------------
def _cluster_num(cluster_label):
    """Numeric prefix of an Allen cluster label (e.g. '168_L2/3 IT CTX' -> 168)."""
    out = np.empty(len(cluster_label), dtype=float)
    for i, c in enumerate(cluster_label):
        try:
            out[i] = float(str(c).split("_")[0])
        except ValueError:
            out[i] = np.nan
    return out


def _diffusion_coord(pcs, k=15, seed=0):
    """First non-trivial diffusion component (root-free continuum coordinate)."""
    try:
        import anndata as ad
        import scanpy as sc

        warnings.filterwarnings("ignore")
        sc.settings.verbosity = 0
        a = ad.AnnData(np.zeros((pcs.shape[0], 1), dtype="float32"))
        a.obsm["X_pca"] = pcs.astype("float32")
        sc.pp.neighbors(a, n_neighbors=k, use_rep="X_pca", random_state=seed)
        sc.tl.diffmap(a, n_comps=5)
        return a.obsm["X_diffmap"][:, 1].copy()
    except Exception as e:  # pragma: no cover - diffusion is a secondary coordinate
        print(f"[l23] diffusion coordinate failed ({e}); using PC1 instead", flush=True)
        return pcs[:, 0].copy()


def build_reference(ref, identity_genes, n_pcs=15, k=15, seed=0, add_diffusion=True):
    """Build the ground-truth L2/3 gradient from the identity genes (full depth)."""
    from sklearn.neighbors import NearestNeighbors

    pcs, n_present = embed_pca(ref, identity_genes, eff=1.0, n_pcs=n_pcs, seed=seed)
    clu_num = _cluster_num(ref["cluster_label"])
    # deterministic PC1 orientation: positive correlation with cluster numeric order
    good = np.isfinite(clu_num)
    if np.corrcoef(pcs[good, 0], clu_num[good])[0, 1] < 0:
        pcs = pcs.copy()
        pcs[:, 0] *= -1.0
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pcs)
    _, idx = nn.kneighbors(pcs)
    ref_knn = idx[:, 1:]
    N = pcs.shape[0]
    R = np.zeros((N, N), dtype=bool)
    R[np.arange(N)[:, None], ref_knn] = True
    out = dict(pcs=pcs, coord1d=pcs[:, 0].copy(), coord2d=pcs[:, :2].copy(),
               R=R, k=k, n_genes=n_present, clu_num=clu_num)
    if add_diffusion:
        out["dc1"] = _diffusion_coord(pcs, k=k, seed=seed)
    return out


def get_reference(ref=None, identity_genes=None, cache=REF_DIR / "reference_gradient.npz",
                  rebuild=False, **kw):
    """Build the reference gradient once and cache it to ``cache`` (the build pays a one-time
    ~2 min scanpy/numba JIT for the diffusion coordinate). Pass ``rebuild=True`` to refresh."""
    cache = Path(cache)
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] if z[k].shape else z[k].item() for k in z.files}
    if ref is None:
        ref = load_reference_cells()
    if identity_genes is None:
        identity_genes, _ = load_identity_genes(ref)
    refb = build_reference(ref, identity_genes, **kw)
    np.savez_compressed(cache, **refb)
    return refb


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
def _sq_dists(P):
    """Pairwise squared-Euclidean distance matrix from coordinates P (n x d)."""
    G = P @ P.T
    sq = np.diag(G).copy()
    D = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(D, 0.0, out=D)
    return D.astype("float32")


def _dist_spearman(A, B, max_pairs=1_000_000, seed=0):
    """Spearman correlation of pairwise distances between embeddings A and B."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    da = pdist(A)
    db = pdist(B)
    if da.size > max_pairs:
        sel = np.random.default_rng(seed).choice(da.size, max_pairs, replace=False)
        da, db = da[sel], db[sel]
    return float(spearmanr(da, db).correlation)


def _cv_ridge_predict(X, y, n_splits=5, alpha=1.0, seed=0):
    """K-fold cross-validated ridge prediction of ``y`` from ``X`` (closed-form, numpy only;
    avoids the sklearn/scipy ``sym_pos`` incompatibility in this env)."""
    n, p = X.shape
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, n_splits)
    eye = alpha * np.eye(p)
    pred = np.empty(n, dtype=float)
    for f in range(n_splits):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != f])
        Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(0)
        Xc = Xtr - mu
        ymu = ytr.mean()
        w = np.linalg.solve(Xc.T @ Xc + eye, Xc.T @ (ytr - ymu))
        pred[te] = (X[te] - mu) @ w + ymu
    return pred


def score_panel(ref, refb, genes, eff=0.1, n_pcs=30, seed=0, n_splits=5,
                max_pairs=1_000_000):
    """Score one panel (gene list) against the reference gradient ``refb``."""
    from scipy.spatial import procrustes
    from scipy.stats import spearmanr
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import r2_score

    pcs, n_present = embed_pca(ref, genes, eff=eff, n_pcs=n_pcs, seed=seed)
    refpcs = refb["pcs"]
    k = refb["k"]

    def recover(y):
        pred = _cv_ridge_predict(pcs, y, n_splits=n_splits, alpha=1.0, seed=seed)
        return float(spearmanr(y, pred).correlation), float(r2_score(y, pred))

    sp1, r2_1 = recover(refb["coord1d"])
    r2_2d = float(np.mean([recover(refb["coord2d"][:, j])[1] for j in range(2)]))
    sp_dc, r2_dc = recover(refb["dc1"]) if "dc1" in refb else (np.nan, np.nan)

    knn_ov = float(pdsn._mean_knn_overlap(_sq_dists(pcs), refb["R"], k))
    proc_sim = float(1.0 - procrustes(refb["coord2d"], pcs[:, :2])[2])
    trust = float(trustworthiness(refpcs, pcs, n_neighbors=k))
    cont = float(trustworthiness(pcs, refpcs, n_neighbors=k))
    dist_sp = _dist_spearman(refpcs, pcs, max_pairs=max_pairs, seed=seed)

    return dict(n_genes_present=n_present, coord_spearman=sp1, coord_r2=r2_1,
                coord2d_r2=r2_2d, dc_spearman=sp_dc, dc_r2=r2_dc, knn_overlap=knn_ov,
                procrustes_sim=proc_sim, trustworthiness=trust, continuity=cont,
                dist_spearman=dist_sp)


def score_panel_reps(ref, refb, genes, eff=0.1, n_pcs=30, seeds=(0, 1), **kw):
    """Mean of ``score_panel`` over several down-sampling seeds (eff<1 is stochastic)."""
    recs = [score_panel(ref, refb, genes, eff=eff, n_pcs=n_pcs, seed=s, **kw) for s in seeds]
    keys = recs[0].keys()
    return {key: float(np.mean([r[key] for r in recs])) for key in keys}


# headline first, then supporting metrics
METRICS = ["coord_spearman", "knn_overlap", "coord_r2", "coord2d_r2", "dc_spearman",
           "procrustes_sim", "trustworthiness", "continuity", "dist_spearman"]
METRIC_LABELS = {
    "coord_spearman": "PC1 continuum recovery (Spearman rho)",
    "knn_overlap": "kNN-graph overlap",
    "coord_r2": "PC1 continuum recovery (R^2)",
    "coord2d_r2": "2D triangle recovery (mean R^2)",
    "dc_spearman": "Diffusion-axis recovery (Spearman rho)",
    "procrustes_sim": "2D Procrustes similarity (1 - disparity)",
    "trustworthiness": "Trustworthiness (few false neighbours)",
    "continuity": "Continuity (few missed neighbours)",
    "dist_spearman": "Pairwise-distance Spearman (global geometry)",
}
