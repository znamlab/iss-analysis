"""Streamlined cross-dataset / cross-panel evaluation for BARseq gene-panel design.

Standardises the two references into a common form so any gene panel can be evaluated on
any dataset with one call, and so the panel_design selection pipeline can be run on either.

Datasets
--------
* allen2020 : Yao 2021 neocortex, 10x (cached bundle in results/panel_cache)
* allen2018 : Tasic VISp, SMART-seq. SMART-seq is ~100x deeper than 10x, so counts are
  rescaled to match the Allen-2020 median per-cell total -> both datasets then use the
  SAME efficiency (eff=0.1) and the same absolute filters, making comparisons symmetric.

Self-contained except for `panel_design` (loaded as a sibling top-level module).
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_design as pdsn  # noqa: E402

CALL = "/camp/home/znamenp/home/users/znamenp/code/iss-preprocess/iss_preprocess/call/"
A2018_DIR = "/camp/home/znamenp/home/shared/resources/allen2018/"
CACHE2020 = "results/panel_cache"
CACHE2018 = "results/panel_cache_2018"

ROLONY = ['Rcan2','Pvalb','Olfm3','Prss23','Pantr1','Stxbp6','Chn2','Nov','Cpne6','Fst','Gpx3',
'Hpcal4','Serpine2','Dkk3','Cartpt','Rspo1','Cxcl14','Lypd6b','Vip','Cryab','Thsd7a','Mdh1','Nefl',
'Lypd6','Rbp4','Spon1','Cdh13','Sparcl1','Spock3','Cd24a','Snca','Rgs10','Gad1','Chrm2','Gap43',
'Etv1','Itm2c','Kcnab1','Cxcl12','Myl4','Arpp21','Nnat','Brinp3','Cplx3','Pcdh8','Pcp4l1','Cnr1',
'Stmn2','Nrep','Tac2','Sst','Synpr','Pdyn','Calb2','Enpp2','Id2','Igfbp4','Lamp5','Marcksl1','Crh',
'Ncald','Npy','Nr4a2','Nrsn1','Pcp4','Pde1a','Gabra1','Penk','Ptn','Rab3b','Reln','Rgs4','Cck','Scg2',
'Serpini1','Calb1']


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def group_means(X, ids, n):
    """Per-group mean of rows of X (cells x genes), groups indexed by ids in [0, n)."""
    M = np.zeros((n, X.shape[1]), dtype=np.float64)
    cnt = np.zeros(n)
    np.add.at(M, ids, X.astype(np.float64))
    np.add.at(cnt, ids, 1.0)
    return M / np.maximum(cnt, 1)[:, None]


def resolve_genes(raw, vocab):
    """Map (possibly compound a_b_c) gene names to those present in vocab (a set)."""
    out = []
    for g in raw:
        for p in [g] + str(g).split("_"):
            if p in vocab:
                out.append(p)
                break
    return list(dict.fromkeys(out))


def reference_panels(vocab):
    """Published reference panels resolved against a gene vocabulary."""
    def _cb(path):
        return resolve_genes(pd.read_csv(path, header=None).iloc[:, 2].astype(str).tolist(), vocab)
    return {
        "rolony/BARseq2": [g for g in ROLONY if g in vocab],
        "codebook_88": _cb(CALL + "codebook_88_20230216.csv"),
        "csm_and_v1": _cb(CALL + "csm_and_v1_codebook_20250515.csv"),
    }


# --------------------------------------------------------------------------------------
# dataset loaders -> standard dict
# --------------------------------------------------------------------------------------

def _standardize(name, X, gene_names, true_sub, true_clu, c2s, sub_labels, clu_labels, region):
    X = pdsn.normalize_library(X)            # per-cell library-size normalisation (depth)
    return dict(name=name, X=X, gene_names=np.asarray(gene_names),
                gindex={g: i for i, g in enumerate(gene_names)},
                true_subclass=np.asarray(true_sub, int), true_cluster=np.asarray(true_clu, int),
                c2s=np.asarray(c2s, int), sub_labels=np.asarray(sub_labels),
                clu_labels=np.asarray(clu_labels), region=np.asarray(region))


def load_allen2020():
    b = pdsn.load_bundle(CACHE2020)
    return _standardize("allen2020", b["subsample_X"], b["gene_names"],
                        b["sub_subclass"], b["sub_cluster"], b["cluster_to_subclass"],
                        b["subclass_labels"], b["cluster_labels"], b["sub_region"])


def build_allen2018_cache(target_median_total, cache=CACHE2018):
    """Read the Tasic VISp CSVs once, filter neurons, rescale depth to
    ``target_median_total`` (to match 10x), and cache as npy/npz."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(A2018_DIR + "mouse_VISp_2018-06-14_samples-columns.csv",
                       low_memory=False).set_index("sample_name")
    genes = pd.read_csv(A2018_DIR + "mouse_VISp_2018-06-14_genes-rows.csv")["gene_symbol"].astype(str).values
    exons = pd.read_csv(A2018_DIR + "mouse_VISp_2018-06-14_exon-matrix.csv", index_col=0)
    exons.index = genes
    keep = meta["class"].isin(["GABAergic", "Glutamatergic"])
    for bad in ["ALM", "Doublet", "Batch", "Low Quality", "No Class"]:
        keep &= ~meta["cluster"].astype(str).str.contains(bad)
    keep &= ~meta["subclass"].astype(str).str.contains("High Intronic")
    meta = meta[keep]
    cells = [s for s in exons.columns if s in meta.index]
    meta = meta.loc[cells]
    X = exons.loc[:, cells].to_numpy().T.astype(np.float64)        # cells x genes
    # rescale each cell so the median per-cell total matches the 10x target
    tot = X.sum(1)
    scale = target_median_total / np.median(tot)
    X = np.rint(X * scale).clip(0, 32767).astype(np.int16)
    sub_labels = np.array(sorted(meta["subclass"].unique()))
    clu_labels = np.array(sorted(meta["cluster"].unique()))
    s2i = {s: i for i, s in enumerate(sub_labels)}; c2i = {c: i for i, c in enumerate(clu_labels)}
    true_sub = meta["subclass"].map(s2i).to_numpy()
    true_clu = meta["cluster"].map(c2i).to_numpy()
    c2s = np.empty(len(clu_labels), int)
    for c, s in meta[["cluster", "subclass"]].drop_duplicates().itertuples(index=False):
        c2s[c2i[c]] = s2i[s]
    np.save(cache / "X.npy", X)
    np.savez(cache / "meta.npz", gene_names=exons.index.to_numpy(), true_sub=true_sub,
             true_clu=true_clu, c2s=c2s, sub_labels=sub_labels, clu_labels=clu_labels,
             scale=scale)
    print(f"[allen2018] cached {X.shape} cells x genes (depth scale={scale:.4g})", flush=True)


def load_allen2018(cache=CACHE2018):
    cache = Path(cache)
    X = np.load(cache / "X.npy")
    m = np.load(cache / "meta.npz", allow_pickle=True)
    return _standardize("allen2018", X, m["gene_names"], m["true_sub"], m["true_clu"],
                        m["c2s"], m["sub_labels"], m["clu_labels"],
                        np.array(["VISp"] * X.shape[0]))


def load_dataset(name):
    return load_allen2020() if name == "allen2020" else load_allen2018()


# --------------------------------------------------------------------------------------
# make a panel_design-compatible bundle (so run_selection works on any dataset)
# --------------------------------------------------------------------------------------

def make_bundle(ds):
    X = ds["X"]
    nsub, nclu = len(ds["sub_labels"]), len(ds["clu_labels"])
    frac = np.zeros((nsub, X.shape[1])); scnt = np.zeros(nsub)
    np.add.at(frac, ds["true_subclass"], (X > 0).astype(np.float64))
    np.add.at(scnt, ds["true_subclass"], 1.0)
    ccnt = np.zeros(nclu); np.add.at(ccnt, ds["true_cluster"], 1.0)
    return dict(
        gene_names=ds["gene_names"], subsample_X=X,
        cluster_means=group_means(X, ds["true_cluster"], nclu),
        subclass_means=group_means(X, ds["true_subclass"], nsub),
        subclass_frac=frac / np.maximum(scnt, 1)[:, None],
        cluster_cnt=ccnt, subclass_cnt=scnt, cluster_to_subclass=ds["c2s"],
        sub_subclass=ds["true_subclass"], sub_cluster=ds["true_cluster"], sub_region=ds["region"],
        subclass_labels=ds["sub_labels"], cluster_labels=ds["clu_labels"], n_kept=X.shape[0])


# --------------------------------------------------------------------------------------
# evaluate any panel on any dataset (within-dataset NB classifier)
# --------------------------------------------------------------------------------------

def evaluate_panel(ds, genes, efficiency=0.1, level="subclass", seed=0):
    cols = [ds["gindex"][g] for g in genes if g in ds["gindex"]]
    rng = np.random.default_rng(seed)
    ids = ds["true_subclass"] if level == "subclass" else ds["true_cluster"]
    n = len(ds["sub_labels"]) if level == "subclass" else len(ds["clu_labels"])
    means = group_means(ds["X"][:, cols], ids, n)
    Xs = pdsn.resample_counts(ds["X"][:, cols].astype("int32"), efficiency, rng).astype("float32")
    A, B = pdsn.nb_coeffs(means, efficiency)
    pred = pdsn._cell_loglik(Xs, A, B, list(range(len(cols)))).argmax(1)
    return float(np.mean(pred == ids)), len(cols)
