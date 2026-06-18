"""Reusable plotting / evaluation helpers for BARseq gene-panel analysis, consolidating
the previously ad-hoc result scripts. Driven by notebooks/cell_types.ipynb.

Self-contained except for the sibling top-level modules ``panel_design`` and ``panel_eval``
(load via sys.path.insert(0, ".../iss_analysis")).

Conventions
-----------
* ``ds`` is a standardised dataset dict from ``panel_eval.load_dataset`` /
  ``load_allen2020`` / ``load_allen2018`` (keys: X, gene_names, gindex, true_subclass,
  true_cluster, c2s, sub_labels, clu_labels, region).
* ``panels`` is a dict {name: ordered_gene_list}.
* All classification simulates BARseq by binomial down-sampling at ``eff`` and uses the NB
  classifier (means estimated within the evaluated dataset).
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_design as pdsn          # noqa: E402
import panel_eval as pe              # noqa: E402

INH = {"Lamp5", "Pvalb", "Sncg", "Sst", "Sst Chodl", "Vip"}


# --------------------------------------------------------------------------------------
# loading panels
# --------------------------------------------------------------------------------------

def load_ranking(out_dir):
    """Read a selection output dir -> ordered gene list (by fused_rank) + the table."""
    df = pd.read_csv(Path(out_dir) / "gene_ranking.csv").sort_values("fused_rank")
    return df["gene"].tolist(), df


def standard_panels(out_dir, ds_vocab):
    """Convenience: {classification, mapping, combined} from a selection dir."""
    df = pd.read_csv(Path(out_dir) / "gene_ranking.csv")
    return {
        "classification": df.dropna(subset=["rank_accuracy"]).sort_values("rank_accuracy")["gene"].tolist(),
        "mapping":        df.dropna(subset=["rank_overlap"]).sort_values("rank_overlap")["gene"].tolist(),
        "combined":       df.sort_values("fused_rank")["gene"].tolist(),
    }


# --------------------------------------------------------------------------------------
# classification helpers
# --------------------------------------------------------------------------------------

def classify(ds, genes, eff=0.1, level="subclass", seed=0):
    """Return (pred, true) integer label arrays for all cells of ``ds``."""
    cols = [ds["gindex"][g] for g in genes if g in ds["gindex"]]
    ids = ds["true_subclass"] if level == "subclass" else ds["true_cluster"]
    n = len(ds["sub_labels"]) if level == "subclass" else len(ds["clu_labels"])
    means = pe.group_means(ds["X"][:, cols], ids, n)
    Xs = pdsn.resample_counts(ds["X"][:, cols].astype("int32"), eff,
                              np.random.default_rng(seed)).astype("float32")
    A, B = pdsn.nb_coeffs(means, eff)
    pred = pdsn._cell_loglik(Xs, A, B, list(range(len(cols)))).argmax(1)
    return pred, ids


# --------------------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------------------

def embed(ds, genes, eff=0.1, cells=None, with_leiden=False, n_hvg=None, seed=0):
    """UMAP of (down-sampled) panel-gene expression. ``cells`` = optional row index."""
    import scanpy as sc, anndata as ad
    warnings.filterwarnings("ignore")
    sc.settings.verbosity = 0
    cols = [ds["gindex"][g] for g in genes if g in ds["gindex"]]
    X = ds["X"] if cells is None else ds["X"][cells]
    Xs = pdsn.resample_counts(X[:, cols].astype("int32"), eff, np.random.default_rng(seed)).astype("float32")
    a = ad.AnnData(Xs)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=min(50, len(cols) - 1))
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(50, len(cols) - 1))
    sc.tl.umap(a)
    leid = None
    if with_leiden:
        sc.tl.leiden(a, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
        leid = a.obs["leiden"].astype(int).values
    return a.obsm["X_umap"], leid


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

def selection_curves(out_dir, ax=None):
    """Accuracy + manifold-overlap vs panel size from a selection output dir."""
    out_dir = Path(out_dir)
    curve = pd.read_csv(out_dir / "accuracy_curve.csv")
    meta = np.load(out_dir / "selection_meta.npz", allow_pickle=True)
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(curve["n_genes"], curve["subclass_acc"], "-o", label="subclass")
    ax[0].plot(curve["n_genes"], curve["cluster_acc"], "-o", label="cluster")
    if "subclass_acc_visp" in curve:
        ax[0].plot(curve["n_genes"], curve["subclass_acc_visp"], "--s", label="subclass (VISp)")
        ax[0].plot(curve["n_genes"], curve["cluster_acc_visp"], "--s", label="cluster (VISp)")
    ax[0].set(xlabel="# genes", ylabel="accuracy", title="classification accuracy")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(np.arange(len(meta["hist_ovl"])) + 1, meta["hist_ovl"], color="C2")
    ax[1].set(xlabel="# genes", ylabel="kNN-graph overlap", title="manifold preservation")
    ax[1].grid(alpha=.3)
    return curve


def expression_across_subclasses(ds, genes, ax=None, max_genes=60):
    """Box per gene = mean expression across subclasses (red = peak subclass). Ranked order."""
    sub_means = pe.group_means(ds["X"], ds["true_subclass"], len(ds["sub_labels"]))
    genes = [g for g in genes if g in ds["gindex"]][:max_genes]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.22 * len(genes) + 1))
    data = [sub_means[:, ds["gindex"][g]] for g in genes]
    pos = np.arange(len(genes))[::-1]
    ax.boxplot(data, positions=pos, vert=False, widths=.6, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#cfe2f3", edgecolor="#3a6ea5", lw=.5),
               medianprops=dict(color="#1f3b5b", lw=.8))
    for p, g in zip(pos, genes):
        v = sub_means[:, ds["gindex"][g]]
        ax.plot(v.max(), p, "o", color="crimson", ms=3)
    ax.set_yticks(pos); ax.set_yticklabels([f"{i+1} {g}" for i, g in enumerate(genes)], fontsize=5)
    ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("mean expression / cell")
    ax.grid(axis="x", alpha=.25)


def per_gene_boxplots(ds, genes, pdf_path, per_page=12, cap=4000, eff=None):
    """One panel per gene: subclasses on x, single-cell expression on y. Multi-page PDF.
    eff=None plots raw counts; otherwise down-samples first."""
    from matplotlib.backends.backend_pdf import PdfPages
    labels = [s.replace(" CTX", "") for s in ds["sub_labels"]]
    n = len(labels)
    rng = np.random.default_rng(0)
    cby = [(lambda c: rng.choice(c, cap, False) if len(c) > cap else c)(
        np.where(ds["true_subclass"] == s)[0]) for s in range(n)]
    cols = [(ds["gindex"][g], g) for g in genes if g in ds["gindex"]]
    box_c = ["#e08214" if ds["sub_labels"][s] in INH else "#4a78b5" for s in range(n)]
    R, C = 4, 3
    with PdfPages(pdf_path) as pdf:
        for s0 in range(0, len(cols), per_page):
            page = cols[s0:s0 + per_page]
            fig, axes = plt.subplots(R, C, figsize=(16, 18)); axes = axes.ravel()
            for ax, (j, g) in zip(axes, page):
                col = ds["X"][:, j].astype(float)
                data = [col[ci] for ci in cby]
                if eff:
                    data = [pdsn.resample_counts(d.astype("int32"), eff, rng).astype(float) for d in data]
                bp = ax.boxplot(data, positions=np.arange(n), widths=.6, showfliers=False,
                                patch_artist=True, medianprops=dict(color="k", lw=.6))
                for pa, c in zip(bp["boxes"], box_c):
                    pa.set_facecolor(c); pa.set_edgecolor("0.3"); pa.set_linewidth(.4)
                ax.plot(np.arange(n), [d.mean() for d in data], "D", color="crimson", ms=3)
                ax.set_yscale("symlog", linthresh=1)
                ax.set_xticks(np.arange(n)); ax.set_xticklabels(labels, rotation=90, fontsize=5)
                ax.set_title(g, fontsize=9); ax.grid(axis="y", alpha=.25)
            for ax in axes[len(page):]:
                ax.axis("off")
            pdf.savefig(fig, dpi=110); plt.close(fig)


def confusion_grid(ds, genes, sizes=(100, 200, 300, 400), level="subclass", eff=0.1,
                   region=None, min_cells=15):
    """Confusion matrices at several panel sizes (optionally restricted to a region)."""
    from sklearn.metrics import confusion_matrix
    mask = np.ones(len(ds["true_subclass"]), bool) if region is None else (ds["region"] == region)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13)); axes = axes.ravel()
    labels = ds["sub_labels"] if level == "subclass" else ds["clu_labels"]
    keep = np.arange(len(labels))
    if level == "cluster":
        ids, cnt = np.unique(ds["true_cluster"][mask], return_counts=True)
        keep = ids[cnt >= min_cells]
    for ax, N in zip(axes, sizes):
        pred, true = classify(ds, genes[:N], eff, level)
        m = mask & np.isin(true, keep)
        cm = confusion_matrix(true[m], pred[m], labels=keep, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"top-{N} | acc={np.mean(pred[mask] == true[mask]):.3f}", fontsize=11)
        if level == "subclass":
            ax.set_xticks(range(len(keep))); ax.set_xticklabels(labels[keep], rotation=90, fontsize=6)
            ax.set_yticks(range(len(keep))); ax.set_yticklabels(labels[keep], fontsize=6)
        else:
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    return fig


def umap_grid(ds, panels, sizes=(100, 200, 300, 400), refs=None, eff=0.1, cells=None,
              title=""):
    """Grid of UMAPs: rows=panels (x sizes) + optional reference rows; coloured by subclass."""
    refs = refs or {}
    sub = ds["true_subclass"] if cells is None else ds["true_subclass"][cells]
    cmap = plt.get_cmap("tab20")
    pt = np.array([cmap(s % 20) for s in sub])
    nrow = len(panels) + len(refs)
    fig, axes = plt.subplots(nrow, len(sizes), figsize=(4 * len(sizes), 4 * nrow),
                             squeeze=False)
    for r, (pname, glist) in enumerate(panels.items()):
        for c, N in enumerate(sizes):
            um, _ = embed(ds, glist[:N], eff, cells)
            ax = axes[r, c]
            ax.scatter(um[:, 0], um[:, 1], c=pt, s=2, alpha=.6, linewidths=0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"top {N}", fontsize=12)
            if c == 0:
                ax.set_ylabel(pname, fontsize=12)
    for ri, (rname, rg) in enumerate(refs.items()):
        um, _ = embed(ds, rg, eff, cells)
        ax = axes[len(panels) + ri, 0]
        ax.scatter(um[:, 0], um[:, 1], c=pt, s=2, alpha=.6, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel(rname, fontsize=11)
        for c in range(1, len(sizes)):
            axes[len(panels) + ri, c].axis("off")
    present = np.unique(sub)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap(s % 20), label=ds["sub_labels"][s])
               for s in present]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.99, 0.98])
    return fig


def leiden_vs_labels(ds, genes, eff=0.1, cells=None):
    """UMAP coloured by subclass vs by Leiden + composition heatmap + ARI."""
    from sklearn.metrics import adjusted_rand_score as ARI
    sub = ds["true_subclass"] if cells is None else ds["true_subclass"][cells]
    um, leid = embed(ds, genes, eff, cells, with_leiden=True)
    cmap = plt.get_cmap("tab20")
    ari = ARI(sub, leid)
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.5))
    ax[0].scatter(um[:, 0], um[:, 1], c=[cmap(s % 20) for s in sub], s=2, alpha=.6, linewidths=0)
    ax[0].set_title("subclass"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].scatter(um[:, 0], um[:, 1], c=[cmap(l % 20) for l in leid], s=2, alpha=.6, linewidths=0)
    ax[1].set_title(f"Leiden ({len(np.unique(leid))}); ARI={ari:.2f}")
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ct = pd.crosstab(pd.Series(leid, name="leiden"),
                     pd.Series(ds["sub_labels"][sub], name="subclass"))
    ctn = ct.div(ct.sum(1), axis=0)
    im = ax[2].imshow(ctn.values, aspect="auto", cmap="viridis")
    ax[2].set_xticks(range(ct.shape[1])); ax[2].set_xticklabels(ct.columns, rotation=90, fontsize=7)
    ax[2].set_yticks(range(ct.shape[0])); ax[2].set_yticklabels(ct.index, fontsize=7)
    ax[2].set_title("Leiden composition"); plt.colorbar(im, ax=ax[2], fraction=.046)
    plt.tight_layout()
    return fig, ari


def panel_table(datasets, panels, sizes=(100, 200, 400), eff=0.1):
    """Accuracy table: every panel x dataset x size (subclass + cluster)."""
    rows = []
    for size in sizes:
        for pname, glist in panels.items():
            for dname, ds in datasets.items():
                sa, n = pe.evaluate_panel(ds, glist[:size], eff, "subclass")
                ca, _ = pe.evaluate_panel(ds, glist[:size], eff, "cluster")
                rows.append(dict(size=size, panel=pname, data=dname, n=n,
                                 subclass=sa, cluster=ca))
    return pd.DataFrame(rows)


def region_accuracy(ds, genes, regions, sizes=(50, 100, 200, 300, 400), eff=0.1):
    """Accuracy vs panel size for several regions (means from all cells)."""
    rows = []
    for N in sizes:
        ps, ts = classify(ds, genes[:N], eff, "subclass")
        pc, tc = classify(ds, genes[:N], eff, "cluster")
        for r, lab in regions.items():
            m = ds["region"] == lab
            rows.append(dict(size=N, region=r,
                             subclass=float(np.mean(ps[m] == ts[m])),
                             cluster=float(np.mean(pc[m] == tc[m]))))
    return pd.DataFrame(rows)
