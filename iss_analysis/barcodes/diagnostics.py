import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import iss_preprocess as issp


def plot_gmm_clusters(all_barcode_spots, gmm, thresholds=None):
    """Plot the clusters of the GMM model on the 4 metrics.

    Args:
        all_barcode_spots (pd.DataFrame): DataFrame containing the barcode spots.
        gmm (GaussianMixture): The trained Gaussian Mixture Model.
        thresholds (dict): The thresholds for each metric. Default is None.

    Returns:
        pgrid (seaborn.axisgrid.PairGrid): The pairplot of the clusters.
    """
    if thresholds is None:
        thresholds = {}

    metrics = ["dot_product_score", "spot_score", "mean_intensity", "mean_score"]
    d = all_barcode_spots[metrics][:: len(all_barcode_spots) // 5000].copy()
    d["labels"] = gmm.predict(d.values)
    pgrid = sns.pairplot(d, hue="labels", plot_kws={"alpha": 0.1}, palette="tab10")
    pgrid.figure.suptitle("GMM clusters", y=1.02)
    for prop, th in thresholds.items():
        if prop in metrics:
            ind = metrics.index(prop)
            for iax, ax in enumerate(pgrid.axes[ind]):
                if iax == ind:
                    continue
                ax.axhline(th, color="red")
            for iax, ax in enumerate(pgrid.axes[:, ind]):
                if iax == ind:
                    continue
                ax.axvline(th, color="red")
    return pgrid


def plot_error_along_sequence(error_dict, nrows=2, plot_matrix=False, **kwargs):
    """Plot the error along the sequence of the barcode spots.

    Args:
        error_dict (dict): The dictionary containing the error along the sequence.
        nrows (int): The number of rows in the plot. Default is 2.
        plot_matrix (bool): Whether to plot the matrix or the mean. Default is False.
        **kwargs: Additional keyword arguments for the plot.

    Returns:
        fig_dict (dict): Dictionary containing the figures for each chamber.
    """
    fig_dict = {}
    for chamber, cdict in error_dict.items():
        fig = plt.figure(figsize=(10, 5))
        nrois = len(cdict)
        ncols = nrois // nrows
        for iroi, (roi, error_along_sequence) in enumerate(cdict.items()):
            if iroi:
                sharex = ax
                sharey = ax
            else:
                sharex = None
                sharey = None
            ax = fig.add_subplot(nrows, ncols, iroi + 1, sharex=sharex, sharey=sharey)
            if plot_matrix:
                ax.imshow(error_along_sequence, aspect="auto", **kwargs)
            else:
                ax.plot(np.mean(error_along_sequence, axis=0), **kwargs)
            ax.set_title(f"ROI {roi}")
        fig.suptitle(f"Chamber {chamber}")
        fig.tight_layout()
        fig_dict[chamber] = fig
    return fig_dict
