import seaborn as sns


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
