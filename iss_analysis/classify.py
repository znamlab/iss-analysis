import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import pciSeq as pci
from .io import load_data_tasic_2018
from itertools import cycle


def classify_cells(
    masks_file,
    spots_data,
    ref_data,
    opts=None,
    filter_neurons=True,
    classify_by="cluster",
):
    if opts is None:
        opts = {}
        opts["Inefficiency"] = 0.001
        opts["MisreadDensity"] = 0.0000001
        opts["SpotReg"] = 0.01

    masks = np.load(masks_file, allow_pickle=True)
    spots = pd.read_pickle(spots_data)
    spots.rename(columns={"gene": "Gene"}, inplace=True)
    gene_names = spots["Gene"].unique()
    exons_df, genes = load_data_tasic_2018(ref_data, filter_neurons=filter_neurons)
    sc_data = (
        exons_df.set_index(classify_by)
        .filter(regex=r"\d")
        .set_axis(genes, axis=1, inplace=False)
    )

    cell_data, gene_data = pci.fit(
        spots, coo_matrix(masks), sc_data[gene_names].T, opts=opts
    )
    cell_data["BestClass"] = cell_data.apply(
        lambda r: r["ClassName"][np.argmax(r["Prob"])], axis=1
    )
    cell_data["BestProb"] = cell_data.apply(lambda r: np.max(r["Prob"]), axis=1)
    return cell_data, gene_data


def plot_cell_types(cell_data):
    colors = {
        "L2/3": "deepskyblue",
        "L4": "dodgerblue",
        "L5 IT": "blue",
        "L5 NP": "magenta",
        "L5 PT": "blueviolet",
        "L6 CT": "forestgreen",
        "L6 IT": "violet",
        "L6b": "black",
        "Pvalb": "darkorange",
        "Sst": "orangered",
        "Sncg": "deeppink",
        "Serpinf1": "limegreen",
        "Lamp5": "tomato",
        "Vip": "crimson",
    }
    markers = cycle("ov^<>spPXD*")
    plt.figure(figsize=(15, 15))
    zero_class = cell_data[cell_data["BestClass"] == "Zero"]
    cell_data = cell_data[cell_data["BestClass"] != "Zero"]
    ax = plt.subplot(1, 1, 1)
    clusters = np.sort(cell_data["BestClass"].unique())
    for i, cluster in enumerate(clusters):
        color = "gray"
        for type in colors:
            if cluster.find(type) != -1:
                color = colors[type]
        plt.plot(
            cell_data[cell_data["BestClass"] == cluster]["X"],
            cell_data[cell_data["BestClass"] == cluster]["Y"],
            next(markers),
            c=color,
            markersize=10,
        )

    plt.legend(clusters, loc="right", ncol=2, bbox_to_anchor=(0.0, 0.0, 1.5, 1.0))
    plt.plot(zero_class["X"], zero_class["Y"], "o", markersize=5, c="gray")
    ax.set_aspect("equal", "box")
    ax.invert_yaxis()
