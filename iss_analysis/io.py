import numpy as np
import pandas as pd
import re
import scipy.sparse as ss
import h5py
import iss_preprocess as issp
from iss_preprocess.pipeline.ara_registration import spots_ara_infos


def get_chamber_datapath(acquisition_folder, chamber_list=None, verbose=False):
    """Get the chamber folders from the acquisition folder.

    Simple utility function to get the chamber folders from the acquisition folder or
    the chamber folder itself.

    Args:
        acquisition_folder (str): The path to the acquisition folder.
        chamber_list (list, optional): The list of chambers to include. Default is None.
        verbose (bool, optional): Print verbose output. Default is False.

    Returns:
        list: A list of chamber folders.
    """
    main_folder = issp.io.get_processed_path(acquisition_folder)
    if "chamber" in main_folder.name:  # single chamber
        assert chamber_list is None, "chamber_list should be None for single chamber"
        chambers = [acquisition_folder]
    else:  # mouse folder
        chambers = list(main_folder.glob("chamber_*"))
        chambers = [chamber for chamber in chambers if chamber.is_dir()]
        if chamber_list is not None:
            chambers = [chamber for chamber in chambers if chamber.name in chamber_list]
        # keep only if there are spots in the folder
        all_chambers = tuple(chambers)
        for chamber in all_chambers:
            if not len(list(chamber.glob("*spots*.pkl"))):
                if verbose:
                    print(f"No spots in {chamber}, removing from list")
                chambers.remove(chamber)
        # make the path relative to project, like acquisition_folder
        root = str(main_folder)[: -len(acquisition_folder)]
        chambers = [str(chamber.relative_to(root)) for chamber in chambers]
    return chambers


def get_sections_info(project, mouse, chamber=None):
    """Load section positions and sort them by absolute section

    Args:
        project (str): project name
        mouse (str): mouse name
        chamber (str, optional): chamber name, default None

    Returns:
        sections_info (pd.DataFrame): DataFrame with section positions
    """
    assert isinstance(project, str), "project should be a string"
    assert isinstance(mouse, str), "mouse should be a string"

    if chamber is None:
        # they all contain the same information, so use whichever
        data_path = get_chamber_datapath(f"{project}/{mouse}")[0]
    else:
        data_path = f"{project}/{mouse}/{chamber}"
    sections_info = issp.io.load_section_position(data_path)
    sections_info.rename(columns={"chamber_position": "roi"}, inplace=True)
    sections_info["chamber"] = sections_info["chamber"].map(
        lambda x: f"chamber_{x:02d}"
    )
    sections_info.sort_values("absolute_section", inplace=True)
    sections_info.reset_index(drop=True, inplace=True)
    sections_info.head()
    return sections_info


def get_mcherry_cells(
    project, mouse, verbose=True, which="curated", prefix="mCherry_1"
):
    """Get the mCherry cells from the manual click.

    Args:
        project (str): The project name.
        mouse (str): The mouse name.
        verbose (bool, optional): Print verbose output. Default is True.
        which (str, optional): Which starter cells to load. Use 'manual' to load the
            initial manual click in `analysis/starter_cells`, 'curated' for the more
            exhaustive curated masks in `chamber_XX/cells`. Default is 'curated'.
        prefix (str, optional): The prefix of the mCherry cells files. Default is
            'mCherry_1'.

    Returns:
        pd.DataFrame: The mCherry cells.
    """
    mouse_path = issp.io.get_processed_path(f"{project}/{mouse}")
    mcherry = []
    if which == "manual":
        manual_click = mouse_path / "analysis" / "mcherry_cells"
        assert manual_click.exists()
        for fname in manual_click.glob("mcherry_cells*.csv"):
            # names are like mcherry_cells_`mouse`_`chamber`_roi_`roinum`.csv
            # so we can get the chamber and roi from them
            match = re.match(rf"mcherry_cells_{mouse}_(.+)_roi_(\d+).csv", fname.name)
            if match is None:
                raise ValueError(f"Invalid filename {fname.name}")
            chamber, roi = match.groups()
            clicked = pd.read_csv(fname)
            if not len(clicked):
                if verbose:
                    print(f"No mCherry cells for {fname.stem} (csv is empty)")
                continue

            mch = pd.DataFrame(
                columns=["x", "y"],
                index=np.arange(len(clicked)),
                data=clicked[["axis-1", "axis-0"]].values,
            )
            mch["chamber"] = chamber
            mch["roi"] = int(roi)
            mch["original_index"] = clicked["index"].astype(int)
            mcherry.append(mch)
    elif which == "curated":
        chambers = get_chamber_datapath(f"{project}/{mouse}")
        for chamber_path in chambers:
            chamber_path = issp.io.get_processed_path(chamber_path)
            chamber = chamber_path.stem
            mch_folder = chamber_path / "cells" / f"{prefix}_cells"
            df = mch_folder / f"{prefix}_df_corrected_curated.pkl"
            assert df.exists(), f"No mCherry cells found in {df}"
            mch = pd.read_pickle(df)
            mch["chamber"] = chamber
            mch["original_index"] = mch.index
            mch["x"] = mch["centroid-1"]
            mch["y"] = mch["centroid-0"]
            mcherry.append(mch)
    else:
        raise ValueError(f"Invalid which parameter {which}")
    mcherry = pd.concat(mcherry, ignore_index=True)

    if which == "manual":
        mcherry["mcherry_uid"] = (
            mcherry["chamber"]
            + "_"
            + mcherry["roi"].astype(str)
            + "_"
            + mcherry["original_index"].astype(str)
        )
    else:
        # original index includes the roi number
        mcherry["mcherry_uid"] = (
            mcherry["chamber"] + "_" + mcherry["original_index"].astype(str)
        )
    if verbose:
        print(f"Loaded {len(mcherry)} mCherry cells position")
    return mcherry


def get_starter_cells(project, mouse, verbose=True):
    """Get the starter cells from the manual click.

    Args:
        project (str): The project name.
        mouse (str): The mouse name.
        verbose (bool, optional): Print verbose output. Default is True.

    Returns:
        pd.DataFrame: The starter cells.
    """

    manual_click = (
        issp.io.get_processed_path(f"{project}/{mouse}") / "analysis" / "starter_cells"
    )
    assert manual_click.exists()
    starters = []
    for data_path in get_chamber_datapath(f"{project}/{mouse}"):
        chamber = issp.io.get_processed_path(data_path).stem
        for roi in range(1, 11):
            fname = manual_click / f"starter_cells_{mouse}_{chamber}_roi_{roi}.csv"
            if not fname.exists():
                if verbose:
                    print(f"No manual click for {chamber} roi {roi}")
                continue
            clicked = pd.read_csv(fname)
            if not len(clicked):
                if verbose:
                    print(f"No starter cells for {chamber} roi {roi} (csv is empty)")
                continue
            st = pd.DataFrame(
                columns=["chamber", "roi", "x", "y"], index=np.arange(len(clicked))
            )
            st["chamber"] = chamber
            st["roi"] = roi
            st["x"] = clicked["axis-1"].values
            st["y"] = clicked["axis-0"].values
            starters.append(st)
    starters = pd.concat(starters, ignore_index=True)
    if verbose:
        print(f"Loaded {len(starters)} starter cells position")
    return starters


def get_genes_spots(project, mouse, add_ara_info=True, verbose=False, reload=True):
    """Get the genes spots from the processed data with ARA information.

    Args:
        project (str): The project name.
        mouse (str): The mouse name.
        chamber (str): The chamber name.
        add_ara_info (bool, optional): Add ARA information. Default is True.
        verbose (bool, optional): Print more info. Default is False
        reload (bool, optional): Reload the ARA information. Default is True.

    Returns:
        pd.DataFrame: The genes spots with ARA information.
    """
    sec_inf = get_sections_info(project, mouse, chamber=None)
    all_spots = []
    for section, sec_df in sec_inf.iterrows():
        roi = sec_df["roi"]
        chamber = sec_df["chamber"]
        data_path = f"{project}/{mouse}/{chamber}"
        spot_file = (
            issp.io.get_processed_path(data_path) / f"genes_round_spots_{roi}.pkl"
        )
        assert spot_file.exists(), f"{spot_file} does not exist"
        spots = pd.read_pickle(spot_file)
        spots["slice"] = f"{chamber}_{roi:02d}"
        spots["roi"] = roi
        spots["chamber"] = chamber
        if not len(spots):
            raise ValueError(f"No spots for {spot_file}")
        if add_ara_info:
            try:
                spots = spots_ara_infos(
                    data_path,
                    spots,
                    roi,
                    atlas_size=10,
                    acronyms=True,
                    inplace=True,
                    full_scale_coordinates=False,
                    reload=reload,
                    verbose=verbose,
                )
            except IOError as e:
                print(f"Error for {chamber} roi {roi}: {e}")
        all_spots.append(spots)
    return pd.concat(all_spots, ignore_index=True)


def filter_genes(gene_names):
    # get rid of gene models etc
    genes_Rik = np.array([re.search("Rik$", s) is not None for s in gene_names])
    genes_Gm = np.array([re.search(r"Gm\d", s) is not None for s in gene_names])
    genes_LOC = np.array([re.search(r"LOC\d", s) is not None for s in gene_names])
    genes_AA = np.array(
        [re.search(r"^[A-Z]{2}\d*$", s) is not None for s in gene_names]
    )
    keep_genes = np.logical_not(genes_Rik + genes_Gm + genes_LOC + genes_AA)
    return keep_genes


def load_data_tasic_2018(datapath, filter_neurons=True):
    """
    Load the scRNAseq data from Tasic et al., "Shared and distinct
    transcriptomic cell types across neocortical areas", Nature, 2018.

    Args:
        datapath: path to the data

    Returns:
        exons_matrix: n cells x n genes matrix (numpy.ndarray) of read counts
        cluster_ids: numpy.array of cluster assignments from the cell metadata
        cluster_means: n clusters x n genes matrix (numpy.ndarray) of mean read
            counts for each cluster
        cluster_labels: list of cluster names
        gene_names: pandas.Series of gene names

    """
    fname_metadata = f"{datapath}mouse_VISp_2018-06-14_samples-columns.csv"
    metadata = pd.read_csv(fname_metadata, low_memory=False)
    fname = f"{datapath}mouse_VISp_2018-06-14_exon-matrix.csv"
    exons = pd.read_csv(fname, low_memory=False)
    fname_genes = f"{datapath}mouse_VISp_2018-06-14_genes-rows.csv"
    genes = pd.read_csv(fname_genes, low_memory=False)
    gene_names = genes["gene_symbol"]
    metadata.set_index("sample_name", inplace=True)
    keep_genes = filter_genes(gene_names)
    exons = exons.iloc[keep_genes]
    gene_names = gene_names.iloc[keep_genes]
    exons_df = metadata.join(exons.T, on="sample_name")
    # only include neurons
    include_classes = ["GABAergic", "Glutamatergic"]
    # get rid of low quality cells etc
    if filter_neurons:
        exons_subset = exons_df[exons_df["class"].isin(include_classes)]
    else:
        exons_subset = exons_df
    exons_subset = exons_subset[~exons_subset["cluster"].str.contains("ALM")]
    exons_subset = exons_subset[~exons_subset["cluster"].str.contains("Doublet")]
    exons_subset = exons_subset[~exons_subset["cluster"].str.contains("Batch")]
    exons_subset = exons_subset[~exons_subset["cluster"].str.contains("Low Quality")]
    exons_subset = exons_subset[~exons_subset["subclass"].str.contains("High Intronic")]

    return exons_subset, gene_names


def load_data_yao_2021(datapath):
    """
    Load the scRNAseq data from Yao et al., "A taxonomy of transcriptomic cell
    types across the isocortex and hippocampal formation", Cell, 2021.

    Args:
        datapath: path to the data

    Returns:
        exons_matrix: n cells x n genes matrix (numpy.ndarray) of read counts
        cluster_ids: numpy.array of cluster assignments from the cell metadata
        cluster_means: n clusters x n genes matrix (numpy.ndarray) of mean read
            counts for each cluster
        cluster_labels: list of cluster names
        gene_names: pandas.Series of gene names

    """

    def extract_sparse_matrix(h5f, data_path):
        """Load HDF5 data as a sparse matrix"""
        data = h5f[data_path]
        x = data["x"]
        i = data["i"]
        p = data["p"]
        dims = data["dims"]

        sparse_matrix = ss.csc_matrix(
            (x[0 : x.len()], i[0 : i.len()], p[0 : p.len()]),
            shape=(dims[0], dims[1]),
            dtype=np.int32,
        )
        return sparse_matrix

    fname = f"{datapath}expression_matrix.hdf5"

    h5f = h5py.File(fname, "r")

    exons = extract_sparse_matrix(h5f, "/data/exon")
    samples = [sample.decode("utf-8") for sample in h5f["sample_names"]]
    gene_names = [gene.decode("utf-8") for gene in h5f["gene_names"]]
    keep_genes = filter_genes(gene_names)
    gene_names = np.array(gene_names)[keep_genes]
    exons = exons[:, keep_genes]

    fname_metadata = f"{datapath}metadata.csv"

    metadata = pd.read_csv(fname_metadata, low_memory=False)
    include_classes = [
        "L5 PT CTX",
        "L5 IT CTX",
        "L4/5 IT CTX",
        "L6 IT CTX",
        "L6 CT CTX",
        "L5/6 NP CTX",
        "Pvalb",
        "Vip",
        "L2/3 IT CTX",
        "Lamp5",
        "Sst",
        "Sst Chodl",
        "Sncg",
        "Car3",
        "L6b CTX",
        "CR",
        "Meis2",
    ]
    metadata = metadata[
        (metadata["region_label"] == "VISp")
        & (metadata["subclass_label"].isin(include_classes))
        & (metadata["sample_name"].isin(samples))
    ]
    keep_cells = np.array(
        [sample in metadata["sample_name"].unique() for sample in samples]
    )
    samples = np.array(samples)[keep_cells]
    exons = exons[keep_cells, :]
    exons_df = pd.DataFrame(
        exons.todense(), columns=["gene_" + gene for gene in gene_names]
    )

    exons_df["sample_name"] = samples
    exons_df = metadata.join(exons_df.set_index("sample_name"), on="sample_name")

    return exons_df, pd.Series(gene_names)
