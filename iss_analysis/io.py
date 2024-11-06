import numpy as np
import pandas as pd
import re
import scipy.sparse as ss
import h5py
import iss_preprocess as issp
import anndata
from pathlib import Path
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


def get_mcherry_cells(project, mouse, verbose=True):
    """Get the mCherry cells from the manual click.

    Args:
        project (str): The project name.
        mouse (str): The mouse name.
        verbose (bool, optional): Print verbose output. Default is True.

    Returns:
        pd.DataFrame: The mCherry cells.
    """
    manual_click = (
        issp.io.get_processed_path(f"{project}/{mouse}") / "analysis" / "mcherry_cells"
    )
    assert manual_click.exists()
    mcherry = []
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
    mcherry = pd.concat(mcherry, ignore_index=True)
    mcherry["mcherry_uid"] = (
        mcherry["chamber"]
        + "_"
        + mcherry["roi"].astype(str)
        + "_"
        + mcherry["original_index"].astype(str)
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


def expression_matrix_filter(expression_matrix, 
                             feature_matrix_label, 
                             cell_extended,  
                             region_of_interest = 'MO-FRP', 
                             neurotransmitters = ['GABA', 'Glut']):
    '''
    Generates an anndata object that has all of the metadata information for cluster and ROI. Because
    of how the anndata files are saved, this corresponds just to a feature_matrix. 
    I don't know why the number of entries for a feature matrix in the metadata
    is smaller than in the data by about 1K, but I've decided to not really care for
    now.  I've made a note and I'll probably raise an issue on their repo (scary)

    Args: 
        expression_matrix(anndata): a feature_matrix as read by abc_cache
        feature_matrix_label(str): a string indicating the name of the feature matrix on the server. 
        cell_extended(df): a pandas df generated by build_extended_metadata
        other params: filters for the kinds of cell you want to keep. 

    Out: 
        filtered_expression_matrix(anndata): an anndata object with massive metadata. 
    '''
    #Build metadata
    pred = (cell_extended['feature_matrix_label'] == feature_matrix_label)
    cell_filtered = cell_extended[pred]
    print('Cell labels in metadata:')
    print(len(cell_filtered.index))
    print('Cell labels in cells:')
    print(len(expression_matrix.obs.index))

    #Asserting labels are unique
    assert expression_matrix.obs.index.is_unique, "expression_matrix.obs has duplicate cell_label indices."
    assert cell_filtered.index.is_unique, "Metadata df has duplicate cell_label indices."

    #Dropping columns that are in both expression_matrix.obs and metadata
    #to allow joining of the two by cell label. 
    cell_filtered = cell_filtered.drop(columns = 'library_label')
    cell_filtered = cell_filtered.drop(columns = 'cell_barcode')

    #joining
    expression_matrix.obs = expression_matrix.obs.join(cell_filtered, how='left')

    #filtering for the right ROI and neurotransmitter
    filtered_expression_matrix = expression_matrix[(expression_matrix.obs['region_of_interest_acronym']==region_of_interest) & (expression_matrix.obs['neurotransmitter'].isin(neurotransmitters))]

    return filtered_expression_matrix

def build_extended_metadata(abc_cache, directory = 'WMB-10X'):
    '''
    Gets the right kind of metadata from the cache as a dataframe. 
    Adds the ROI information and cluster information to the df. They're
    originally stored as different files, but we want to filter by
    all of these and include subclass information. 

        Out: cell_extended (df): metadata for every cell in the directory
    '''
    #get basic metadata
    cell = abc_cache.get_metadata_dataframe(
    directory=directory,
    file_name='cell_metadata',
    dtype={'cell_label': str}
    )  
    cell.set_index('cell_label', inplace=True)

    #add region of interest
    roi = abc_cache.get_metadata_dataframe(directory=directory, file_name='region_of_interest_metadata')
    roi.set_index('acronym', inplace=True)
    roi.rename(columns={'order': 'region_of_interest_order',
                        'color_hex_triplet': 'region_of_interest_color'},
            inplace=True)
    
    #add clustering
    cluster_details = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_pivoted',
    keep_default_na=False
    )
    cluster_details.set_index('cluster_alias', inplace=True)

    #merge
    cell_extended = cell.join(cluster_details, on='cluster_alias')
    cell_extended = cell_extended.join(roi[['region_of_interest_order', 'region_of_interest_color']], on='region_of_interest_acronym')

    return cell_extended


def generate_csv(expression_matrix, download_base, region_of_interest, neurotransmitters):
    """
    Generates CSV files for expression matrix and gene mapping data.
    
    Parameters:
        expression_matrix (AnnData): The gene expression data.
        download_base (Path): The base directory to save the CSV files.
        region_of_interest (str): The region of interest.
        neurotransmitters (list or str): List or string of neurotransmitter(s).
    
    Returns:
        tuple: DataFrames for expression matrix and gene names.
    """
    # Convert neurotransmitters list to a string if it's a list
    if isinstance(neurotransmitters, list):
        neurotransmitters_str = "_".join(neurotransmitters)
    else:
        neurotransmitters_str = neurotransmitters

    # Extract the gene expression data (X matrix) and convert it to a DataFrame
    df = pd.DataFrame(expression_matrix.X.toarray(), columns=expression_matrix.var_names)

    # Add the 'subclass' and 'cluster' columns from the `obs` DataFrame to `df`
    df['subclass'] = expression_matrix.obs['subclass']
    df['cluster'] = expression_matrix.obs['cluster']

    # Create gene mapping DataFrame
    gene_mapping = pd.DataFrame({
        'gene_index': range(len(expression_matrix.var)),
        'gene_identifier': expression_matrix.var_names,
        'gene_symbol': expression_matrix.var.gene_symbol.values
    })

    # Define the path to save the DataFrame as a CSV
    df_path = download_base / f'WMB_processed_{region_of_interest}_{neurotransmitters_str}.csv'
    df.to_csv(df_path, index=False)

    # Define the path for gene mapping CSV
    gene_path = download_base / f'gene_mapping_{region_of_interest}_{neurotransmitters_str}.csv'
    gene_mapping.to_csv(gene_path, index=False)

    # Return the generated DataFrames
    gene_names = expression_matrix.var.gene_symbol
    return df, gene_names



def read_yao_2023(datapath):
    '''
    Reads the expression matrix and gene mapping files in the specified directory.
    The function will automatically detect files that start with "WMB_processed" or "gene_mapping".
    
    Raises an error if multiple files match either pattern.

    Parameters:
        datapath (str or Path): Path to the directory containing the files.

    Returns:
        tuple: DataFrames for expression matrix and gene names.
    '''
    datapath = Path(datapath)

    # Find files that start with WMB_processed or gene_mapping
    csv_files = list(datapath.glob('WMB_processed*.csv'))
    gene_files = list(datapath.glob('gene_mapping*.csv'))

    # Check if there are multiple files for either pattern
    if len(csv_files) > 1:
        raise FileExistsError("Multiple files starting with 'WMB_processed' found. Please ensure there is only one matching file.")
    if len(gene_files) > 1:
        raise FileExistsError("Multiple files starting with 'gene_mapping' found. Please ensure there is only one matching file.")

    # Check if files are found
    if not csv_files or not gene_files:
        raise FileNotFoundError("Could not find required files starting with 'WMB_processed' or 'gene_mapping'.")

    # Read the single files found
    df = pd.read_csv(csv_files[0])
    gene_names = pd.read_csv(gene_files[0])

    return df, gene_names


def main_yao_2023(abc_cache, 
                  directory, 
                  filename_list, 
                  download_base,
                  region_of_interest = 'MO-FRP', 
                  neurotransmitters = ['GABA', 'Glut']):
    
    '''
    Accesses the Allen dataset on Yao et. al 2023, "A high-resolution 
    transcriptomic and spatial atlas of cell types in the 
    whole mouse brain". Downloads relevant data in their format
    according to abc_atlas_access, which is their package to interact with
    the dataset and generates a csv with subclass and cluster layer. 
    The taxonomic denominations are according to the paper. 

    Args: 
        abc_cache(class): generated by the Allen abc_cache package

        directory(str): the directory in the Allen server, generally WMB-10Xv3
        for 10X scRNAseq. 

        filename_list(list of str): the list of expression matrix that we want

        download_base(Path): the location of the abc_cache. This is also where
        the outputs will be saved. 

        region_of_interest(str): one of the dissected regions of the
        atlas. The pipeline will only keep data from there. 
        
        neurotransmitters(list of str): neurotransmitters to filter the dataset by. 

    Out: 
        df: cells x genes dataframe. Indexed by cell_label and Ensembl
        gene ID, like the Allen data. Contains two additional columns that
        indicate the subclass and cluster identity of each cell, for the pick_genes 
        pipeline. 

        gene_names: indexed by Ensembl ID, the names of the genes. 
    '''

    filelist = []

    for filename in filename_list:
        
        print('Downloading or accessing data. If not already in the cache, could be long')
        file = abc_cache.get_data_path(directory = directory, 
                                        file_name = f'{filename}/raw')
        
        expression_matrix = anndata.read_h5ad(file, backed='r')

        print('Building metadata')
        cell_extended = build_extended_metadata(abc_cache)

        print('Filtering data with roi/neurotransmitter')
        filtered_expression_matrix = expression_matrix_filter(expression_matrix, 
                                    feature_matrix_label = filename, 
                                    cell_extended = cell_extended,  
                                    region_of_interest = region_of_interest, 
                                    neurotransmitters = neurotransmitters)
        filelist.append(filtered_expression_matrix)

    filelist = tuple(filelist)
    assert filelist[0].var.equals(filelist[1].var), "The .var DataFrames are not identical."

    #merging the aadata with 'unique' strategy preserves metadata in obs and var because there
    #is only one possible value for it (a counterexample is the sum of all values for)
    #a variable, or its mean, which change depending on the subset). In our case, 
    #it's the gene names, so 'unique' for each variable. 
    
    concat_expression_matrix = anndata.concat([filelist[0],filelist[1]], merge = 'unique')

    print('Extracting and generating csv')
    df, gene_names = generate_csv(concat_expression_matrix, 
                                  download_base = download_base, 
                                  region_of_interest=region_of_interest, 
                                  neurotransmitters=neurotransmitters)

    return df, gene_names
    
