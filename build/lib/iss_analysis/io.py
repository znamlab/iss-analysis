import numpy as np
import pandas as pd
import re
import scipy.sparse as ss
import h5py
from .pick_genes import compute_means


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
    fname_metadata = f'{datapath}mouse_VISp_2018-06-14_samples-columns.csv'
    metadata = pd.read_csv(fname_metadata, low_memory=False)
    fname = f'{datapath}mouse_VISp_2018-06-14_exon-matrix.csv'
    exons = pd.read_csv(fname, low_memory=False)
    fname_genes = f'{datapath}mouse_VISp_2018-06-14_genes-rows.csv'
    genes = pd.read_csv(fname_genes, low_memory=False)
    gene_names = genes['gene_symbol']
    metadata.set_index('sample_name', inplace=True)
    keep_genes = filter_genes(gene_names)
    exons = exons.iloc[keep_genes]
    gene_names = gene_names.iloc[keep_genes]
    exons_df = metadata.join(exons.T, on='sample_name')
    # only include neurons
    include_classes = [
        'GABAergic',
        'Glutamatergic'
    ]
    # get rid of low quality cells etc
    if filter_neurons:
        exons_subset = exons_df[exons_df['class'].isin(include_classes)]
    else:
        exons_subset = exons_df
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('ALM')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Doublet')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Batch')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Low Quality')]
    exons_subset = exons_subset[~exons_subset['subclass'].str.contains('High Intronic')]

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
        """ Load HDF5 data as a sparse matrix """
        data = h5f[data_path]
        x = data['x']
        i = data['i']
        p = data['p']
        dims = data['dims']

        sparse_matrix = ss.csc_matrix((x[0:x.len()],
                                       i[0:i.len()],
                                       p[0:p.len()]),
                                      shape=(dims[0], dims[1]),
                                      dtype=np.int32)
        return sparse_matrix

    fname = f'{datapath}expression_matrix.hdf5'

    h5f = h5py.File(fname, 'r')

    exons = extract_sparse_matrix(h5f, '/data/exon')
    samples = [ sample.decode('utf-8') for sample in h5f['sample_names'] ]
    gene_names = [ gene.decode('utf-8') for gene in h5f['gene_names'] ]
    keep_genes = filter_genes(gene_names)
    gene_names = np.array(gene_names)[keep_genes]
    exons = exons[:, keep_genes]

    fname_metadata = f'{datapath}metadata.csv'

    metadata = pd.read_csv(fname_metadata, low_memory=False)
    include_classes = [
        'L5 PT CTX', 'L5 IT CTX', 'L4/5 IT CTX', 'L6 IT CTX', 'L6 CT CTX',
        'L5/6 NP CTX', 'Pvalb', 'Vip', 'L2/3 IT CTX', 'Lamp5', 'Sst',
        'Sst Chodl', 'Sncg', 'Car3', 'L6b CTX', 'CR', 'Meis2'
    ]
    metadata = metadata[
        (metadata['region_label'] == 'VISp') &
        (metadata['subclass_label'].isin(include_classes)) &
        (metadata['sample_name'].isin(samples))
        ]
    keep_cells = np.array([ sample in metadata['sample_name'].unique() for sample in samples ])
    samples = np.array(samples)[keep_cells]
    exons = exons[keep_cells, :]
    exons_df = pd.DataFrame(
        exons.todense(),
        columns=['gene_' + gene for gene in gene_names]
    )

    exons_df['sample_name'] = samples
    exons_df = metadata.join(exons_df.set_index('sample_name'), on='sample_name')

    return exons_df, pd.Series(gene_names)


def filter_genes(gene_names):
    # get rid of gene models etc
    genes_Rik = np.array([ re.search('Rik$', s) is not None for s in gene_names ])
    genes_Gm = np.array([ re.search('Gm\d', s) is not None for s in gene_names ])
    genes_LOC = np.array([ re.search('LOC\d', s) is not None for s in gene_names ])
    genes_AA = np.array([ re.search('^[A-Z]{2}\d*$', s) is not None for s in gene_names ])
    keep_genes = np.logical_not(genes_Rik + genes_Gm + genes_LOC + genes_AA)
    
    #Pseudocode for filtering out genes that are below a mean read threshold across all clusters
    genes_lowcount = np.array()
    _, _, cluster_means, _ = compute_means(load_data_tasic_2018('/camp/lab/znamenskiyp/home/shared/resources/allen2018/'), 'cluster') # n clusters x n genes matrix
    for gene in gene_names:
        if np.max(cluster_means[gene] < gene_threshold:
            genes_lowcount.append(gene)
    keep_genes = np.logical_not(genes_Rik + genes_Gm + genes_LOC + genes_AA + genes_lowcount)
    return keep_genes