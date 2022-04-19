import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import h5py
import scipy.sparse as ss


def filter_genes(gene_names):
    # get rid of gene models etc
    genes_Rik = np.array([ re.search('Rik$', s) is not None for s in gene_names ])
    genes_Gm = np.array([ re.search('Gm\d', s) is not None for s in gene_names ])
    genes_LOC = np.array([ re.search('LOC\d', s) is not None for s in gene_names ])
    genes_AA = np.array([ re.search('^[A-Z]{2}\d*$', s) is not None for s in gene_names ])
    keep_genes = np.logical_not(genes_Rik + genes_Gm + genes_LOC + genes_AA)
    return keep_genes


def load_data_tasic_2018(datapath, classify_by='cluster'):
    """
    Load the scRNAseq data from Tasic et al., "Shared and distinct
    transcriptomic cell types across neocortical areas", Nature, 2018.

    Args:
        datapath: path to the data
        classify_by: field to use for classification (Default: 'cluster')

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
    exons_subset = exons_df[exons_df['class'].isin(include_classes)]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('ALM')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Doublet')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Batch')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Low Quality')]

    exons_matrix = exons_subset.filter(regex='\d').to_numpy() # n cells x n genes matrix
    # names of columns containing expression data are integer numbers
    expression_by_cluster = exons_subset.groupby([classify_by]).mean().filter(regex='\d')
    cluster_means = expression_by_cluster.to_numpy()   # n clusters x n genes matrix

    cluster_ids = np.empty(len(exons_subset[classify_by]), dtype=int)
    for i, cluster in enumerate(exons_subset[classify_by]):
        cluster_ids[i] = np.nonzero(expression_by_cluster.index == cluster)[0]
    cluster_labels = expression_by_cluster.index

    return exons_matrix, cluster_ids, cluster_means, cluster_labels, gene_names


def load_data_yao_2021(datapath, classify_by='cluster_label'):
    """
    Load the scRNAseq data from Yao et al., "A taxonomy of transcriptomic cell
    types across the isocortex and hippocampal formation", Cell, 2021.

    Args:
        datapath: path to the data
        classify_by: field to use for classification (Default: 'cluster_label')

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
    exons_matrix = exons_df.filter(regex='gene_').to_numpy()

    expression_by_cluster = exons_df.groupby([classify_by]).mean().filter(regex='gene_')
    cluster_means = expression_by_cluster.to_numpy()   # n clusters x n genes matrix
    cluster_ids = np.empty(len(exons_df[classify_by]), dtype=int)
    for i, cluster in enumerate(exons_df[classify_by]):
        cluster_ids[i] = np.nonzero(expression_by_cluster.index == cluster)[0]
    cluster_labels = expression_by_cluster.index
    return exons_matrix, cluster_ids, cluster_means, cluster_labels, pd.Series(gene_names)


def resample_counts(exons_matrix, cluster_means, efficiency=0.01):
    """
    Resample read counts to simulate lower efficiency of in situ sequencing.

    New read counts are sampled for each gene and cell from a binomial distribution
    with n = original read count and p = efficiency.

    Args:
        exons_matrix: n cells x n genes matrix (numpy.ndarray) of read counts
        cluster_means: n clusters x n genes matrix (numpy.ndarray) of mean read
            counts for each cluster
        efficiency: simulated efficiency of ISS

    Returns:
        exons_matrix, cluster_means

    """
    assert 0 < efficiency <= 1
    cluster_means = cluster_means * efficiency
    exons_matrix = np.random.binomial(n=exons_matrix, p=efficiency)
    return exons_matrix, cluster_means


def lognbinom(k, mu):
    """
    Log negative binomial PDF with r = 2.

    Fixing r to 2 is very computationally convenient because we don't need to
    compute any factorials.

    Args:
        k: counts
        mu: mean parameter

    Returns:
        Log negative binomial probability.

    """
    return np.log(k + 1) + np.log(mu / (mu + 2)) * k + np.log(2/(mu + 2)) * 2


def next_best_gene(include_genes, cluster_probs, cluster_ids):
    """
    Select the gene that maximizes classification accuracy when added to the
    gene set.

    Args:
        include_genes: boolean numpy.array of currently included genes.
        cluster_probs: cells x genes x clusters matrix (numpy.ndarray) of log-negative
            binomial probabilities generated from exons_matrix of observing
            a given number of reads in a given cluster
        cluster_ids: annotated cluster labels for cells

    Returns:
        Index of the gene that gives the highest accuracy when added to the
            current gene set.
        Resulting accuracy.

    """
    ngenes = len(include_genes)
    accuracy = np.zeros(ngenes)
    # first sum log-NB probabilities for already included genes for each cell
    cell_probs = cluster_probs[:, include_genes, :].sum(axis=1)
    for igene in tqdm(range(ngenes), leave=False):
        if not include_genes[igene]:
                # add the new gene and pick the cluster with the highest sum log-NB probability
                cluster_assignments = (cell_probs + cluster_probs[:, igene, :]).argmax(axis=1)
                accuracy[igene] = np.mean(cluster_ids == cluster_assignments)

    return np.argmax(accuracy), np.max(accuracy)


def remove_bad_gene(include_genes, cluster_probs, cluster_ids):
    """
    Check if removing any of the genes in the current set improves accuracy.

    Args:
        include_genes: boolean numpy.array of currently included genes.
        cluster_probs: cells x genes x clusters matrix (numpy.ndarray) of log-negative
            binomial probabilities generated from exons_matrix of observing
            a given number of reads in a given cluster
        cluster_ids: annotated cluster labels for cells

    Returns:
        If removing any of the genes improves accuracy, returns index of that
            gene. Otherwise returns None.
        Resulting accuracy.

    """
    cell_probs = cluster_probs[:, include_genes, :].sum(axis=1)
    starting_accuracy = np.mean(cluster_ids == cell_probs.argmax(axis=1))
    accuracy = np.zeros(len(include_genes))
    for igene in np.nonzero(include_genes)[0]:
        cluster_assignments = (cell_probs - cluster_probs[:, igene, :]).argmax(axis=1)
        accuracy[igene] = np.mean(cluster_ids == cluster_assignments)
    if np.max(accuracy) > starting_accuracy:
        return np.argmax(accuracy), np.max(accuracy)
    else:
        return None, starting_accuracy


def compute_cluster_probabilities(exons_matrix, cluster_means, nu=0.001):
    """
    Precompute log-negative binomial probabilities of observing the read counts
    for each cell assuming it comes from each cluster. Doing this in advance
    means we don't need to recompute it for each round of gene selection.

    Args:
        exons_matrix: n cells x n genes matrix (numpy.ndarray) of read counts
        cluster_means: n clusters x n genes matrix (numpy.ndarray) of mean read
            counts for each cluster
        nu: optional parameter added to the means of each cluster to "regularize"
            them. Otherwise any clusters with mean of 0 will have 0 probability
            of observing >0 reads.

    Returns:
        cells x genes x clusters matrix (numpy.ndarray)

    """
    cluster_probs = np.empty((exons_matrix.shape[0], exons_matrix.shape[1], cluster_means.shape[0]))
    for i, cluster in enumerate(cluster_means):
        cluster_probs[:,:,i] = lognbinom(exons_matrix, cluster + nu)
    return cluster_probs


def optimize_gene_set(cluster_probs, cluster_ids, gene_names, gene_set=(), niter=100):
    """
    Iteratively optimize the gene set to maximize classification accuracy.

    Uses a greedy algorithm, where at each step we add a gene that provides the
    largest increase in classification accuracy. We then check if we can improve
    accuracy by removing any of the already added genes.

    Args:
        cluster_probs: cells x genes x clusters matrix (numpy.ndarray) of log-negative
            binomial probabilities generated from exons_matrix of observing
            a given number of reads in a given cluster
        cluster_ids: numpy.array of annotated cluster labels
        gene_names: list of gene names corresponding to columns of cluster_probs
        gene_set: list of genes to include at the start of optimization. Optional.
            Default: empty list.
        niter: number of iterations. Optional, default: 100.

    Returns:
        Boolean numpy.array of inclded genes at the end of optimization.
        List of boolean arrays for every step of optimizations
        List of accuracies at every step of optimization

    """
    gene_set_history = []
    accuracy_history = []
    include_genes = np.isin(np.array(gene_names), gene_set)
    for i in range(niter):
        b, accuracy = next_best_gene(include_genes, cluster_probs, cluster_ids)
        include_genes[b] = True
        print(f'added {gene_names.iloc[b]}, accuracy = {accuracy}')
        if i > 0:
            r, accuracy = remove_bad_gene(include_genes, cluster_probs, cluster_ids)
            if r is not None:
                include_genes[r] = False
                print(f'removed {gene_names.iloc[r]}, accuracy = {accuracy}')
        gene_set_history.append(include_genes)
        accuracy_history.append(accuracy)
    return include_genes, gene_set_history, accuracy_history


def classify_cells(exons_matrix, cluster_means, gene_set, gene_names, nu=0.001):
    """
    Classify cells using a provided gene set.

    Args:
        exons_matrix: n cells x n genes matrix (numpy.ndarray) of read counts
        cluster_means: n clusters x n genes matrix (numpy.ndarray) of mean read
            counts for each cluster
        gene_set: list of genes to use for classification
        gene_names: list of genes corresponding to columns of exons_matrix
        nu: optional parameter added to the means of each cluster to "regularize"
            them. Otherwise any clusters with mean of 0 will have 0 probability
            of observing >0 reads.

    Returns:
        numpy.array of cluster assignments for each cell

    """
    include_genes = np.isin(np.array(gene_names), gene_set)
    cell_probs = np.empty((exons_matrix.shape[0], cluster_means.shape[0]))
    for i, cluster in enumerate(cluster_means):
        cell_probs[:,i] = lognbinom(exons_matrix[:, include_genes], cluster[include_genes] + nu).sum(axis=1)
    cluster_assignments = cell_probs.argmax(axis=1)
    return cluster_assignments


def evaluate_gene_set(exons_matrix, cluster_means, gene_set, gene_names, cluster_ids):
    """ Plot classification accuracy while incrementally growing the gene set """
    ngenes = len(gene_set)
    accuracy = np.empty(ngenes)
    for i in range(ngenes):
        cluster_assignments = classify_cells(
            exons_matrix, cluster_means, gene_set[:i], gene_names)
        accuracy[i] = np.mean(cluster_assignments == cluster_ids)
    plt.plot(accuracy)
    plt.xlabel('# genes')
    plt.ylabel('accuracy')
    plt.show()


def plot_confusion_matrix(cluster_ids, cluster_assignments, cluster_labels,
                          display_counts=True):
    """
    Plot a confusion matrix for the provided cluster assignments.

    Args:
        cluster_ids: numpy.array of "true" cluster ids
        cluster_assignments: numpy.array of cluster assignments
        cluster_labels: list of cluster names
        display_counts: (Default: True) whether to show counts or normalized
            proportions

    Returns:
        Confusion matrix

    """
    c = confusion_matrix(
        cluster_ids,
        cluster_assignments,
    )

    if display_counts:
        normalize = 'true'
        include_values = True
    else:
        normalize = None
        include_values = False
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    ConfusionMatrixDisplay.from_predictions(
        cluster_ids,
        cluster_assignments,
        display_labels=cluster_labels,
        xticks_rotation='vertical',
        cmap='Blues',
        ax=ax,
        normalize=normalize,
        include_values=include_values
    )
    plt.show()
    return c


def main():
    datapath = '/Users/znamenp/data/mouse_VISp_gene_expression_matrices_2018-06-14/'
    exons_matrix, cluster_ids, cluster_means, cluster_labels, gene_names = load_data_tasic_2018(datapath)

    #datapath = '/Users/znamenp/data/'
    #exons_matrix, cluster_ids, cluster_means, cluster_labels, gene_names = load_data_yao_2021(datapath)
    gene_set = [ 'Slc17a7', 'Gad1', 'Fezf2', 'Enpp2', 'Pvalb', 'Sst', 'Npy', 'Htr3a', 'Foxp2',
         'Rorb', 'Pcp4', 'Rab3b', 'Rgs4', 'Cdh13', 'Cck', 'Kcnip4', 'Igfbp4', 'Cnr1',
         'Serpini1', 'Sema3e', 'Thsd7a', 'Id2', 'Gabra1', 'Crh', 'Cd24a', 'Arpp21',
         'Lamp5', 'Cartpt', 'Etv1', 'Galnt14', 'Luzp2', 'Car4', 'Nrn1', 'Reln', 'Ptgds',
         'Lypd1', 'Rspo1', 'Pnoc', 'Alcam', 'Pde1a', 'Cxcl14', 'Ptprt', 'Grp', 'Bdnf',
         'Lipa', 'Igfn1', 'Col23a1', 'Tcap', 'Ptger4', 'Pygm', 'Lrrc38', 'Gpx8', 'Slc18a3',
         'Cadm2', 'Coro2b', 'Ccdc109b', 'Glp2r', 'Ifit3', 'Lama3', 'Cebpd', 'Cfap58',
         'Fst', 'Gdf10', 'Dusp2', 'Dhx58', 'Slc7a11', 'Crispld2', 'Map3k15', 'Sall1',
         'Lgr5', 'Pcdhgb6', 'Amigo3', 'Gdpd2', 'Chia1', 'Crb2', 'Ect2', 'Peak1os', 'Scml4',
         'Ccdc172', 'Gcgr', 'Cdca3', 'Kcne4', 'Adam12', 'Tbc1d22bos', 'Foxh1', 'Hspb8',
         'Cenpn', 'Kbtbd12', 'Gjb6', 'Svopl', 'Kif14', 'Fbxl13', 'Mei1', 'Helb',
         'Slc25a30', 'Tspan2os', 'D5Ertd615e', 'Nodal', 'Bst1', 'Agtr1a', 'Abca12',
         'Aadat', 'Zbbx', 'Dmrtb1', 'Bpifc', 'C5ar1', 'Chrnb4', 'Kif19a', 'Steap1', 'Klf4' ]

    include_genes = np.isin(np.array(gene_names), gene_set)
    include_genes[::50] = True
    gene_names = gene_names[include_genes]
    exons_matrix = exons_matrix[:,include_genes]
    cluster_means = cluster_means[:,include_genes]
    exons_matrix, cluster_means = resample_counts(
        exons_matrix, cluster_means, efficiency=0.01
    )
    probs = compute_cluster_probabilities(exons_matrix, cluster_means, nu=0.001)
    return optimize_gene_set(probs, cluster_ids, gene_names)
