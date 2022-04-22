import numpy as np
import defopt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from .io import load_data_yao_2021, load_data_tasic_2018
import time


def compute_means(exons_df, classify_by, gene_filter='\d'):
    exons_matrix = exons_df.filter(regex=gene_filter).to_numpy() # n cells x n genes matrix
    # names of columns containing expression data are integer numbers
    expression_by_cluster = exons_df.groupby([classify_by]).mean().filter(regex=gene_filter)
    cluster_means = expression_by_cluster.to_numpy()   # n clusters x n genes matrix

    cluster_ids = np.empty(len(exons_df[classify_by]), dtype=int)
    for i, cluster in enumerate(exons_df[classify_by]):
        cluster_ids[i] = np.nonzero(expression_by_cluster.index == cluster)[0]
    cluster_labels = expression_by_cluster.index
    return exons_matrix, cluster_ids, cluster_means, cluster_labels


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


def optimize_gene_set(cluster_probs, cluster_ids, gene_names, gene_set=(),
                      niter=100, subsample_cells=1):
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
        subsample_cells: whether to subsample cells on each iteration. If <1,
            then a given fraction of cells will be selected.

    Returns:
        Boolean numpy.array of included genes at the end of optimization.
        List of boolean arrays for every step of optimizations
        List of accuracies at every step of optimization

    """
    gene_set_history = []
    accuracy_history = []
    include_genes = np.isin(np.array(gene_names), gene_set)
    for i in range(niter):
        if subsample_cells < 1:
            cell_idx = np.random.rand(cluster_probs.shape[0]) < subsample_cells
            b, accuracy = next_best_gene(include_genes, cluster_probs[cell_idx,:,:], cluster_ids[cell_idx])
        else:
            b, accuracy = next_best_gene(include_genes, cluster_probs, cluster_ids)

        include_genes[b] = True
        print(f'added {gene_names.iloc[b]}, accuracy = {accuracy}')
        if i > 0:
            if subsample_cells < 1:
                r, accuracy = remove_bad_gene(include_genes, cluster_probs[cell_idx,:,:], cluster_ids[cell_idx])
            else:
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


def evaluate_gene_set(train_set, test_set, gene_set, gene_names):
    """ Plot classification accuracy while incrementally growing the gene set """
    ngenes = len(gene_set)
    accuracy_train = np.empty(ngenes)
    accuracy_test = np.empty(ngenes)
    for i in range(ngenes):
        cluster_assignments_train = classify_cells(
            train_set['exons_matrix'],
            train_set['cluster_means'],
            gene_set[:i], gene_names
        )
        accuracy_train[i] = np.mean(cluster_assignments_train == train_set['cluster_ids'])
        cluster_assignments_test = classify_cells(
            test_set['exons_matrix'],
            train_set['cluster_means'],
            gene_set[:i], gene_names
        )
        accuracy_test[i] = np.mean(cluster_assignments_test == test_set['cluster_ids'])

    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.xlabel('# genes')
    plt.ylabel('accuracy')
    plt.show()
    return accuracy_train, accuracy_test


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
        normalize = None
        include_values = True
    else:
        normalize = 'true'
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


def train_test_split(exons_df, classify_by, gene_filter, efficiency=0.01):
    train = {}
    test = {}
    train['exons_matrix'], train['cluster_ids'], train['cluster_means'], cluster_labels = compute_means(
        exons_df.iloc[::2],
        classify_by=classify_by,
        gene_filter=gene_filter
    )
    train['exons_matrix'], train['cluster_means'] = resample_counts(
        train['exons_matrix'], train['cluster_means'], efficiency=efficiency
    )
    test['exons_matrix'], test['cluster_ids'], test['cluster_means'], cluster_labels = compute_means(
        exons_df.iloc[1::2],
        classify_by=classify_by,
        gene_filter=gene_filter
    )
    test['exons_matrix'], test['cluster_means'] = resample_counts(
        test['exons_matrix'], test['cluster_means'], efficiency=efficiency
    )
    return train, test, cluster_labels


def main(savepath, *, efficiency=0.01,
         datapath='/camp/lab/znamenskiyp/home/shared/resources/allen2018/',
         subsample=1, classify='cluster'):
    """
    Optimize gene set for cell classification.

    Args:
        savepath (str): where to save output
        efficiency (float): simulated efficiency of in situ sequencing
        datapath (str): location of reference data
        subsample (float): whether to subsample cells on each iteration of
            gene selection
        classify (str): which field to use for classification. Default: 'cluster'

    """
    print('loading reference data...', flush=True)
    exons_df, gene_names = load_data_tasic_2018(datapath)
    exons_matrix, cluster_ids, cluster_means, cluster_labels = compute_means(
        exons_df,
        classify_by=classify,
        gene_filter='\d'
    )
    print('resampling reference data...', flush=True)
    exons_matrix, cluster_means = resample_counts(exons_matrix, cluster_means, efficiency=efficiency)
    print('computing cluster probabilities...', flush=True)
    probs = compute_cluster_probabilities(exons_matrix, cluster_means, nu=0.001)
    print('optimizing gene set...', flush=True)
    include_genes, gene_set_history, accuracy_history = optimize_gene_set(
        probs, cluster_ids, gene_names, subsample_cells=subsample
    )
    timestr = time.strftime("%Y%m%d_%H%M%S")
    print(gene_names[include_genes])
    np.savez(
        f'{savepath}genes_{classify}_e{efficiency}_s{subsample}_{timestr}.npz',
        include_genes=include_genes,
        gene_set_history=gene_set_history,
        accuracy_history=accuracy_history,
        gene_names=gene_names
    )


def entry_point():
    defopt.run(main)
