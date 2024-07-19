import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from numba import njit, prange
from tqdm import tqdm
from collections.abc import Callable

import iss_preprocess as issp
from iss_preprocess.call import BASES
from functools import partial
from ..io import get_chamber_datapath


def get_barcodes(
    acquisition_folder,
    n_components=2,
    valid_components=None,
    mean_intensity_threshold=0.01,
    dot_product_score_threshold=0.2,
    mean_score_threshold=0.75,
):
    """
    Get barcode spots from the given data path.

    Args:
        acquisiton_folder (str): The relative path to the data, either a single chamber
            or a mouse folder
        n_components (int): The number of clusters for the Gaussian Mixture Model.
            Default is 2.
        valid_components (list): The list of valid components. If None, keep only
            the last. Default is None.
        mean_intensity_threshold (float): The threshold for mean intensity. Default is
            0.01.
        dot_product_score_threshold (float): The threshold for dot product score.
            Default is 0.2.
        mean_score_threshold (float): The threshold for mean score. Default is 0.75.

    Returns:
        barcode_spots (pd.DataFrame): DataFrame containing the barcode spots.
        gmm (GaussianMixture): The trained Gaussian Mixture Model.
        all_barcode_spots (pd.DataFrame): DataFrame containing all barcode spots.
    """
    chambers = get_chamber_datapath(acquisition_folder)
    all_barcode_spots = []
    for chamber in chambers:
        ops = issp.io.load.load_ops(chamber)
        data_folder = issp.io.get_processed_path(chamber)

        if "use_rois" not in ops:
            rois = issp.io.get_roi_dimensions(chamber)[:, 0]
        else:
            rois = ops["use_rois"]
        for roi in rois:
            barcode_spots = pd.read_pickle(
                data_folder / f"barcode_round_spots_{roi}.pkl"
            )
            barcode_spots["roi"] = roi
            barcode_spots["chamber"] = chamber.split("/")[-1]
            all_barcode_spots.append(barcode_spots)
    all_barcode_spots = pd.concat(all_barcode_spots, ignore_index=True)

    # filter the utter crap
    all_barcode_spots = all_barcode_spots[
        (all_barcode_spots.mean_intensity > mean_intensity_threshold)
        & (all_barcode_spots.dot_product_score > dot_product_score_threshold)
        & (all_barcode_spots.mean_score > mean_score_threshold)
    ].copy()

    # Do a GMM on the 4 metrics dot_product_score, spot_score, mean_intensity, mean_score,
    # with just 2 clusters
    # Extract the four metrics from the dataframe
    metrics = ["dot_product_score", "spot_score", "mean_intensity", "mean_score"]
    skip = len(all_barcode_spots) // 10000
    data = all_barcode_spots[metrics].values[::skip]

    # Perform GMM with two clusters, 0 low values, 1 high values
    percentiles = list(np.linspace(1, 30, n_components - 1)) + [99]
    means_init = np.nanpercentile(data, percentiles, axis=0)
    gmm = GaussianMixture(
        n_components=n_components, means_init=means_init, random_state=123
    )
    gmm.fit(data)

    labels = gmm.predict(all_barcode_spots[metrics].values)
    all_barcode_spots["gmm_label"] = labels
    if valid_components is None:
        valid_components = [n_components - 1]
    elif not isinstance(valid_components, list):
        valid_components = [valid_components]
    barcode_spots = all_barcode_spots[np.isin(labels, valid_components)].copy()
    return barcode_spots, gmm, all_barcode_spots


def correct_barcode_sequences(
    spots,
    max_edit_distance=2,
    weights=None,
    minimum_match=None,
    return_merge_dict=False,
    verbose=True,
):
    """Error correct barcode sequences.

    Args:
        spots (pandas.DataFrame): DataFrame of spots with a "sequence" column.
        max_edit_distance (int, optional): Maximum edit distance for correction. Default
            is 2.
        weights (numpy.ndarray, optional): Weights for the sequences. Default is None.
        minimum_match (int, optional): Minimum number of matching bases. Default is
            None.
        return_merge_dict (bool, optional): Whether to return a dictionary with the
            original sequences as keys and the corrected sequences as values. Default is
            False.
        verbose (bool, optional): Whether to print the progress. Default is True.

    Returns:
        pandas.DataFrame: DataFrame with corrected sequences and bases.
        dict: Dictionary with the original sequences as keys and the corrected
            sequences as values. Only returned if return_merge_dict is True.

    """

    sequences = np.stack(spots["sequence"].to_numpy())
    if weights is None:
        weights = np.ones(sequences.shape[1])
    else:
        assert np.array(weights).shape == (
            sequences.shape[1],
        ), "Weights must have the same length as the sequences"

    # unique would split NaNs into different sequences, so we replace them by a value
    seq_no_nan = np.nan_to_num(sequences, nan=4)
    unique_sequences, counts = np.unique(seq_no_nan, axis=0, return_counts=True)
    if verbose:
        print(f"{unique_sequences.shape[0]} unique sequences found.", flush=True)
    # sort sequences according to abundance
    order = np.flip(np.argsort(counts))
    unique_sequences = unique_sequences[order]
    counts = counts[order]

    # move sequences with NaNs to the end so that they cannot match with anything but
    # other sequences with NaNs
    nan_sequences = np.sum(unique_sequences == 4, axis=1) > 0
    unique_sequences = np.concatenate(
        [unique_sequences[~nan_sequences], unique_sequences[nan_sequences]]
    )

    # assign outputs
    corrected_sequences = unique_sequences.copy()
    reassigned = np.zeros(corrected_sequences.shape[0])
    if return_merge_dict:
        merge_dict = {}

    iterator = enumerate(unique_sequences)
    if verbose:
        print("Finding matching sequences ...", flush=True)
        iterator = tqdm(iterator, total=len(unique_sequences))
    for i, sequence in iterator:
        # if within edit distance and lower in the list (i.e. lower abundance),
        # then update the sequence
        # note that unique_sequences has no NaN but sequence does
        diff_with_nan = unique_sequences - sequence
        differences = (diff_with_nan != 0).astype(float)
        differences[np.isnan(diff_with_nan)] = np.nan

        edit_distance = np.nansum(differences.astype(float) * weights, axis=1)
        sequences_to_correct = np.logical_and(
            edit_distance <= max_edit_distance, np.logical_not(reassigned)
        )
        if minimum_match is not None:
            matches = np.sum(diff_with_nan == 0, axis=1) > minimum_match
            sequences_to_correct = np.logical_and(sequences_to_correct, matches)

        sequences_to_correct[: i + 1] = False  # the first have already been corrected
        corrected_sequences[sequences_to_correct, :] = sequence
        if return_merge_dict:
            merge_dict[str(sequence)] = [unique_sequences[sequences_to_correct], []]
        reassigned[sequences_to_correct] = True

    iterator = zip(unique_sequences, corrected_sequences)
    if verbose:
        print(
            f"{len(np.unique(corrected_sequences, axis=0))} unique sequences"
            + " after correction.",
            flush=True,
        )
        print(f"{int(np.sum(reassigned))} sequences corrected.")
        print("Correcting sequences ...", flush=True)
        iterator = tqdm(iterator, total=len(unique_sequences))
    for original_sequence, new_sequence in iterator:
        if not np.array_equal(original_sequence, new_sequence):
            to_change = np.all((seq_no_nan - original_sequence) == 0, axis=1)
            seq_no_nan[to_change, :] = new_sequence
            if return_merge_dict:
                merge_dict[str(original_sequence)][1].append(to_change.sum())

    if verbose:
        print("Adding to spots ...", flush=True)
    # seq_no_nan is corrected, but has "4" instead of NaNs. Use that to select bases
    bases_list = list(BASES) + ["N"]
    spots["corrected_bases"] = [
        "".join([bases_list[int(s)] for s in seq]) for seq in seq_no_nan
    ]
    # but put back nans in the corrected sequences
    sequences = np.array(seq_no_nan).astype(float)
    sequences[sequences == 4] = np.nan
    spots["corrected_sequence"] = [seq for seq in sequences]

    if return_merge_dict:
        return spots, merge_dict
    return spots


def error_per_round(
    spot_df,
    edit_distance=1,
    spot_count_threshold=30,
    sequence_column="sequence",
    filter_column="bases",
):
    """Calculate the error per round for each barcode.

    Args:
        spot_df (pd.DataFrame): DataFrame with the spots.
        edit_distance (int): Maximum edit distance for correction. Default is 1.
        spot_count_threshold (int): Minimum number of spots for a barcode to be
            considered.
        sequence_column (str): Name of the column with the sequences. Default is
            'sequence'.
        filter_column (str): Name of the column to filter the sequences. Default is
            'bases'.

    Returns:
        dict: Dictionary with the errors per round.

    """
    # count the number of unique sequence for each rolonie
    rol_cnt = spot_df[filter_column].value_counts()
    good = rol_cnt[rol_cnt.values > 30].index
    # remove the on with a N
    good = [g for g in good if "N" not in g]

    print(f"Found {len(good)} barcodes with more than {spot_count_threshold} spots")
    ch_gp = spot_df.groupby("chamber")
    nroi_per_ch = [len(gp["roi"].unique()) for _, gp in ch_gp]
    with tqdm(total=sum(nroi_per_ch) * len(good)) as pbar:
        all_errors = dict()
        base_list = list(BASES) + ["N"]
        for chamber, cdf in spot_df.groupby("chamber"):
            all_errors[chamber] = dict()
            for roi, df in cdf.groupby("roi"):
                pbar.set_description(f"Processing {chamber}, roi {roi}")
                sequences = np.stack(df[sequence_column].to_numpy())
                error_along_sequence = np.zeros((len(good), sequences.shape[1]))
                for ibar, barcode in enumerate(good):
                    seq = np.array([base_list.index(b) for b in barcode]).astype(float)
                    seq[seq == len(base_list)] = np.nan
                    diff = sequences - seq
                    nan_mask = np.logical_not(np.isnan(diff)).astype(float)
                    diff = (diff != 0) * nan_mask  # put all nan to 0 difference
                    bad_barcode = diff[np.sum((diff != 0), axis=1) <= edit_distance]
                    error_along_sequence[ibar] = np.any(bad_barcode != 0, axis=0)
                    pbar.update(1)
                all_errors[chamber][roi] = error_along_sequence

    return all_errors
