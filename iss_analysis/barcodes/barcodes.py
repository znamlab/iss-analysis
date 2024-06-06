import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

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


def assign_barcodes_to_masks(
    spots,
    masks,
    p=0.9,
    m=0.1,
    background_spot_prior=0.0001,
    spot_distribution_sigma=50,
    max_iterations=100,
    distance_threshold=200,
    verbose=False,
    base_column="bases",
    debug=False,
):
    """Assign barcodes to masks using a probabilistic model.

    Args:
        spots (pd.DataFrame): DataFrame with the spots. Must contain the columns 'x',
            'y', and `base_column`.
        mask_centres (pd.DataFrame): DataFrame with the mask centres. Must contain the
            columns 'x' and 'y'.
        p (float): Power of the spot count prior. Default is 0.9.
        m (float): Length scale of the spot count prior. Default is 0.1.
        background_spot_prior (float): Prior for the background spots. Default is 0.0001.
        spot_distribution_sigma (float): Sigma for the spot distribution. Default is 20.
        max_iterations (int): Maximum number of iterations. Default is 100.
        distance_threshold (float): Threshold for the distance in pixels between spots
            and masks. Default is 50.
        verbose (bool): Whether to print the progress. Default is False.
        base_column (str): Name of the column with the bases. Default is 'bases'.
        debug (bool): Whether to return debug information. Default is False.

    Returns:
        np.ndarray: 1D array with the mask assignment if debug is False. Otherwise 2D
            array with the mask assignment for each iteration.
    """
    # compute the distance between each spot and each mask
    mask_centers = masks[["x", "y"]].values
    spot_positions = spots[["x", "y"]].values
    distances = np.linalg.norm(
        spot_positions[:, None, :] - mask_centers[None, :, :], axis=2
    )
    # we will use only spots that are close to at least one mask.
    spots_in_range = np.any(distances < distance_threshold, axis=1)
    spots = spots[spots_in_range].copy().reset_index(drop=True)
    distances = distances[spots_in_range]

    if verbose:
        print(f"Using {spots_in_range.sum()}/{len(spots_in_range)} spots in range")
        print(f"Assigning {len(spots)} spots to {len(mask_centers)} masks")

    log_background_spot_prior = np.log(background_spot_prior)
    # compute the probability of the spot being in each mask given the distance
    log_spot_distribution = -0.5 * (distances / spot_distribution_sigma) ** 2
    # first assign each spot to its nearest mask
    mask_assignment = np.argmax(log_spot_distribution, axis=1)
    # compute the change in likelihood for each spot if it is assigned to each mask
    log_likelihood_change = np.zeros((len(mask_centers)))
    barcodes = spots[base_column].unique()
    if verbose:
        print(f"Found {len(barcodes)} unique barcodes")
    sp_prior = partial(_spot_count_prior, p=p, m=m)
    if debug:
        output = [mask_assignment]
    for iter in range(max_iterations):
        spots_moved = 0
        for barcode in barcodes:
            # count the number of spots assigned to each mask
            spot_is_this_barcode = spots[base_column] == barcode
            this_barcode = mask_assignment[spot_is_this_barcode]
            mask_counts = np.bincount(
                this_barcode[this_barcode >= 0], minlength=len(mask_centers)
            )
            for spot_index in spots[spot_is_this_barcode].index:
                # reset likelihood change
                log_likelihood_change += -np.inf
                current_mask = mask_assignment[spot_index]
                mask_is_closeby = distances[spot_index] < distance_threshold
                assert np.sum(mask_is_closeby) > 0
                for new_mask in np.where(mask_is_closeby)[0]:
                    current_likelihood, new_likelihood = _calc_log_likelihoods(
                        current_mask,
                        new_mask,
                        log_background_spot_prior,
                        sp_prior,
                        mask_counts,
                        spot_index,
                        log_spot_distribution,
                    )
                    log_likelihood_change[new_mask] = (
                        new_likelihood - current_likelihood
                    )
                # compute likelihood for switching to a background spot
                if current_mask == -1:
                    log_likelihood_change_background = 0
                else:
                    assert mask_counts[current_mask] > 0
                    log_likelihood_change_background = (
                        log_background_spot_prior
                        + sp_prior(mask_counts[current_mask] - 1)
                        - sp_prior(mask_counts[current_mask])
                        - log_spot_distribution[spot_index, current_mask]
                    )
                # if max of log_likelihood_change is higher than log_likelihood_change_background
                # then assign the spot to the mask that gives the highest increase in likelihood
                if np.max(log_likelihood_change) > log_likelihood_change_background:
                    new_mask_assignment = np.argmax(log_likelihood_change)
                    if new_mask_assignment != current_mask:
                        mask_assignment[spot_index] = new_mask_assignment
                        if current_mask != -1:
                            assert mask_counts[current_mask] > 0
                            mask_counts[current_mask] -= 1
                        mask_counts[new_mask_assignment] += 1
                        spots_moved += 1
                elif current_mask != -1:
                    mask_assignment[spot_index] = -1
                    assert mask_counts[current_mask] > 0
                    mask_counts[current_mask] -= 1
                    spots_moved += 1
        if verbose:
            print(f"Iteration {iter}: {spots_moved} spots resassigned")
        if debug:
            output.append(mask_assignment.copy())
        if spots_moved == 0:
            for barcode in barcodes:
                # count the number of spots assigned to each mask
                valid_spots = spots[base_column] == barcode
                this_barcode = mask_assignment[valid_spots]
                mask_counts = np.bincount(
                    this_barcode[this_barcode >= 0], minlength=len(mask_centers)
                )
                for current_mask in range(len(mask_centers)):
                    if mask_counts[current_mask] > 0:
                        # likelihood change if all spots in the mask are background spots
                        log_likelihood_change_background = (
                            log_background_spot_prior * mask_counts[current_mask]
                            - sp_prior(mask_counts[current_mask])
                            - log_spot_distribution[
                                (mask_assignment == current_mask) & valid_spots,
                                current_mask,
                            ].sum()
                        )
                        if log_likelihood_change_background > 0:
                            mask_assignment[
                                (mask_assignment == current_mask) & valid_spots
                            ] = -1
                            spots_moved += mask_counts[current_mask]
                            mask_counts[current_mask] = 0
            if debug:
                output.append(mask_assignment.copy())
            if spots_moved == 0:
                break

    # recreate a full assignment with -2 for spots that are not assigned
    def _recreate_full_assignment(mask_assignment, spots_in_range):
        full_assignment = np.full(len(spots_in_range), -2)
        full_assignment[spots_in_range] = mask_assignment
        return full_assignment

    if debug:
        for m in range(len(output)):
            output[m] = _recreate_full_assignment(output[m], spots_in_range)
        return np.vstack(output)
    mask_assignment_full = _recreate_full_assignment(mask_assignment, spots_in_range)
    return mask_assignment_full


def _spot_count_prior(nspots, p=0.9, m=0.1):
    """Compute the prior for the number of spots in a mask.

    Args:
        nspots (int): Number of spots in the mask.
        p (float): Power of the spot count prior. Default is 0.9.
        m (float): Lambda of the spot count prior. Default is 0.1.

    Returns:
        float: Prior for the number of spots in the mask.

    """
    return -(nspots**p) / m


def _calc_log_likelihoods(
    current_mask,
    new_mask,
    log_background_spot_prior,
    sp_prior,
    mask_counts,
    spot_index,
    log_spot_distribution,
):
    """Inner function of assign_barcodes_to_masks.

    Calculate the likelihoods for the current and new mask.

    Args:
        current_mask (int): The current mask.
        new_mask (int): The new mask.
        log_background_spot_prior (float): The log background spot prior.
        sp_prior (function): The spot count prior function.
        mask_counts (np.ndarray): The mask counts.
        spot_index (int): The spot index.
        log_spot_distribution (np.ndarray): The log spot distribution.

    Returns:
        tuple: Tuple with the current and new likelihoods.
    """
    if current_mask == -1:
        # changing from a background spot to a mask
        current_likelihood = log_background_spot_prior + sp_prior(mask_counts[new_mask])
        new_likelihood = log_spot_distribution[spot_index, new_mask] + sp_prior(
            mask_counts[new_mask] + 1
        )
    else:
        current_likelihood = (
            log_spot_distribution[spot_index, current_mask]
            + sp_prior(mask_counts[current_mask])
            + sp_prior(mask_counts[new_mask])
        )
        new_likelihood = (
            log_spot_distribution[spot_index, new_mask]
            + sp_prior(mask_counts[current_mask] - 1)
            + sp_prior(mask_counts[new_mask] + 1)
        )
    return current_likelihood, new_likelihood
