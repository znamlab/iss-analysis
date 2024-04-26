import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import iss_preprocess as issp
from iss_preprocess.call import BASES
import iss_analysis as issa


def get_barcodes(
    acquisition_folder,
    mean_intensity_threshold=0.01,
    dot_product_score_threshold=0.2,
    mean_score_threshold=0.75,
):
    """
    Get barcode spots from the given data path.

    Args:
        acquisiton_folder (str): The relative path to the data, either a single chamber
            or a mouse folder
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
    main_folder = issp.io.get_processed_path(acquisition_folder)
    if not main_folder.exists():
        raise FileNotFoundError(f"Folder {main_folder} does not exist")

    if "chamber" in main_folder.name:  # single chamber
        chambers = [acquisition_folder]
    else:  # mouse folder
        chambers = list(main_folder.glob("chamber_*"))
        # make the path relative to project, like acquisition_folder
        root = str(main_folder)[: -len(acquisition_folder)]
        chambers = [str(chamber.relative_to(root)) for chamber in chambers]
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
    means_init = np.nanpercentile(data, [1, 20, 99], axis=0)
    gmm = GaussianMixture(n_components=3, means_init=means_init)
    gmm.fit(data)

    labels = gmm.predict(all_barcode_spots[metrics].values)
    all_barcode_spots["gmm_label"] = labels
    barcode_spots = all_barcode_spots[labels == len(means_init) - 1].copy()
    return barcode_spots, gmm, all_barcode_spots


def correct_barcode_sequences(
    spots, max_edit_distance=2, weights=None, return_merge_dict=False, verbose=True
):
    """Error correct barcode sequences.

    Args:
        spots (pandas.DataFrame): DataFrame of spots with a "sequence" column.
        max_edit_distance (int): Maximum edit distance for correction. Default is 2.
        weights (numpy.ndarray): Weights for the sequences. Default is None.
        return_merge_dict (bool): Whether to return a dictionary with the original
            sequences as keys and the corrected sequences as values. Default is False.
        verbose (bool): Whether to print the progress. Default is True.

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

    unique_sequences, counts = np.unique(sequences, axis=0, return_counts=True)
    if verbose:
        print(f"{unique_sequences.shape[0]} unique sequences found.", flush=True)
    # sort sequences according to abundance
    order = np.flip(np.argsort(counts))
    unique_sequences = unique_sequences[order]
    counts = counts[order]

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
        differences = ((unique_sequences - sequence) != 0).astype(float)
        edit_distance = np.sum(differences * weights, axis=1)

        sequences_to_correct = np.logical_and(
            edit_distance <= max_edit_distance, np.logical_not(reassigned)
        )
        sequences_to_correct[: i + 1] = False  # the first have already been corrected
        corrected_sequences[sequences_to_correct, :] = sequence
        if return_merge_dict:
            merge_dict[tuple(sequence)] = [unique_sequences[sequences_to_correct], []]
        reassigned[sequences_to_correct] = True

    iterator = zip(unique_sequences, corrected_sequences)
    if verbose:
        print(
            f"{len(np.unique(corrected_sequences, axis=0))} unique sequences after correction.",
            flush=True,
        )
        print(f"{int(np.sum(reassigned))} sequences corrected.")
        print("Correcting sequences ...", flush=True)
        iterator = tqdm(iterator, total=len(unique_sequences))
    for original_sequence, new_sequence in iterator:
        if not np.array_equal(original_sequence, new_sequence):
            to_change = np.all((sequences - original_sequence) == 0, axis=1)
            sequences[to_change, :] = new_sequence
            if return_merge_dict:
                merge_dict[tuple(original_sequence)][1].append(to_change.sum())

    if verbose:
        print("Adding to spots ...", flush=True)
    spots["corrected_sequence"] = [seq for seq in sequences]
    spots["corrected_bases"] = [
        "".join(BASES[seq]) for seq in spots["corrected_sequence"]
    ]
    if return_merge_dict:
        return spots, merge_dict
    return spots


def error_per_round(
    spot_df, edit_distance=1, spot_count_threshold=30, sequence_column="bases"
):
    """Calculate the error per round for each barcode.

    Args:
        spot_df (pd.DataFrame): DataFrame with the spots.
        edit_distance (int): Maximum edit distance for correction. Default is 1.
        spot_count_threshold (int): Minimum number of spots for a barcode to be
            considered.
        sequence_column (str): Name of the column with the sequences. Default is
            'bases'.

    Returns:
        dict: Dictionary with the errors per round.

    """
    # count the number of unique sequence for each rolonie
    rol_cnt = spot_df[sequence_column].value_counts()
    good = rol_cnt[rol_cnt.values > 30].index

    print(f"Found {len(good)} barcodes with more than {spot_count_threshold} spots")
    ch_gp = spot_df.groupby("chamber")
    nroi_per_ch = [len(gp["roi"].unique()) for _, gp in ch_gp]
    with tqdm(total=sum(nroi_per_ch) * len(good)) as pbar:
        all_errors = dict()
        base_list = list(BASES)
        for chamber, cdf in spot_df.groupby("chamber"):
            all_errors[chamber] = dict()
            for roi, df in cdf.groupby("roi"):
                pbar.set_description(f"Processing {chamber}, roi {roi}")
                sequences = np.stack(df["sequence"].to_numpy())
                error_along_sequence = np.zeros((len(good), sequences.shape[1]))
                for ibar, barcode in enumerate(good):
                    seq = [base_list.index(b) for b in barcode]
                    diff = sequences - seq
                    edits = np.sum(diff != 0, axis=1)
                    actual_errs = edits <= edit_distance
                    bad_barcode = diff[actual_errs]
                    error_along_sequence[ibar] = np.any(bad_barcode != 0, axis=0)
                    pbar.update(1)
                all_errors[chamber][roi] = error_along_sequence

    return all_errors
