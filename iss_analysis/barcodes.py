import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

import iss_preprocess as issp
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

    for chamber in chambers:
        ops = issp.io.load.load_ops(chamber)
        data_folder = issp.io.get_processed_path(chamber)
        all_barcode_spots = []
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
    means_init = np.nanpercentile(data, [1, 99], axis=0)
    gmm = GaussianMixture(n_components=2, means_init=means_init)
    gmm.fit(data)

    labels = gmm.predict(all_barcode_spots[metrics].values)
    barcode_spots = all_barcode_spots[labels == 1].copy()
    return barcode_spots, gmm, all_barcode_spots


def correct_barcode_sequences(spots, max_edit_distance=2):
    """Error correct barcode sequences.

    Args:
        spots (pandas.DataFrame): DataFrame of spots with a "sequence" column.
        max_edit_distance (int): Maximum edit distance for correction. Default is 2.

    Returns:
        pandas.DataFrame: DataFrame with corrected sequences and bases.
    """

    sequences = np.stack(spots["sequence"].to_numpy())
    unique_sequences, counts = np.unique(sequences, axis=0, return_counts=True)
    # sort sequences according to abundance
    order = np.flip(np.argsort(counts))
    unique_sequences = unique_sequences[order]
    counts = counts[order]

    corrected_sequences = unique_sequences.copy()
    reassigned = np.zeros(corrected_sequences.shape[0])
    for i, sequence in enumerate(unique_sequences):
        # if within edit distance and lower in the list (i.e. lower abundance),
        # then update the sequence
        edit_distance = np.sum((unique_sequences - sequence) != 0, axis=1)
        sequences_to_correct = np.logical_and(
            edit_distance <= max_edit_distance, np.logical_not(reassigned)
        )
        sequences_to_correct[: i + 1] = False
        corrected_sequences[sequences_to_correct, :] = sequence
        reassigned[sequences_to_correct] = True

    for original_sequence, new_sequence in zip(unique_sequences, corrected_sequences):
        if not np.array_equal(original_sequence, new_sequence):
            sequences[
                np.all((sequences - original_sequence) == 0, axis=1), :
            ] = new_sequence

    spots["corrected_sequence"] = [seq for seq in sequences]
    spots["corrected_bases"] = [
        "".join(BASES[seq]) for seq in spots["corrected_sequence"]
    ]
    return spots


if __name__ == "__main__":
    barcode_spots, gmm, all_barcode_spots = get_barcodes(
        acquisition_folder="becalia_rabies_barseq/BRAC8498.3e",
        mean_intensity_threshold=0.01,
        dot_product_score_threshold=0.2,
        mean_score_threshold=0.75,
    )
    print(barcode_spots.head())
