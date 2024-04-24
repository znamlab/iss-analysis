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

    if 'chamber' in main_folder.name:  # single chamber
        chambers = [acquisition_folder]
    else:  # mouse folder
        chambers = list(main_folder.glob("chamber_*"))
        # make the path relative to project, like acquisition_folder
        root = str(main_folder)[:-len(acquisition_folder)]
        chambers = [str(chamber.relative_to(root)) for chamber in chambers]
    
    for chamber in chambers:
        ops = issp.io.load.load_ops(chamber)
        data_folder = issp.io.get_processed_path(chamber)
        all_barcode_spots = []
        if "use_rois" not in ops:
            rois = issp.io.get_roi_dimensions(chamber)[:,0]
        else:
            rois = ops["use_rois"]
        for roi in rois:
            barcode_spots = pd.read_pickle(data_folder / f"barcode_round_spots_{roi}.pkl")
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

if __name__ == "__main__":
    barcode_spots, gmm, all_barcode_spots = get_barcodes(
        acquisition_folder="becalia_rabies_barseq/BRAC8498.3e",
        mean_intensity_threshold=0.01,
        dot_product_score_threshold=0.2,
        mean_score_threshold=0.75,
    )
    print(barcode_spots.head())
