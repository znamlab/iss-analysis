import pandas as pd
import flexiznam as flz
from znamutils import slurm_it
from .barcodes import get_barcodes, correct_barcode_sequences


@slurm_it(conda_env="iss-preprocess", print_job_id=True)
def error_correct_acquisition(
    project,
    mouse_name,
    n_components=2,
    valid_components=None,
    mean_intensity_threshold=0.01,
    dot_product_score_threshold=0.2,
    mean_score_threshold=0.75,
    max_edit_distance=2,
    weights=None,
    minimum_match=None,
    verbose=True,
    conflicts="abort",
):
    """Error correct barcode sequences for an acquisition.

    Args:
        project (str): The name of the project on flexilims.
        mouse_name (str): The name of the mouse.
        n_components (int, optional): The number of clusters for the Gaussian Mixture
            Model. Default is 2.
        valid_components (list, optional): The list of valid components. If None, keep
            only the last. Default is None.
        mean_intensity_threshold (float, optional): The threshold for mean intensity.
            Default is 0.01.
        dot_product_score_threshold (float, optional): The threshold for dot product
            score. Default is 0.2.
        mean_score_threshold (float, optional): The threshold for mean score. Default is
            0.75.
        max_edit_distance (int, optional): The maximum edit distance for the sequences.
            Default is 2.
        weights (numpy.ndarray, optional): Weights for the sequences. Default is None.
        minimum_match (int, optional): Minimum number of matching bases. Default is
            None.
        verbose (bool, optional): Whether to print the progress. Default is True.
        conflicts (str, optional): The conflict resolution strategy. Default is "abort".

    Returns:
        pandas.DataFrame: DataFrame with corrected sequences and bases.
    """
    flm_sess = flz.get_flexilims_session(project_id=project)
    mouse = flz.get_entity("mouse", name=mouse_name, flexilims_session=flm_sess)

    attributes = dict(
        get_barcodes=dict(
            n_components=n_components,
            valid_components=valid_components,
            mean_intensity_threshold=mean_intensity_threshold,
            dot_product_score_threshold=dot_product_score_threshold,
            mean_score_threshold=mean_score_threshold,
        ),
        correct_barcode_sequences=dict(
            max_edit_distance=max_edit_distance,
            weights=weights,
            minimum_match=minimum_match,
            verbose=verbose,
        ),
    )

    # overwrite would crash if the dataset doesn't exist yet, we want to run with
    # conflicts=skip to avoid this and then decide to reload or not
    err_corr_ds = flz.Dataset.from_origin(
        origin_id=mouse.id,
        dataset_type="error_corrected_barcodes",
        flexilims_session=flm_sess,
        conflicts=conflicts if conflicts != "overwrite" else "skip",
        extra_attributes=attributes,
        ignore_attributes=["started", "ended"],
    )
    if conflicts == "skip" and (err_corr_ds.flexilims_status() == "up-to-date"):
        if verbose:
            print("Reloading error corrected barcode sequences")
        return pd.read_pickle(err_corr_ds.path_full)

    # We need to make sure that the dataset is created to avoid another job
    # to create the same dataset.
    err_corr_ds.extra_attributes["started"] = str(pd.Timestamp.now())
    err_corr_ds.update_flexilims(mode="overwrite")
    print(f"Started error correcting barcode sequences for {project}/{mouse_name}")
    print(f"Dataset {err_corr_ds.dataset_name} with ID {err_corr_ds.id}.")
    barcode_spots, gmm, all_barcode_spots = get_barcodes(
        acquisition_folder=f"{project}/{mouse_name}",
        **attributes["get_barcodes"],
    )
    corrected_spots = correct_barcode_sequences(
        barcode_spots,
        **attributes["correct_barcode_sequences"],
    )
    err_corr_ds.path = err_corr_ds.path.with_suffix(".pkl")
    corrected_spots.to_pickle(err_corr_ds.path_full)
    # Add a timestamp of end time
    err_corr_ds.extra_attributes["ended"] = str(pd.Timestamp.now())
    err_corr_ds.update_flexilims(mode="overwrite")

    if verbose:
        print(f"Saved error corrected barcode sequences to {err_corr_ds.path_full}")
    return corrected_spots
