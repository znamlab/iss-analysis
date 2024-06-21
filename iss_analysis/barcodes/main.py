import numpy as np
import pandas as pd
import gc
from pathlib import Path
from datetime import datetime

import flexiznam as flz
from znamutils import slurm_it
import iss_preprocess as issp

from .barcodes import get_barcodes, correct_barcode_sequences, assign_barcodes_to_masks
from ..io import get_chamber_datapath
from iss_preprocess.segment.cells import get_cell_masks


def _log(msg, verbose):
    if verbose:
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{time_str}: {msg}", flush=True)


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
    use_slurm=True,
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
        ignore_attributes=["started", "ended", "job_id"],
    )
    if conflicts == "skip" and (err_corr_ds.flexilims_status() == "up-to-date"):
        if verbose:
            print("Reloading error corrected barcode sequences")
        return pd.read_pickle(err_corr_ds.path_full)
    print(f"Error correcting barcode sequences for {project}/{mouse_name}")
    err_corr_ds.extra_attributes["started"] = str(pd.Timestamp.now())
    slurm_folder = Path.home() / "slurm_logs" / project / mouse_name
    scripts_name = err_corr_ds.dataset_name

    corrected_spots = run_error_correction(
        dataset_id=err_corr_ds.id,
        project=project,
        mouse_name=mouse_name,
        attributes=attributes,
        verbose=verbose,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=scripts_name,
    )
    if use_slurm:
        err_corr_ds.extra_attributes["job_id"] = corrected_spots
        # update already to mark job start and ensure the next one has a different
        # name
        err_corr_ds.update_flexilims(mode="overwrite")
        print(f"Started job {corrected_spots}")
    return corrected_spots


@slurm_it(conda_env="iss-preprocess", print_job_id=True)
def run_error_correction(
    dataset_id,
    project,
    mouse_name,
    attributes,
    verbose=True,
):
    """Run error correction for barcode sequences.

    Args:
        dataset_id (int): The ID of the dataset.
        project (str): The name of the project on flexilims.
        mouse_name (str): The name of the mouse.
        attributes (dict): The attributes for error correction.
        verbose (bool, optional): Whether to print the progress. Default is True.

    Returns:
        pandas.DataFrame: DataFrame with corrected sequences and bases.
    """

    flm_sess = flz.get_flexilims_session(project_id=project)
    err_corr_ds = flz.Dataset.from_flexilims(id=dataset_id, flexilims_session=flm_sess)
    # We need to make sure that the dataset is created to avoid another job
    # to create the same dataset.
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


def assign_barcode_all_chambers(
    project,
    mouse_name,
    error_correction_dsname=None,
    segmentation_mask_prefix="hybridisation_round_3_1_masks",
    p=0.9,
    m=0.1,
    background_spot_prior=0.0001,
    spot_distribution_sigma=50,
    max_iterations=100,
    distance_threshold=200,
    base_column="corrected_bases",
    verbose=True,
    conflicts="abort",
    use_slurm=True,
):
    """Assign barcodes to masks for all chambers.

    Args:
        project (str): The project name.
        mouse_name (str): The name of the mouse.
        error_correction_dsname (str, optional): The name of the error corrected barcode
            sequences dataset. Defaults: None.
        segmentation_mask_prefix (str, optional): The prefix of the segmentation mask
            files. Defaults: "hybridisation_round_3_1_masks".
        p (float): Power of the spot count prior. Default: 0.9.
        m (float): Length scale of the spot count prior. Default: 0.1.
        background_spot_prior (float): Prior for the background spots. Default: 0.0001.
        spot_distribution_sigma (float): Sigma for the spot distribution. Default: 20.
        max_iterations (int): Maximum number of iterations. Default: 100.
        distance_threshold (float): Threshold for the distance in pixels between spots
            and masks. Default: 50.
        base_column (str, optional): The column name for the corrected bases. Defaults:
            "corrected_bases".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        conflicts (str, optional): The conflict resolution strategy for dataset
            conflicts. Defaults: "abort".
        use_slurm (bool, optional): Whether to use SLURM for job submission.
            Defaults: True.

    Returns:
        pandas.DataFrame: The assigned barcodes for all masks.
    """

    print(f"Started assigning barcodes to masks for {project}/{mouse_name}")
    flm_sess = flz.get_flexilims_session(project_id=project)
    attributes = dict(
        error_correction_dsname=error_correction_dsname,
        p=p,
        m=m,
        background_spot_prior=background_spot_prior,
        spot_distribution_sigma=spot_distribution_sigma,
        max_iterations=max_iterations,
        distance_threshold=distance_threshold,
        base_column=base_column,
    )

    # compile the list of chamber/rois to run
    chambers = get_chamber_datapath(f"{project}/{mouse_name}")
    output = {}
    slurm_folder = Path.home() / "slurm_logs" / project / mouse_name / "assign_barcodes"
    slurm_folder.mkdir(exist_ok=True)
    for chamber_datapath in chambers:
        ops = issp.io.load_ops(chamber_datapath)
        use_rois = ops.get("use_rois", None)
        if use_rois is None:
            roi_dims = issp.io.get_roi_dimensions(chamber_datapath)
            use_rois = roi_dims[:, 0]
        chamber = Path(chamber_datapath).stem
        chamber_entity = flz.get_entity(
            name=f"{mouse_name}_{chamber}", flexilims_session=flm_sess
        )
        for roi in use_rois:
            attributes["roi"] = roi
            attributes["chamber"] = chamber
            out_ds = flz.Dataset.from_origin(
                origin_id=chamber_entity.id,
                dataset_type="barcodes_mask_assignment",
                flexilims_session=flm_sess,
                base_name=f"barcodes_mask_assignment_roi{roi}",
                conflicts=conflicts,
                extra_attributes=attributes,
                ignore_attributes=["started", "ended", "job_id"],
                verbose=verbose,
            )
            reload = True
            if conflicts != "skip":
                reload = False
            if reload and ("ended" not in out_ds.extra_attributes):
                reload = False
            if reload:
                if verbose:
                    print("Reloading error corrected barcode sequences")
                try:
                    output[(chamber, roi)] = pd.read_pickle(out_ds.path_full)
                    continue
                except FileNotFoundError:
                    print(f"File {out_ds.path_full} not found, running again")

            print(f"Assigning barcodes to {chamber}, {roi}")
            out_ds.path = out_ds.path.with_suffix(".pkl")
            out_ds.extra_attributes["started"] = str(pd.Timestamp.now())
            out_ds.update_flexilims(mode="overwrite")
            job_id = run_mask_assignment(
                project=project,
                mouse_name=mouse_name,
                assigned_datasets_name=out_ds.full_name,
                segmentation_mask_prefix=segmentation_mask_prefix,
                use_slurm=use_slurm,
                slurm_folder=slurm_folder,
                scripts_name=f"assign_barcodes_{chamber}_{roi}",
                **attributes,
            )
            output[(chamber, roi)] = job_id
    return output


@slurm_it(conda_env="iss-preprocess", print_job_id=True, slurm_options={"mem": "128GB"})
def run_mask_assignment(
    project,
    mouse_name,
    chamber,
    roi,
    assigned_datasets_name,
    error_correction_dsname,
    segmentation_mask_prefix,
    p,
    m,
    background_spot_prior,
    spot_distribution_sigma,
    max_iterations,
    distance_threshold,
    base_column,
    verbose=True,
):
    _log(
        f"Assigning barcodes to masks for {project}/{mouse_name}/{chamber}/{roi}",
        verbose,
    )
    flm_sess = flz.get_flexilims_session(project_id=project)
    error_dataset = flz.Dataset.from_flexilims(
        flexilims_session=flm_sess, name=error_correction_dsname
    )
    assigned_dataset = flz.Dataset.from_flexilims(
        flexilims_session=flm_sess, name=assigned_datasets_name
    )
    _log(f"Error corrected barcodes from {error_dataset.path_full}", verbose)
    _log(f"Flexilims dataset {assigned_dataset.full_name} with attributes:", verbose)
    _log(
        "\n".join(f"{k}: {v}" for k, v in assigned_dataset.extra_attributes.items()),
        verbose,
    )
    _log("Stitching masks", verbose)
    masks = get_cell_masks(data_path=f"{project}/{mouse_name}/{chamber}", roi=roi)

    _log("Making cell dataframe", verbose)
    mask_df = issp.pipeline.segment.make_cell_dataframe(
        f"{project}/{mouse_name}/{chamber}",
        roi,
        masks=masks,
        mask_expansion=0,
        atlas_size=None,
    )
    del masks
    gc.collect()
    _log("Loading spots", verbose)
    bc = pd.read_pickle(error_dataset.path_full)
    spots = bc[(bc.chamber == chamber) & (bc.roi == roi)].copy()
    del bc
    gc.collect()
    _log("Assigning barcodes to masks", verbose)
    mask_assignment_id = assign_barcodes_to_masks(
        spots,
        mask_df,
        p=p,
        m=m,
        background_spot_prior=background_spot_prior,
        spot_distribution_sigma=spot_distribution_sigma,
        max_iterations=max_iterations,
        distance_threshold=distance_threshold,
        verbose=verbose,
        base_column=base_column,
    )
    _log("Saving mask assignment", verbose)
    # replace mask id with actual mask label
    mask_assignment = np.array(mask_df.index[mask_assignment_id])
    # -1 will match the last value, put it back to -1
    mask_assignment[mask_assignment_id == -1] = -1
    _log(f"Assigned {mask_assignment_id.size} spots to masks", verbose)
    output = pd.DataFrame(index=spots.index, columns=["mask", "chamber", "roi", "spot"])
    output.loc[spots.index, "spot"] = spots.index.values
    output.loc[spots.index, "mask"] = mask_assignment
    output.loc[spots.index, "chamber"] = chamber
    output.loc[spots.index, "roi"] = roi
    output.to_pickle(assigned_dataset.path_full)
    _log(f"Saved mask assignment to {assigned_dataset.path_full}", verbose)
    assigned_dataset.extra_attributes["ended"] = str(pd.Timestamp.now())
    assigned_dataset.update_flexilims(mode="overwrite")
    _log("Updated dataset in flexilims", verbose)
    return output
