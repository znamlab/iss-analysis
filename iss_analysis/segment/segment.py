from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imwrite
import iss_preprocess as issp
import flexiznam as flz
from znamutils import slurm_it


def get_cell_masks(data_path, roi, projection="corrected", mask_expansion=0):
    """Small wrapper to get cell masks from a given data path.

    Wrap to ensure we use the same projection for all calls

    Args:
        data_path (str): Path to acquisition data (chamber folder)
        roi (int): Region of interest
        projection (str): Projection to use
        mask_expansion (int): Expansion of the mask

    Returns:
        np.ndarray: Cell masks
    """
    ops = issp.io.load_ops(data_path)
    seg_prefix = f"{ops['segmentation_prefix']}_masks"
    masks = issp.pipeline.stitch_registered(
        data_path,
        prefix=seg_prefix,
        roi=roi,
        projection=projection,
    )
    if mask_expansion > 0:
        masks = issp.pipeline.segment.get_big_masks(
            data_path, roi, masks, mask_expansion
        )
    return masks


@slurm_it(conda_env="iss-preprocess", print_job_id=True)
def get_barcode_in_cells(
    chamber, roi, acq_path, error_corrected_barcodes, save_folder=None
):
    """Count barcodes in cells for a given chamber and roi.

    Args:
        chamber (str): Chamber name.
        roi (int): Region of interest.
        acq_path (str): Path to acquisition data. Usually mouse folder
        error_corrected_barcodes (pd.DataFrame): Error corrected barcodes.

    Returns:
        pd.DataFrame: Barcodes in cells."""

    if isinstance(error_corrected_barcodes, str):
        error_corrected_barcodes = pd.read_pickle(error_corrected_barcodes)

    bigmask = issp.pipeline.stitch_registered(
        acq_path + f"/{chamber}",
        prefix="mCherry_1_masks",
        roi=roi,
        projection="corrected",
    )
    bigmask = issp.pipeline.segment.get_big_masks(
        acq_path + f"/{chamber}", roi, bigmask, mask_expansion=5
    )[..., 0]
    df = error_corrected_barcodes[
        (error_corrected_barcodes["chamber"] == chamber)
        & (error_corrected_barcodes["roi"] == roi)
    ].copy()
    barcode_in_cells = issp.segment.cells.count_spots(
        df, grouping_column="corrected_bases", masks=bigmask
    )
    # add cells with 0 count
    all_cells = np.unique(bigmask)
    missing = set(all_cells) - set(barcode_in_cells.index)
    missing_df = pd.DataFrame(
        index=list(missing),
        columns=barcode_in_cells.columns,
        data=np.zeros((len(missing), len(barcode_in_cells.columns))),
    )
    barcode_in_cells = pd.concat([barcode_in_cells, missing_df])

    if save_folder is not None:
        barcode_in_cells.to_pickle(
            Path(save_folder) / f"barcode_in_cells_{chamber}_{roi}.pkl"
        )
        print(f"Saved to {save_folder}")
    print("Done")
    return barcode_in_cells


@slurm_it(conda_env="iss-preprocess", print_job_id=True, slurm_options={"mem": "128G"})
def save_stitched_for_manual_clicking(
    project, mouse, chamber, roi, error_correction_ds_name, mask_assignment_dataset_name
):
    data_path = f"{project}/{mouse}/{chamber}"
    destination = issp.io.get_processed_path(data_path) / "manual_starter_click"
    destination.mkdir(exist_ok=True, parents=True)
    ops = issp.io.load_ops(data_path)
    # save a genes stack
    print("Stitching genes")
    rab = issp.pipeline.stitch_registered(
        data_path,
        prefix="genes_round_1_1",
        roi=roi,
        channels=ops["ref_ch"],
    )
    imwrite(destination / f"{mouse}_{chamber}_{roi}_genes.tif", rab)
    del rab

    # save a hyb stack
    print("Stitching hyb1")
    rab = issp.pipeline.stitch_registered(
        data_path,
        prefix="hybridisation_round_1_1",
        roi=roi,
        channels=range(4),
    )
    imwrite(destination / f"{mouse}_{chamber}_{roi}_hyb1.tif", rab)
    del rab

    # save a rabies stack
    print("Stitching rabies")
    rab = issp.pipeline.stitch_registered(
        data_path,
        prefix="barcode_round_1_1",
        roi=roi,
        channels=ops["ref_ch"],
    )
    imwrite(destination / f"{mouse}_{chamber}_{roi}_rabies.tif", rab)
    del rab

    # save stitched mCherry stack
    print("Stitching mCherry")
    mCherry = issp.pipeline.stitch_registered(
        data_path,
        prefix=f"mCherry_1",
        roi=roi,
        channels=[2, 3],
    )
    imwrite(destination / f"{mouse}_{chamber}_{roi}_mCherry.tif", mCherry)
    del mCherry

    # and the reference stack
    print("Stitching reference")
    reference = issp.pipeline.stitch_registered(
        data_path,
        prefix=ops["reference_prefix"],
        roi=roi,
        channels=ops["ref_ch"],
    )
    imwrite(destination / f"{mouse}_{chamber}_{roi}_reference.tif", reference)
    del reference

    if False:
        # save mcherry centers as npy
        print("Finding mCherry centers")
        mCherry_masks = issp.pipeline.stitch_registered(
            data_path,
            prefix=f"mCherry_1_masks",
            roi=roi,
            projection="corrected",
        )[..., 0]
        imwrite(
            destination / f"{mouse}_{chamber}_{roi}_mCherry_masks.tif", mCherry_masks
        )
        mCherry_centers = issp.pipeline.segment.make_cell_dataframe(
            data_path, roi, masks=mCherry_masks, mask_expansion=0, atlas_size=None
        )
        pts = mCherry_centers[["x", "y"]].values
        np.save(destination / f"{mouse}_{chamber}_{roi}_mCherry_centers.npy", pts)
        del mCherry_masks, mCherry_centers

    # find masks of rabies cells
    print("Finding cells")
    cell_masks = get_cell_masks(
        data_path,
        roi=roi,
        mask_expansion=5,
    )[..., 0]
    print("Getting mask assignment")
    flm_sess = flz.get_flexilims_session(project_id=project, reuse_token=True)
    rabies_err_corr_ds = flz.Dataset.from_flexilims(
        name=error_correction_ds_name, flexilims_session=flm_sess
    )
    rabies_mask_dss = flz.get_datasets(
        origin_name=f"{mouse}_{chamber}",
        flexilims_session=flm_sess,
        dataset_type="barcodes_mask_assignment",
    )
    # find the one that matches the roi
    rabies_mask_ds = None
    for ds in rabies_mask_dss:
        if ds.extra_attributes["roi"] == roi:
            rabies_mask_ds = ds
            break
    if rabies_mask_ds is None:
        raise ValueError("No dataset found for the given roi")
    err_corr = pd.read_pickle(rabies_err_corr_ds.path_full)
    rabies_assignment = pd.read_pickle(rabies_mask_ds.path_full)
    # add mask to the err_corr dataframe
    err_corr["cell_mask"] = rabies_assignment["mask"]
    # keep only the relevant roi
    rabies_assignment = err_corr[(err_corr.chamber == chamber) & (err_corr.roi == roi)]
    valid_masks = rabies_assignment[
        rabies_assignment["cell_mask"] != -1
    ].cell_mask.unique()
    print("Saving only rabies cells")
    rabies_cells = np.zeros_like(cell_masks)
    for mask in valid_masks:
        rabies_cells[cell_masks == mask] = mask
    imwrite(
        destination / f"{mouse}_{chamber}_{roi}_rabies_cells_mask.tif", rabies_cells
    )
    # save spots that are assigned to a cell, with the cell mask and barcode
    print("Saving spots")
    valid_spots = rabies_assignment[rabies_assignment["cell_mask"] > 0]
    valid_spots = valid_spots[["x", "y", "cell_mask", "corrected_bases"]].copy()
    seq = valid_spots.corrected_bases.unique()
    valid_spots["seq_id"] = np.nan
    for i, s in enumerate(seq):
        valid_spots.loc[valid_spots.corrected_bases == s, "seq_id"] = i

    spot_array = valid_spots[["x", "y", "seq_id", "cell_mask"]].values
    np.save(destination / f"{mouse}_{chamber}_{roi}_rabies_spots.npy", spot_array)
    print("Done")
