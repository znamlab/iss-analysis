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
