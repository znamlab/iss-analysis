import numpy as np
import pandas as pd
import iss_preprocess as issp
from znamutils import slurm_it


@slurm_it(conda_env="iss_preprocess", print_job_id=True)
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
            save_folder / f"barcode_in_cells_{chamber}_{roi}.pkl"
        )
    return barcode_in_cells
