from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imwrite
import iss_preprocess as issp
import iss_analysis as issa
import flexiznam as flz
from znamutils import slurm_it

from iss_preprocess.segment.cells import get_cell_masks


def get_barcode_in_cells(
    project,
    mouse,
    error_correction_ds_name,
    valid_chambers=None,
    save_folder=None,
    verbose=True,
):
    """Count barcodes in cells for a given chamber and roi.

    rab_spot_df will be read from error_correction_ds directly

    For each chamber/roi the one barcodes_mask_assignment dataset will be found on
    flexilims and the assignment read from there.


    Args:
        project (str): The project name.
        mouse (str): The mouse name.
        error_correction_ds_name (str): The error correction dataset name.
        valid_chambers (list, optional): The list of chambers to include. Defaults to
            None.
        save_folder (str, optional): The folder where to save the results. Defaults to
            None.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        pd.DataFrame: The rabies spots dataframe.
        pd.DataFrame: The rabies cells barcodes, Ncells x Nbarcodes.
        pd.DataFrame: The rabies cells properties, Ncells x Nproperties.
    """

    # load the error corrected barcodes
    flm_sess = flz.get_flexilims_session(project_id=project)
    err_corr_ds = flz.Dataset.from_flexilims(
        name=error_correction_ds_name, flexilims_session=flm_sess
    )
    rab_spot_df = pd.read_pickle(err_corr_ds.path_full)
    if verbose:
        print(f"Loaded {len(rab_spot_df)} rabies barcodes")

    # for all chambers, get the mask assignment datasets
    chambers = issa.io.get_chamber_datapath(f"{project}/{mouse}")
    if verbose:
        print(f"Getting mask assignments for {len(chambers)} chambers")
    rab_spot_df["cell_mask"] = np.zeros_like(rab_spot_df.x.values) * np.nan
    for data_path in chambers:
        chamber = issp.io.get_processed_path(data_path).stem
        if (valid_chambers is not None) and (chamber not in valid_chambers):
            continue
        rabies_mask_dss = flz.get_datasets(
            origin_name=f"{mouse}_{chamber}",
            flexilims_session=flm_sess,
            dataset_type="barcodes_mask_assignment",
        )

        for roi in range(1, 11):
            # find the one that matches the roi
            rabies_mask_ds = None
            for ds in rabies_mask_dss:
                if ds.extra_attributes["roi"] == roi:
                    rabies_mask_ds = ds
                    break
            if rabies_mask_ds is None:
                raise ValueError("No dataset found for the given roi")
            rab_ass = pd.read_pickle(rabies_mask_ds.path_full)
            assert np.isnan(rab_spot_df.loc[rab_ass.index, "cell_mask"]).all()
            rab_spot_df.loc[rab_ass.index, "cell_mask"] = rab_ass["mask"].astype(float)

    # make a dataframe for rabies positive cells
    # get rid of background spots
    assigned_rab = rab_spot_df[rab_spot_df.cell_mask > 0].copy()
    # make unique identifier in case
    assigned_rab["mask_uid"] = [
        f"{r.chamber}_{r.roi}_{int(r.cell_mask)}" for _, r in assigned_rab.iterrows()
    ]
    if verbose:
        print(f"Counting spots")
    rab_cells_barcodes = issp.segment.count_spots(
        assigned_rab,
        grouping_column="corrected_bases",
        masks=None,
        mask_id_column="mask_uid",
    )
    # create a separate dataframe for properties of these cells that are not barcode
    rab_cells_properties = (
        assigned_rab[["mask_uid", "x", "y"]].groupby("mask_uid").aggregate("median")
    )
    rab_cells_properties["cell_id"] = rab_cells_properties.index.map(
        lambda x: int(x.split("_")[-1])
    )
    rab_cells_properties["roi"] = rab_cells_properties.index.map(
        lambda x: int(x.split("_")[-2])
    )
    rab_cells_properties["chamber"] = rab_cells_properties.index.map(
        lambda x: "_".join(x.split("_")[:2])
    )
    rab_cells_properties["max_n_spots"] = rab_cells_barcodes.max(axis=1)
    rab_cells_properties["main_barcode"] = rab_cells_barcodes.idxmax(axis=1)
    rab_cells_properties["n_unique_barcodes"] = rab_cells_barcodes.astype(bool).sum(
        axis=1
    )
    if verbose:
        print(
            f"Data frame with {len(rab_cells_barcodes)} rabies cells and {len(rab_cells_barcodes.columns)} unique barcodes"
        )

    if save_folder is not None:
        rab_spot_df.to_pickle(
            Path(save_folder) / f"barcode_in_cells_{chamber}_{roi}.pkl"
        )
        print(f"Saved to {save_folder}")
    print("Done")
    return rab_spot_df, rab_cells_barcodes, rab_cells_properties


def match_starter_to_barcodes(
    project,
    mouse,
    rabies_cell_properties,
    rab_spot_df,
    starters=None,
    redo=False,
    verbose=True,
):
    """Match starter cells to rabies cells.

    Args:
        project (str): The project name.
        mouse (str): The mouse name.
        rabies_cell_properties (pd.DataFrame): The rabies cell properties.
        rab_spot_df (pd.DataFrame): The rabies spots.
        starters (pd.DataFrame, optional): The starter cells. Defaults to None.
        redo (bool, optional): Redo the matching. Defaults to False.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        pd.DataFrame: The rabies cell properties with starter IDs.
    """
    manual_click = (
        issp.io.get_processed_path(f"{project}/{mouse}") / "analysis" / "starter_cells"
    )
    if starters is None:
        starters = issa.io.get_starter_cells(project, mouse)

    rabies_cell_properties["starter"] = False
    rabies_cell_properties["starter_id"] = "none"
    rabies_cell_properties["distance"] = np.nan
    for (ch, roi), starter_df in starters.groupby(["chamber", "roi"]):
        if ch not in ["chamber_07", "chamber_08"]:
            continue
        fname = manual_click / f"rabies_cells_{mouse}_{ch}_roi_{roi}.pkl"
        if (not redo) and fname.exists():
            if verbose:
                print(f"Loading {fname}")
            rab_this_roi = pd.read_pickle(fname)
            rabies_cell_properties.loc[rab_this_roi.index, "starter"] = rab_this_roi[
                "starter"
            ]
            rabies_cell_properties.loc[rab_this_roi.index, "starter_id"] = rab_this_roi[
                "starter_id"
            ]
            continue
        if verbose:
            print(f"Determining starter cell for {ch} {roi}")
            # get the masks
            print("Loading masks")
        cell_masks = issp.segment.get_cell_masks(f"{project}/{mouse}/{ch}", roi)
        spot_roi = rab_spot_df[
            (rab_spot_df.cell_mask > 0)
            & (rab_spot_df.roi == roi)
            & (rab_spot_df.chamber == ch)
        ].copy()
        if verbose:
            print("Finding starter cell")
        for st_id, st in starter_df.iterrows():
            mask_id = cell_masks[int(st.y), int(st.x)]
            mask_uid = f"{ch}_{roi}_{int(mask_id)}"
            if (mask_id == 0) or (mask_uid not in rabies_cell_properties.index):
                if verbose:
                    print(f"Starter cell {st_id} not found in mask")
                spot_roi["dist"] = np.sqrt(
                    (spot_roi.x - st.x) ** 2 + (spot_roi.y - st.y) ** 2
                )
                mask_id = spot_roi.loc[spot_roi.dist.idxmin(), "cell_mask"]
                mask_uid = f"{ch}_{roi}_{int(mask_id)}"
                while rabies_cell_properties.loc[mask_uid, "starter"]:
                    if verbose:
                        print(
                            f"Mask {mask_uid} already assigned to starter cell ... looking for next closest cell"
                        )
                    spot_roi = spot_roi[spot_roi.cell_mask != mask_id]
                    mask_id = spot_roi.loc[spot_roi.dist.idxmin(), "cell_mask"]
                    mask_uid = f"{ch}_{roi}_{int(mask_id)}"
                distance = spot_roi.loc[spot_roi.dist.idxmin(), "dist"]
                if verbose:
                    print(f"Closest cell found at {distance:.2f} pixels")
                if distance > 100:
                    print(f"XXXXX FAR XXXXXX")
            else:
                if verbose:
                    print(f"Starter cell {st_id} found in mask")
                distance = 0
            assert (
                mask_uid in rabies_cell_properties.index
            ), f"Mask {mask_uid} not found in rabies centroids"
            if rabies_cell_properties.loc[mask_uid, "starter"]:
                raise ValueError(
                    f"Cannot find {st_id}. Multiple starter cells found for {mask_uid}"
                )
            rabies_cell_properties.loc[mask_uid, "starter"] = True
            rabies_cell_properties.loc[
                mask_uid, "starter_id"
            ] = f"{st.chamber}_{st.roi}_{st_id}"
            rabies_cell_properties.loc[mask_uid, "distance"] = distance
        # save for this roi
        rab_this_roi = rabies_cell_properties[
            (rabies_cell_properties.roi == roi) & (rabies_cell_properties.chamber == ch)
        ].copy()
        rab_this_roi.to_pickle(fname)
    return rabies_cell_properties


@slurm_it(conda_env="iss-preprocess", print_job_id=True, slurm_options={"mem": "128G"})
def save_stitched_for_manual_clicking(
    project,
    mouse,
    chamber,
    roi,
    error_correction_ds_name,
    redo=False,
    save_imgs=True,
):
    """Save stitched images for manual clicking.

    Useful to save a bunch of stitched data that can be loaded in napari for manual
    clicking.

    Args:
        project (str): The project name.
        mouse (str): The mouse name.
        chamber (str): The chamber name.
        roi (int): The region of interest.
        error_correction_ds_name (str): The error correction dataset name.
        redo (bool, optional): Redo the stitching. Defaults to False.
        save_imgs (bool, optional): Save images. Defaults to True.

    Returns:
        None
    """
    data_path = f"{project}/{mouse}/{chamber}"
    destination = issp.io.get_processed_path(data_path) / "manual_starter_click"
    destination.mkdir(exist_ok=True, parents=True)
    ops = issp.io.load_ops(data_path)

    if save_imgs:
        stuff_to_save = dict(
            genes="genes_round_1_1",
            hyb="hybridisation_round_1_1",
            rab="barcode_round_1_1",
            reference=ops["reference_prefix"],
            mCherry="mCherry_1",
        )
        channels = dict(
            genes=range(4),
            hyb=range(4),
            rab=range(4),
            mCherry=[2, 3],
            reference=ops["ref_ch"],
        )
    else:
        stuff_to_save = {}
        channels = {}

    # iterating on images to save
    for k, v in stuff_to_save.items():
        fname = destination / f"{mouse}_{chamber}_{roi}_{k}.tif"
        if fname.exists() and not redo:
            print(f"File {fname} already exists, skipping")
        else:
            print(f"Stitching {k}")
            img = issp.pipeline.stitch_registered(
                data_path,
                prefix=v,
                roi=roi,
                channels=channels[k],
            )
            imwrite(fname, img)
            del img

    # get spots
    print("Finding spots")
    spots_to_save = dict(hyb="hybridisation_round_1_1_spots", genes="genes_round_spots")
    for k, v in spots_to_save.items():
        fname = destination / f"{mouse}_{chamber}_{roi}_{k}_spots.pkl"
        if fname.exists() and not redo:
            print(f"File {fname} already exists, skipping")
        else:
            sc = "spot_score" if k == "genes" else "score"
            print(f"Getting {k} spots")
            spot_file = issp.io.get_processed_path(data_path) / f"{v}_{roi}.pkl"
            data = pd.read_pickle(spot_file)
            pts = data[["x", "y", "gene", sc]]
            pts.columns = ["x", "y", "gene", "score"]
            pts.to_pickle(fname)
            print(f"Saved {fname}")

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
    fname = destination / f"{mouse}_{chamber}_{roi}_rabies_cells_mask.tif"
    if fname.exists() and not redo:
        print(f"File {fname} already exists, skipping")
    else:
        cell_masks = get_cell_masks(data_path, roi=roi)
        imwrite(
            fname.with_name(f"{mouse}_{chamber}_{roi}_all_cells_mask.tif"), cell_masks
        )
        valid_masks = rabies_assignment[
            rabies_assignment["cell_mask"] != -1
        ].cell_mask.unique()
        print("Saving only rabies cells")
        rabies_cells = np.zeros_like(cell_masks)
        for mask in valid_masks:
            rabies_cells[cell_masks == mask] = mask
        imwrite(fname, rabies_cells)

    # save spots that are assigned to a cell, with the cell mask and barcode
    fname = destination / f"{mouse}_{chamber}_{roi}_rabies_spots.npy"
    if fname.exists() and not redo:
        print(f"File {fname} already exists, skipping")
    else:
        print("Saving spots")
        valid_spots = rabies_assignment[rabies_assignment["cell_mask"] > 0]
        valid_spots = valid_spots[["x", "y", "cell_mask", "corrected_bases"]].copy()
        seq = valid_spots.corrected_bases.unique()
        valid_spots["seq_id"] = np.nan
        for i, s in enumerate(seq):
            valid_spots.loc[valid_spots.corrected_bases == s, "seq_id"] = i
        spot_array = valid_spots[
            ["x", "y", "seq_id", "cell_mask", "corrected_bases"]
        ].values
        np.save(fname, spot_array)
    print("Done")
