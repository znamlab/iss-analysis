from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imwrite
import iss_preprocess as issp
import iss_analysis as issa
import flexiznam as flz
from znamutils import slurm_it

from iss_preprocess.pipeline.segment import get_cell_masks


def get_barcode_in_cells(
    project,
    mouse,
    error_correction_ds_name,
    valid_chambers=None,
    save_folder=None,
    verbose=True,
    add_ara_properties=True,
    redo=False,
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
        add_ara_properties (bool, optional): Add ARA properties to the dataframe.
            Defaults to True.
        redo (bool, optional): Redo the ARA assignment. Defaults to False.

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
    # add barcode_id for plotting
    barcodes = list(rab_spot_df.corrected_bases.unique())
    rab_spot_df["barcode_id"] = rab_spot_df.corrected_bases.map(
        lambda x: barcodes.index(x)
    )
    # also add "slice" column to avoid indexing by chmaber and roi everytime
    rab_spot_df["slice"] = (
        rab_spot_df.chamber + "_" + rab_spot_df.roi.map(lambda x: f"{x:02d}")
    )

    if verbose:
        print(f"Loaded {len(rab_spot_df)} rabies barcodes")
    if add_ara_properties:
        ara_cols = ["ara_x", "ara_y", "ara_z", "area_id", "area_acronym"]
        for w in ara_cols:
            rab_spot_df[w] = "NaN" if w == "area_acronym" else np.nan
        ara_info_folder = (
            issp.io.get_processed_path(f"{project}/{mouse}") / "analysis" / "ara_infos"
        )
    # for all chambers, get the mask assignment datasets

    if valid_chambers is None:
        chambers = issa.io.get_chamber_datapath(f"{project}/{mouse}")
    else:
        if isinstance(valid_chambers, str):
            valid_chambers = [valid_chambers]
        chambers = [f"{project}/{mouse}/{ch}" for ch in valid_chambers]

    if verbose:
        print(f"Getting mask assignments for {len(chambers)} chambers")
    rab_spot_df["cell_mask"] = np.zeros_like(rab_spot_df.x.values) * np.nan
    for data_path in chambers:
        chamber = issp.io.get_processed_path(data_path).stem
        if verbose:
            print(f"    {chamber}")
        try:
            rabies_mask_dss = flz.get_datasets(
                origin_name=f"{mouse}_{chamber}",
                flexilims_session=flm_sess,
                dataset_type="barcodes_mask_assignment",
            )
        except flz.FlexilimsError:
            print(f"No mask assignment found for {chamber}")
            continue
        roi_dim = issp.io.get_roi_dimensions(data_path)
        for roi in roi_dim[:, 0]:
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

            if add_ara_properties:
                target = (
                    ara_info_folder
                    / f"{error_correction_ds_name}_{chamber}_{roi}_rabies_spots_ara_info.pkl"
                )
                if target.exists() and (not redo):
                    ara_info = pd.read_pickle(target)
                else:
                    print("ARA info not found, recreate them")
                    ara_info = issa.barcodes.main.save_ara_info(
                        project,
                        mouse,
                        chamber,
                        roi,
                        error_correction_ds_name=error_correction_ds_name,
                        acronyms=True,
                        full_scale=True,
                        verbose=verbose,
                        use_slurm=False,
                    )
                assert np.all(
                    np.isnan(rab_spot_df.loc[ara_info.index, "ara_x"])
                ), "Conflict in spot ids"
                for col in ara_cols:
                    rab_spot_df.loc[ara_info.index, col] = ara_info[col].values

    assigned = rab_spot_df.cell_mask > 0
    rab_spot_df.loc[assigned, "mask_uid"] = [
        f"{c}_{r}_{int(m)}"
        for c, r, m in rab_spot_df.loc[assigned, ["chamber", "roi", "cell_mask"]].values
    ]

    # make a dataframe for rabies positive cells
    # get rid of background spots
    assigned_rab = rab_spot_df[assigned].copy()
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
    bc = rab_cells_barcodes.columns
    is_present = rab_cells_barcodes.astype(bool)
    rab_cells_properties["all_barcodes"] = [
        list(bc[present]) for present in is_present.values
    ]
    if add_ara_properties:
        ara_coords = (
            assigned_rab[["mask_uid"] + ara_cols[:3]]
            .groupby("mask_uid")
            .aggregate("median")
        )
        rab_cells_properties = rab_cells_properties.join(ara_coords)
        ara_area = (
            assigned_rab[["mask_uid"] + ara_cols[3:]]
            .groupby("mask_uid")
            .aggregate(pd.Series.mode)
        )
        rab_cells_properties = rab_cells_properties.join(ara_area)

    if verbose:
        print(
            f"Data frame with {len(rab_cells_barcodes)} rabies cells and"
            + f" {len(rab_cells_barcodes.columns)} unique barcodes"
        )

    if save_folder is not None:
        rab_spot_df.to_pickle(
            Path(save_folder) / f"barcode_in_cells_{chamber}_{roi}.pkl"
        )
        print(f"Saved to {save_folder}")
    if verbose:
        print("Done")
    return rab_spot_df, rab_cells_barcodes, rab_cells_properties


def match_starter_to_barcodes(
    project,
    mouse,
    rabies_cell_properties,
    rab_spot_df,
    mcherry_cells=None,
    verbose=True,
    max_starter_distance=10,
    min_spot_number=5,
):
    """Match starter cells to rabies cells.

    Args:
        project (str): The project name.
        mouse (str): The mouse name.
        rabies_cell_properties (pd.DataFrame): The rabies cell properties.
        rab_spot_df (pd.DataFrame): The rabies spots.
        starters (pd.DataFrame, optional): The starter cells. Defaults to None.
        verbose (bool, optional): Print progress. Defaults to True.
        max_starter_distance (int, optional): The radius around starter cell to look for
            spots, in um. Defaults to 10.
        min_spot_number (int, optional): The minimum number of spots in this radius
            required to consider a cell to be starter cell. Defaults to 5.

    Returns:
        pd.DataFrame: The rabies cell properties with starter IDs.
        pd.DataFrame: The mcherry cell properties with starter IDs.
    """
    manual_click = (
        issp.io.get_processed_path(f"{project}/{mouse}") / "analysis" / "mcherry_cells"
    )
    if mcherry_cells is None:
        mcherry_cells = issa.io.get_mcherry_cells(project, mouse, verbose=verbose)

    mcherry_cell_properties = []
    for (ch, roi), mcherry_df in mcherry_cells.groupby(["chamber", "roi"]):
        px_size = issp.io.get_pixel_size(f"{project}/{mouse}/{ch}")
        max_distance = max_starter_distance / px_size

        if verbose:
            print(f"Finding starter cell for {ch} {roi}")

        spots_roi = rab_spot_df[
            (rab_spot_df.cell_mask > 0)
            & (rab_spot_df.roi == roi)
            & (rab_spot_df.chamber == ch)
        ].copy()
        rab_cell_roi = rabies_cell_properties.query(
            f"roi == {roi} and chamber == '{ch}'"
        ).copy()

        for _, cell_info in mcherry_df.iterrows():
            mcherry_cell_properties.append(
                is_this_cell_a_starter(
                    cell_info=cell_info,
                    spots_roi=spots_roi,
                    rab_cell_roi=rab_cell_roi,
                    max_distance=max_distance,
                    min_spot_number=min_spot_number,
                    verbose=verbose,
                )
            )
    mcherry_cell_properties = pd.DataFrame(mcherry_cell_properties)

    # Now add starter info to rabies_cell_properties
    rabies_cell_properties["mcherry_uid"] = None
    rabies_cell_properties["distance2mcherry"] = None
    rabies_cell_properties["is_starter"] = False
    starters = mcherry_cell_properties.query("mask_uid != 'NaN'")
    rabies_cell_properties.loc[starters.mask_uid, "is_starter"] = True
    # add the relevant columns of rabies_cell_properties to mcherry_cell_properties
    col2populate = dict(
        mask_uid="mcherry_uid", distance_to_mask_centroid="distance2mcherry"
    )
    for col, target in col2populate.items():
        v = starters.groupby("mask_uid")[col].apply(list)
        rabies_cell_properties.loc[v.index, target] = v.values
    n_starter_per_mask = v.map(len)
    for mask, n in n_starter_per_mask[n_starter_per_mask > 1].items():
        print(f"Multiple starters ({n}) for mask {mask}.")
        mcherry_cell_properties.loc[
            mcherry_cell_properties.mask_uid == mask, "error"
        ] = f"{n} starters"

    return rabies_cell_properties, mcherry_cell_properties


def is_this_cell_a_starter(
    cell_info: pd.DataFrame,
    spots_roi: pd.DataFrame,
    rab_cell_roi: pd.DataFrame,
    max_distance: float = 100,
    min_spot_number: int = 5,
    verbose: bool = True,
):
    """Check if a cell is a starter cell.

    Args:
        cell_info (pd.Series): The cell information.
        spots_roi (pd.DataFrame): The spots in the region of interest.
        rab_cell_roi (pd.DataFrame): The rabies cells in the region of interest.
        max_distance (int, optional): The maximum distance to consider a cell as a
            starter cell in pixels. Defaults to 100 px.
        min_spot_number (int, optional): The minimum number of spots to consider a
            cell as a starter cell. Defaults to 5.
        verbose (bool, optional): Print info about errors. Defaults to True.

    Returns:
        bool: True if the cell is a starter cell.
    """
    # Initialize the output
    vals = dict(
        is_starter=False,
        n_spots_close=np.nan,
        n_spots_assigned=np.nan,
        distance_to_spot_centroid=np.nan,
        distance_to_mask_centroid=np.nan,
        main_barcode="NaN",
        mask_uid="NaN",
        error="None",
        all_barcodes=[],
        n_spot_per_barcode=[],
    )
    out = pd.concat([cell_info, pd.Series(data=vals)])

    # get spots that are close enough from the cell
    dist = np.sqrt((spots_roi.x - cell_info.x) ** 2 + (spots_roi.y - cell_info.y) ** 2)
    spots = spots_roi[dist < max_distance]
    out["n_spots_close"] = len(spots)
    out["n_spots_assigned"] = len(spots[spots.cell_mask > 0])
    if out["n_spots_assigned"] < min_spot_number:
        return out

    # get the centroids of all barcodes that have at least 5 spots
    valid_spots = spots.groupby("corrected_bases").filter(lambda x: len(x) > 4)
    if len(valid_spots) == 0:
        return out
    out["all_barcodes"] = list(valid_spots.corrected_bases.unique())
    out["n_spot_per_barcode"] = valid_spots.corrected_bases.value_counts().to_dict()
    # get the centroid of each barcode
    centroids = (
        valid_spots[["corrected_bases", "x", "y"]]
        .groupby("corrected_bases")
        .aggregate("mean")
    )
    # get the distance of each barcode to the cell
    centroids["dist"] = np.sqrt(
        (centroids.x - cell_info.x) ** 2 + (centroids.y - cell_info.y) ** 2
    )
    # get the closest barcode
    closest = centroids.loc[centroids.dist.idxmin()]
    out["main_barcode"] = closest.name
    spots = valid_spots[valid_spots.corrected_bases == closest.name]
    spots_per_mask = spots.mask_uid.value_counts()
    if spots_per_mask.max() < min_spot_number:
        if verbose:
            print(
                f"Assignment issue for mcherry cell {cell_info.name} (index "
                + f"{cell_info.original_index}). {spots_per_mask.max()} "
                + f"spots in main barcode ({closest.name} assigned to "
                + f"{spots_per_mask.idxmax()})"
            )
        out["error"] = "Assignment issue. Not enough spots in the closest barcode"
    out["mask_uid"] = spots_per_mask.idxmax()
    out["distance_to_spot_centroid"] = closest.dist
    mask_centroid = rab_cell_roi.loc[out["mask_uid"], ["x", "y"]]
    out["distance_to_mask_centroid"] = np.sqrt(
        (mask_centroid.x - cell_info.x) ** 2 + (mask_centroid.y - cell_info.y) ** 2
    )
    out["is_starter"] = True
    return out


@slurm_it(conda_env="iss-preprocess", print_job_id=True, slurm_options={"mem": "128G"})
def save_stitched_for_manual_clicking(
    project,
    mouse,
    chamber,
    roi,
    error_correction_ds_name,
    redo=False,
    save_imgs=True,
    save_mcherry_masks=True,
    save_rabies_masks=True,
    save_spots=True,
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
            issp.io.write_stack(stack=img, fname=fname, bigtiff=True)
            del img

    # get spots
    if save_spots:
        print("Finding spots")
        spots_to_save = dict(
            hyb="hybridisation_round_1_1_spots", genes="genes_round_spots"
        )
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

    if save_mcherry_masks:
        # save mcherry centers as npy
        print("Finding mCherry centers")
        fname = destination / f"{mouse}_{chamber}_{roi}_mCherry_masks.tif"
        if fname.exists() and not redo:
            print(f"File {fname} already exists, skipping")
        else:
            mCherry_masks = get_cell_masks(
                data_path,
                roi=roi,
                prefix=f"mCherry_1",
                projection="corrected",
            )
            issp.io.write_stack(
                stack=mCherry_masks,
                fname=destination / f"{mouse}_{chamber}_{roi}_mCherry_masks.tif",
                bigtiff=True,
            )
            mCherry_centers = issp.pipeline.segment.make_cell_dataframe(
                data_path,
                roi,
                masks=mCherry_masks,
                mask_expansion=None,
                atlas_size=None,
            )
            pts = mCherry_centers[["x", "y"]].values
            np.save(destination / f"{mouse}_{chamber}_{roi}_mCherry_centers.npy", pts)
            del mCherry_masks, mCherry_centers

    if save_rabies_masks:
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
        rabies_assignment = err_corr[
            (err_corr.chamber == chamber) & (err_corr.roi == roi)
        ]
        fname = destination / f"{mouse}_{chamber}_{roi}_rabies_cells_masks.tif"
        if fname.exists() and not redo:
            print(f"File {fname} already exists, skipping")
        else:
            cell_masks = get_cell_masks(data_path, roi=roi)
            issp.io.write_stack(
                stack=cell_masks,
                fname=fname.with_name(f"{mouse}_{chamber}_{roi}_all_cells_mask.tif"),
                bigtiff=True,
            )

            valid_masks = rabies_assignment[
                rabies_assignment["cell_mask"] != -1
            ].cell_mask.unique()
            print("Saving only rabies cells")
            rabies_cells = np.zeros_like(cell_masks)
            for mask in valid_masks:
                rabies_cells[cell_masks == mask] = mask
            issp.io.write_stack(stack=rabies_cells, fname=fname, bigtiff=True)

    if save_spots and save_rabies_masks:
        # save spots that are assigned to a cell, with the cell mask and barcode
        fname = destination / f"{mouse}_{chamber}_{roi}_rabies_spots.npy"
        if fname.exists() and not redo:
            print(f"File {fname} already exists, skipping")
        else:
            print("Saving spots")
            rabies_assignment["seq_id"] = np.nan
            seq = rabies_assignment.corrected_bases.unique()
            for i, s in enumerate(seq):
                rabies_assignment.loc[
                    rabies_assignment.corrected_bases == s, "seq_id"
                ] = i

            valid_spots = rabies_assignment[rabies_assignment["cell_mask"] > 0]
            valid_spots = valid_spots[
                ["x", "y", "cell_mask", "seq_id", "corrected_bases"]
            ].copy()

            spot_array = valid_spots[
                ["x", "y", "seq_id", "cell_mask", "corrected_bases"]
            ].values
            np.save(fname, spot_array)
            # also save non assigned spots
            fname = destination / f"{mouse}_{chamber}_{roi}_rabies_spots_unassigned.npy"
            valid_spots = rabies_assignment[rabies_assignment["cell_mask"] <= 0]
            valid_spots = valid_spots[
                ["x", "y", "cell_mask", "seq_id", "corrected_bases"]
            ].copy()
            spot_array = valid_spots[
                ["x", "y", "seq_id", "cell_mask", "corrected_bases"]
            ].values
            np.save(fname, spot_array)

    print("Done")
