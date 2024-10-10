import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from image_tools.registration.phase_correlation import phase_correlation
from znamutils import slurm_it
from scipy.interpolate import RBFInterpolator
from iss_preprocess.segment.spots import make_spot_image
from iss_preprocess.io import get_processed_path

from ..io import get_sections_info, get_genes_spots
from ..segment import get_barcode_in_cells
from . import ara_registration
from . import utils
from ..vis import diagnostics


def register_all_serial_sections(
    project: str,
    mouse: str,
    error_correction_ds_name: str,
    window_size: int = 500,
    min_spots: int = 10,
    max_barcode_number: int = 50,
    gaussian_width: float = 30,
    n_workers: int = 1,
    verbose=True,
    use_slurm=False,
    reload=True,
    window=(-1, 2),
):
    """Register all serial sections using phase correlation

    Args:
        project (str): Project name
        mouse (str): Mouse name
        error_correction_ds_name (str): Dataset name for error correction
        window_size (int, optional): Window size in um. Defaults to 500.
        min_spots (int, optional): Minimum number of spots found on each slice to
            consider a barcode. Defaults to 10.
        max_barcode_number (int, optional): Maximum number of barcodes to consider.
            The one with the most spots will be kept. Defaults to 50.
        gaussian_width (float, optional): Width of the gaussian kernel for spot images.
            Defaults to 30.
        n_workers (int, optional): Number of workers for parallel processing.
            Defaults to 1.
        verbose (bool, optional): Print progress. Defaults to True.
        use_slurm (bool, optional): Use slurm for parallel processing. Defaults to
            False.
        reload (bool, optional): Reload registration results data. Defaults to True.
        window (tuple, optional): Window around the reference slice to consider.

    Returns:
        dict: Results dataframe with "shift_y", "shift_z", "maxcorr", and "n_barcodes"
            columns for previous and next sections
    """
    section_infos = get_sections_info(project, mouse)
    output = {}
    analysis_folder = get_processed_path(f"{project}") / mouse / "analysis"
    save_folder = analysis_folder / "serial_section_registration"
    save_folder.mkdir(exist_ok=True)
    if use_slurm:
        slurm_folder = Path.home() / "slurm_logs" / "serial_section_registration"
        slurm_folder.mkdir(exist_ok=True)
    else:
        slurm_folder = None
    redo = True
    if verbose and reload:
        print(f"Trying to reload previous registration results")
    for section, sec_info in section_infos.iterrows():
        if reload:
            redo = False
            ref_slice = f"{sec_info['chamber']}_{sec_info['roi']:02d}"
            surrounding_rois = utils.get_surrounding_slices(
                project=project,
                mouse=mouse,
                ref_chamber=sec_info["chamber"],
                ref_roi=sec_info["roi"],
                include_ref=False,
                window=window,
            )
            res = {}
            for sec in surrounding_rois.absolute_section:
                shift = sec - sec_info.absolute_section
                if shift < 0:
                    name = "previous"
                elif shift == 0:
                    raise ValueError("Should not have the reference slice")
                elif shift == 1:
                    name = "next"
                else:
                    assert shift == 2, f"Unexpected shift {shift}"
                    name = f"n_{shift}"
                fname = save_folder / f"{ref_slice}_{name}_registration.csv"
                if fname.exists():
                    res[name] = pd.read_csv(fname, index_col=0)
                else:
                    redo = True
        if redo:
            if verbose:
                print(f"Registering section {section}")
            res = register_single_section(
                project=project,
                mouse=mouse,
                ref_chamber=sec_info["chamber"],
                ref_roi=sec_info["roi"],
                use_rabies=True,
                error_correction_ds_name=error_correction_ds_name,
                window_size=window_size,
                min_spots=min_spots,
                max_barcode_number=max_barcode_number,
                gaussian_width=gaussian_width,
                n_workers=n_workers,
                verbose=verbose,
                save_folder=save_folder,
                use_slurm=use_slurm,
                scripts_name=f"register_serial_sections_{section}",
                slurm_folder=slurm_folder,
            )
        output[section] = res
    return output


@slurm_it(conda_env="iss-preprocess", slurm_options={"mem": "64GB"})
def register_single_section(
    project: str,
    mouse: str,
    ref_chamber: str,
    ref_roi: int,
    use_rabies: bool = True,
    error_correction_ds_name: str = None,
    window_size: int = 500,
    min_spots: int = 10,
    max_barcode_number: int = 50,
    gaussian_width: float = 30,
    n_workers: int = 1,
    verbose=True,
    save_folder=None,
):
    """Register serial sections using phase correlation

    This function will register serial sections to a reference section using phase
    correlation around each rabies cell. It will find the shift between the reference
    section and the previous and next sections for each cell.

    Args:
        project (str): Project name
        mouse (str): Mouse name
        ref_chamber (str): Reference chamber name
        ref_roi (int): Reference ROI number
        use_rabies (bool, optional): Use rabies data if True, genes data otherwise.
            Defaults to True.
        error_correction_ds_name (str): Dataset name for error correction. Must be
            provided if use_rabies is True.
        window_size (int, optional): Window size in um. Defaults to 500.
        min_spots (int, optional): Minimum number of spots found on each slice to
            consider a barcode. Defaults to 10.
        max_barcode_number (int, optional): Maximum number of barcodes to consider.
            The one with the most spots will be kept. Defaults to 50.
        gaussian_width (float, optional): Width of the gaussian kernel for spot images.
            Defaults to 30.
        n_workers (int, optional): Number of workers for parallel processing.
            Defaults to 1.
        verbose (bool, optional): Print progress. Defaults to True.
        save_folder (str, optional): Folder to save results. Defaults to None.

    Returns:
        dict: Results dataframe with "shift_y", "shift_z", "maxcorr", and "n_barcodes"
            columns for "previous" and "next" sections (if they exist)

    """
    # reload the spot data, with the ara coordinates

    if error_correction_ds_name is None:
        raise ValueError("error_correction_ds_name must be provided")
    (
        rab_spot_df,
        _,
        rabies_cell_properties,
    ) = get_barcode_in_cells(
        project,
        mouse,
        error_correction_ds_name,
        valid_chambers=None,
        save_folder=None,
        verbose=verbose,
        add_ara_properties=True,
    )
    if not use_rabies:
        rab_spot_df = get_genes_spots(project, mouse, add_ara_info=True)
        rab_spot_df["corrected_bases"] = "GENES"

    # add rotated ara coordinates
    transform = ara_registration.get_ara_to_slice_rotation_matrix(spot_df=rab_spot_df)
    rabies_cell_properties = ara_registration.rotate_ara_coordinate_to_slice(
        rabies_cell_properties, transform=transform
    )
    # find cells in the ref slice, we will iterate on them
    cells_in_ref = rabies_cell_properties.query(
        f"chamber == '{ref_chamber}' and roi == {ref_roi}"
    )

    surrounding_rois = utils.get_surrounding_slices(
        ref_chamber, ref_roi, project, mouse, include_ref=True, window=(-1, 2)
    )
    # to avoid to always have to groupby chamber and roi, make "slice"
    surrounding_rois["slice"] = (
        surrounding_rois.chamber
        + "_"
        + surrounding_rois.roi.map(lambda x: f"{int(x):02d}")
    )

    # now do previous and next slice, if they exist
    ref_slice = f"{ref_chamber}_{ref_roi:02d}"
    ref_slice_df = surrounding_rois.query("slice == @ref_slice").iloc[0]
    res_befaft = dict()
    for islice, slice_df in surrounding_rois.iterrows():
        if slice_df.slice == ref_slice:
            # not register to self
            continue
        if slice_df.absolute_section < ref_slice_df.absolute_section:
            name = "previous"
        elif slice_df.absolute_section - ref_slice_df.absolute_section == 1:
            name = "next"
        else:
            name = f"n_{slice_df.absolute_section - ref_slice_df.absolute_section}"
        if verbose:
            print(f"Registering {name} slice: {slice_df.slice}")
        reg_one_cell = partial(
            register_local_spots,
            spot_df=rab_spot_df,
            ref_slice=ref_slice,
            target_slice=slice_df.slice,
            window_size=window_size,
            min_spots=min_spots,
            max_barcode_number=max_barcode_number,
            gaussian_width=gaussian_width,
            verbose=False,
            debug=False,
        )
        cell_coords = cells_in_ref[["ara_y_rot", "ara_z_rot"]].values
        bad_cells = np.isnan(cell_coords).any(axis=1)
        cells_in_ref = cells_in_ref[~bad_cells].copy()
        if bad_cells.any():
            print(f"Found {bad_cells.sum()} cells with NaN coordinates. Skipping them")
            cell_coords = cell_coords[~bad_cells]
        if n_workers == 1:
            reg_out = list(tqdm(map(reg_one_cell, cell_coords), total=len(cell_coords)))
        else:
            if verbose:
                print(f"Registering {len(cell_coords)} cells using {n_workers} workers")
            with Pool(n_workers) as pool:
                print("Starting registration")
                reg_out = list(
                    tqdm(
                        pool.imap(reg_one_cell, cell_coords),
                        total=len(cell_coords),
                    )
                )
        res = pd.DataFrame(
            columns=["shift_z", "shift_y", "maxcorr", "n_barcodes"],
            data=np.vstack([np.hstack(a) for a in reg_out]),
            index=cells_in_ref.index,
        )
        res_befaft[name] = res
    if save_folder is not None:
        save_folder = Path(save_folder)
        assert save_folder.exists(), f"{save_folder} does not exist"
        for name, res in res_befaft.items():
            res.to_csv(save_folder / f"{ref_slice}_{name}_registration.csv")
    return res_befaft


def interpolate_shifts(
    project,
    mouse,
    ref_slice,
    target_position,
    error_correction_ds_name,
    threshold,
    smoothing=10,
    vis=True,
):
    """Interpolate shifts using RBF interpolation

    Args:
        project (str): Project name
        mouse (str): Mouse name
        ref_slice (str): Reference slice name (format `{chamber}_{roi:02d}`)
        target_position (str): Target position name (`previous`, `next` or `n_{n}`)
        error_correction_ds_name (str): Dataset name for error correction
        threshold (float): Maximum distance to consider a shift
        smoothing (float, optional): Smoothing factor for RBF interpolation. Defaults to
            10.
        vis (bool, optional): Plot diagnostics. Defaults to True.

    Returns:
        np.array: Smoothed shifts
        RBFInterpolator: Interpolator for y shifts
        RBFInterpolator: Interpolator for z shifts
    """
    save_folder = (
        get_processed_path(project) / mouse / "analysis" / "serial_section_registration"
    )
    res_file = save_folder / f"{ref_slice}_{target_position}_registration.csv"
    assert res_file.exists(), f"{res_file} does not exist"
    res = pd.read_csv(res_file, index_col=0)
    # find cells in the ref slice
    ref_roi = int(ref_slice.split("_")[-1])
    ref_chamber = "_".join(ref_slice.split("_")[:-1])

    (
        rab_spot_df,
        _,
        rabies_cell_properties,
    ) = get_barcode_in_cells(
        project,
        mouse,
        error_correction_ds_name,
        valid_chambers=[ref_chamber],
        save_folder=None,
        verbose=False,
        add_ara_properties=True,
    )

    # add rotated ara coordinates
    transform = ara_registration.get_ara_to_slice_rotation_matrix(
        spot_df=rab_spot_df, verbose=False
    )
    rabies_cell_properties = ara_registration.rotate_ara_coordinate_to_slice(
        rabies_cell_properties, transform=transform, verbose=False
    )
    cells_in_ref = rabies_cell_properties.query(
        f"chamber == '{ref_chamber}' and roi == {ref_roi}"
    )

    # Find cells with valid shifts
    shifts = res[["shift_z", "shift_y"]].values
    shift_ampl = np.linalg.norm(shifts, axis=1)
    valid = shift_ampl < threshold
    shifts = shifts[valid]
    good_idx = res.index[valid]
    cell_coords = cells_in_ref.loc[good_idx, ["ara_z_rot", "ara_y_rot"]]
    z_shift_interpolator = RBFInterpolator(
        cell_coords, shifts[:, 0], smoothing=smoothing
    )
    y_shift_interpolator = RBFInterpolator(
        cell_coords, shifts[:, 1], smoothing=smoothing
    )

    # Add missing cells in res, these are cells that had NaN in some coords
    missing = cells_in_ref.index.difference(res.index)
    res.loc[missing] = np.nan

    all_cell_coords = cells_in_ref[["ara_z_rot", "ara_y_rot"]].values
    smooth_shifts = np.stack(
        [z_shift_interpolator(all_cell_coords), y_shift_interpolator(all_cell_coords)],
        axis=1,
    )
    res.loc[cells_in_ref.index, ["smooth_shift_z", "smooth_shift_y"]] = smooth_shifts
    # add also cell coordinates to the res dataframe
    res.loc[cells_in_ref.index, ["ara_z_rot", "ara_y_rot"]] = all_cell_coords
    print("Saving results")
    res.to_csv(res_file)

    # Plot diagnostics
    if vis:
        fig = diagnostics.plot_shifts_interpolation(res, threshold)
        fig.suptitle(f"{ref_slice} - {target_position}")
        fig.savefig(
            save_folder / f"{ref_slice}_{target_position}_shifts_interpolation.png"
        )

    return res, z_shift_interpolator, y_shift_interpolator


def register_local_spots(
    center_point: np.array,
    spot_df: pd.DataFrame,
    ref_slice: str,
    target_slice: str,
    window_size: float,
    min_spots: int = 5,
    max_barcode_number: int = 500,
    gaussian_width: float = 30,
    verbose: bool = True,
    debug: bool = False,
):
    """Register spots in two serial sections using phase correlation

    This function will select spots in two serial sections that are close to a given
    center point and have a minimum number of spots in common. It will then create
    spot images for each slice and do phase correlation to find the shift between the
    two slices.

    Args:
        center_point (np.array): Center point (ara_y_rot, ara_z_rot) in mm
        spot_df (pd.DataFrame): DataFrame with spots.
        ref_slice (str): Reference slice name (format `{chamber}_{roi:02d}`)
        target_slice (str): Target slice name (format `{chamber}_{roi:02d}`)
        window_size (float): Window size in um
        min_spots (int, optional): Minimum number of spots in common. Defaults to 5.
        max_barcode_number (int, optional): Maximum number of barcodes to consider.
            Defaults to 500.
        gaussian_width (float, optional): Width of the gaussian kernel for spot images.
            Defaults to 30.
        verbose (bool, optional): Print progress. Defaults to True.
        debug (bool, optional): Return additional information. Defaults to False.

    Returns:
        np.array: Shift between the two slices
        float: Maximum correlation value
        np.array: Phase correlation results, if debug is True
        np.array: Spot images, if debug is True
        pd.Index: Selected barcodes, if debug is True

    """
    center_point = np.asarray(center_point).astype(float)
    if "ara_y_rot" not in spot_df.columns:
        spot_df = ara_registration.rotate_ara_coordinate_to_slice(spot_df)
    if "slice" not in spot_df.columns:
        spot_df["slice"] = (
            spot_df.chamber + "_" + spot_df["roi"].map(lambda x: f"{x:02d}")
        )

    # The ara coordinates are in mm, we will do everything in um and make spot images
    # with 1um/px, so lot of /1000 and *1000
    win_around = np.array([-1, 1]) * window_size / 1000 + center_point[None, :].T
    if verbose:
        print(
            f"Cropping around {np.round(center_point,2)} with window of {window_size}um"
        )

    barcodes_by_roi = []
    spots_by_roi = []
    for slice in [ref_slice, target_slice]:
        spots = spot_df.query(f"slice == '{slice}'")
        for i, coord in enumerate("yz"):
            w = win_around[i]
            spots = spots.query(
                f"ara_{coord}_rot >= {w[0]} and ara_{coord}_rot <= {w[1]}"
            )
        spots_by_roi.append(spots)
        barcodes_by_roi.append(set(spots.corrected_bases.unique()))
    barcodes = barcodes_by_roi[0].intersection(barcodes_by_roi[1])
    if verbose:
        print(
            f"Found {len(barcodes)} barcodes in common (intersection of "
            + f"{len(barcodes_by_roi[0])} and {len(barcodes_by_roi[1])})"
        )
    # select the barcodes that are present in both slices in large numbers
    spots = pd.concat(spots_by_roi)
    if verbose:
        print(f"Found {len(spots)} spots in the surrounding slice")
    spots = spots.query("corrected_bases in @barcodes")
    bc_per_roi = spots.groupby(["slice", "corrected_bases"]).size().unstack().fillna(0)
    best_barcodes = bc_per_roi.min(axis=0).sort_values(ascending=False)
    best_barcodes = best_barcodes[best_barcodes > min_spots]
    if len(best_barcodes) > max_barcode_number:
        best_barcodes = best_barcodes.head(max_barcode_number)

    spots = spots.query("corrected_bases in @best_barcodes.index").copy()
    spots["convolution_y"] = (spots.ara_y_rot * 1000).astype(int)
    spots["convolution_z"] = (spots.ara_z_rot * 1000).astype(int)
    if verbose:
        print(
            f"Found {len(spots)} spots in the pair of slices with the selected "
            + f"{len(best_barcodes)} barcodes"
        )
    if len(best_barcodes) == 0:
        # create empty shift, maxcorr and n_barcodes
        out = np.empty(2) * np.nan, np.nan, 0
        if debug:
            out += (
                np.zeros((0, 2)),  # shifts
                np.zeros(0),  # max_corrs
                np.zeros((0, 2, 0, 0)),  # phase_corrs
                np.zeros((0, 0, 0)),  # spot_images
                pd.Index([]),  # best_barcodes.index
            )
        return out

    origin = np.array([spots.convolution_y.min(), spots.convolution_z.min()])
    output_shape = (np.array([2, 2]) * window_size + 1).astype(int)

    spot_images = np.empty((len(best_barcodes), 2, *output_shape), dtype="single")
    if verbose:
        print(f"Creating spot images with shape {output_shape}")
    for ibc, bc in tqdm(
        enumerate(best_barcodes.index), total=len(best_barcodes), disable=not verbose
    ):
        bc_df = spots[spots["corrected_bases"] == bc]
        for islice, slice in enumerate([ref_slice, target_slice]):
            slice_df = bc_df.query("slice == @slice")
            # rename to x, y for make_spot_image
            sp = pd.DataFrame(
                slice_df[["convolution_y", "convolution_z"]].values - origin,
                columns=["x", "y"],
            )
            img = make_spot_image(
                sp,
                gaussian_width=gaussian_width,
                dtype="single",
                output_shape=output_shape,
            )
            spot_images[best_barcodes.index.get_loc(bc), islice] = img

    # do phase correlation for each pair
    shifts = np.zeros((len(best_barcodes), 2))
    max_corrs = np.zeros(len(best_barcodes))
    phase_corrs = np.zeros((len(best_barcodes), *output_shape))
    if verbose:
        print("Doing phase correlation")
    for ibc in tqdm(range(len(best_barcodes)), disable=not verbose):
        ref = np.nan_to_num(spot_images[ibc, 0])
        target = np.nan_to_num(spot_images[ibc, 1])
        shifts[ibc], max_corrs[ibc], phase_corrs[ibc], _ = phase_correlation(
            ref, target, whiten=False
        )
    avg_corr = phase_corrs.mean(axis=0)
    # find the max and the corresponding shift
    maxcorr = np.max(avg_corr)
    img_shape = np.array(avg_corr.shape)
    shift = np.unravel_index(np.argmax(avg_corr), img_shape) - np.array(img_shape) / 2

    if verbose:
        print(f"Max correlation: {maxcorr} at shift {shift}")
    out = shift, maxcorr, len(best_barcodes)
    if debug:
        out += shifts, max_corrs, phase_corrs, spot_images, best_barcodes.index
    return out
