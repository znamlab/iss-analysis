import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.linalg import lstsq
from skimage.transform import resize
import bg_atlasapi as bga

from znamutils import slurm_it
from iss_preprocess.io.save import write_stack
from iss_preprocess.io.load import get_processed_path
from iss_preprocess.pipeline.ara_registration import (
    load_coordinate_image,
    make_area_image,
)
from iss_preprocess.pipeline.stitch import stitch_registered

from . import utils


def get_ara_to_slice_rotation_matrix(spot_df, ref_chamber=None, ref_roi=None):
    """Rotate the ARA coordinates to the slice orientation.

    The ARA coordinates are in the DV, AP, ML orientation. We want to rotate the AP/ML
    dimension so that the first dimension is parallel to the slicing plane.

    Args:
        spot_df (pd.DataFrame): DataFrame with the spots to rotate. Must contain columns
            'ara_x', 'ara_y', 'ara_z', 'chamber', 'roi'
        ref_chamber (str, optional): Reference chamber name. If None, will use the
            median plane of all chambers in the DataFrame. Defaults to None.
        ref_roi (int, optional): Reference roi number. If None, will use the median
            plane of all rois in the DataFrame. Defaults to None.

    Returns:
        np.ndarray: 3x3 Rotation matrix to apply to the ARA coordinates to rotate them
            to the slice orientation.
    """
    if ref_chamber is None:
        if ref_roi is not None:
            raise ValueError("If ref_roi is not None, ref_chamber must be provided")
        chamber_rois = spot_df.groupby(["chamber", "roi"]).groups.keys()
    else:
        chamber_rois = [(ref_chamber, ref_roi)]
    # exclude spots that are out of the brain
    spot_df = spot_df.query("area_id > 0")
    # any set of 3 spots should be enough to determine the plane, but there is some
    # variability (pixel resolution?) so we will use a bunch of the spots
    max_n_spots = 1000
    all_planes = []
    for chamber, roi in chamber_rois:
        spots = spot_df.query(f"chamber == '{chamber}' and roi == {roi}")
        ara_coords = spots[[f"ara_{i}" for i in "xyz"]].values
        ara_coords = ara_coords[~np.any(np.isnan(ara_coords), axis=1)]
        if not len(ara_coords):
            print(f"No spots in chamber {chamber}, roi {roi}")
            continue
        nspots_to_use = min(len(ara_coords), max_n_spots)
        plane_coeffs = utils.fit_plane_to_points(
            ara_coords[:: len(ara_coords) // nspots_to_use, :]
        )
        all_planes.append(plane_coeffs)
    all_planes = np.array(all_planes)
    median_plane = np.median(all_planes, axis=0)

    # plane normal:
    n = -median_plane[:3] / np.linalg.norm(median_plane[:3])

    # find a vector that is in the plane and pointing dorsaly
    # up is [0,1,0], the plane equation is ax+by+cz=0, so if we set x=0, we get
    # by+cz=0, so y=1 and z=-b/c
    assert plane_coeffs[1] != 0, "Failed to fit. Plane is vertical"
    v = np.array([0, 1, -plane_coeffs[2] / plane_coeffs[1]])
    v /= np.linalg.norm(v)
    # dim 1 is orthogonal to the plane normal and u
    u = np.cross(n, v)
    u /= np.linalg.norm(u)
    if abs(plane_coeffs[2]) > abs(plane_coeffs[1]):
        # if the plane is more vertical than horizontal, swap u and v
        u, v = v, u
    new_base = np.c_[n, v, u]
    return new_base


def rotate_ara_coordinate_to_slice(
    spot_df, transform=None, ref_chamber=None, ref_roi=None
):
    """Rotate the ARA coordinates to the slice orientation.

    The ARA coordinates are in the DV, AP, ML orientation. We want to rotate the AP/ML
    dimension so that the first dimension is parallel to the slicing plane.

    Args:
        spot_df (pd.DataFrame): DataFrame with the spots to rotate. Must contain columns
            'ara_x', 'ara_y', 'ara_z', 'chamber', 'roi'
        transform (np.ndarray, optional): Rotation matrix to apply to the ARA
            coordinates to rotate them to the slice orientation. If None, will determine
            the matrix using the reference chamber and roi. Defaults to None.
        ref_chamber (str, optional): Reference chamber name. If None, will use the
            median plane of all chambers in the DataFrame. Defaults to None.
        ref_roi (int, optional): Reference roi number. If None, will use the median
            plane of all rois in the DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with the rotated ARA coordinates (ara_`axis`_rot)
    """
    if transform is None:
        transform = get_ara_to_slice_rotation_matrix(
            spot_df, ref_chamber=ref_chamber, ref_roi=ref_roi
        )

    ara_coords = spot_df[[f"ara_{i}" for i in "xyz"]].values
    rotated_coords = ara_coords @ transform
    for i, col in enumerate("xyz"):
        spot_df[f"ara_{col}_rot"] = rotated_coords[:, i]
    return spot_df


def get_registered_neighbours(
    ref_chamber, ref_roi, project, mouse, reference_df, df_to_register, verbose=False
):
    """Register the spots/cells in df_to_register to the reference slice

    Will use the ARA coordinates as intermediate space to register the spots/cells.

    Args:
        ref_chamber (str): Reference chamber name
        ref_roi (int): Reference roi number
        project (str): Project name
        mouse (str): Mouse name
        reference_df (pd.DataFrame): DataFrame with the spots to use for finding the ARA
            plane of the reference slice
        df_to_register (pd.DataFrame): DataFrame with the spots or cells to register
        verbose (bool, optional): If True, will print information about the process.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the registered spots/cells for the reference slice
            and the surrounding slices
    """
    surrounding_rois = utils.get_surrounding_slices(
        ref_chamber, ref_roi, project, mouse, include_ref=True
    )
    ref_spots = reference_df.query("chamber == @ref_chamber and roi == @ref_roi").copy()

    # now project ara_x, ara_y, ara_z on the plane of the reference roi
    out = []
    for sec, series in surrounding_rois.iterrows():
        chamber, roi = series[["chamber", "roi"]]
        spots = df_to_register.query("chamber == @chamber and roi == @roi")
        reg_cell = project_to_other_roi(
            ref_spots, spots, verbose=verbose, method="interpolate"
        )
        out.append(reg_cell)
    out = pd.concat(out, ignore_index=False)
    return out


def project_to_other_roi(fixed_spots, moving_spots, method="interpolate", verbose=True):
    """Project moving spots to fixed spots using ARA registration.

    Will fit a plane to the fixed spots ARA coordinates and project the moving spots to
    that plane. Then the projected slice to ARA for fixed spots mapping will be
    interpolated to find the slice position of the projected moving spots.

    Args:
        fixed_spots (pd.DataFrame): Fixed spots dataframe, must contain x, y, ara_x,
            ara_y, ara_z, area_id.
        moving_spots (pd.DataFrame): Moving spots dataframe, must contain ara_x, ara_y,
            ara_z
        method (str, optional): Method to use for the file interpolation. One of
            `linear`, `interpolate`. Defaults to 'interpolate'.
        verbose (bool, optional): If True, will print information about the process.
            Defaults to True.

    Returns:
        pd.DataFrame: Moving spots dataframe with added columns 'ara_x_proj',
            'ara_y_proj', 'ara_z_proj', 'slice_proj_x', 'slice_proj_y
    """
    moving_ara = moving_spots[["ara_x", "ara_y", "ara_z"]].values
    bad = np.any(np.isnan(moving_ara), axis=1)
    moving_spots = moving_spots[~bad].copy()
    moving_ara = moving_spots[["ara_x", "ara_y", "ara_z"]].values
    if verbose and bad.sum():
        print(f"Found {bad.sum()} NaNs in moving spots, removed")

    coords = fixed_spots[["ara_x", "ara_y", "ara_z", "x", "y"]].values
    bad = np.any(np.isnan(coords), axis=1)
    if verbose and bad.sum():
        print(f"Found {bad.sum()} NaNs in fixed spots, removed")

    fixed_spots = fixed_spots[~bad].copy()
    fixed_ara_coords = fixed_spots[["ara_x", "ara_y", "ara_z"]].values
    in_brain = fixed_spots["area_id"] != 0
    fixed_slice_coords = fixed_spots[["x", "y"]].values
    # add a column of z = 0 to the fixed slice coords
    fixed_slice_coords = np.c_[fixed_slice_coords, np.zeros(len(fixed_slice_coords))]

    # Fit a plane, a x + b y + d = z to the fixed spots
    plane_coeffs = utils.fit_plane_to_points(fixed_ara_coords)
    a, b, c, d = plane_coeffs
    if verbose:
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f} z + {d:.2f} = 0")

    # Project moving ara to fixed plane

    projected_ara = _project_points(moving_ara, plane_coeffs)
    for i, col in enumerate(["ara_x_proj", "ara_y_proj", "ara_z_proj"]):
        moving_spots[col] = projected_ara[:, i]
    md = np.mean(_point_to_plane_distance(moving_ara, plane_coeffs)) * 1000
    if abs(md) > 50:
        raise ValueError(f"Average distance to plane is too high: {md:.2f} um")
    if verbose:
        print(f"Projected moving spots to plane, average distance: {md:.2f} um")

    def _linear_interp(fixed_ara, fixed_slice, moving_ara):
        tform, residuals, _, _ = lstsq(fixed_ara, fixed_slice)
        moving_slice_coords = np.dot(moving_ara, tform)
        return moving_slice_coords

    # Interpolate
    match method:
        case "interpolate":
            interpolator = LinearNDInterpolator(fixed_ara_coords, fixed_slice_coords)
            moving_slice_coords = interpolator(projected_ara)
            outofhull = np.any(np.isnan(moving_slice_coords), axis=1)
            # linear interpolation for the out of hull
            moving_slice_coords[outofhull] = _linear_interp(
                fixed_ara_coords, fixed_slice_coords, projected_ara[outofhull]
            )
            if verbose:
                print(f"{outofhull.sum()}/{len(outofhull)} out of hull used linear")
        case "linear":
            moving_slice_coords = _linear_interp(
                fixed_ara_coords, fixed_slice_coords, projected_ara
            )
        case _:
            raise ValueError(
                f"Unknown method: {method}." + " Must be one of 'linear', 'interpolate'"
            )
    for i, col in enumerate(["slice_proj_x", "slice_proj_y"]):
        moving_spots[col] = moving_slice_coords[:, i]

    return moving_spots


def _point_to_plane_distance(points, plane_coeffs):
    """Calculate the distance from a 3D point to a plane.

    Args:
        points (np.ndarray): Nx3 Array of point coordinates.
        plane_coeffs (tuple): Tuple of the plane coefficients (a, b, c,d).

    Returns:
        distance (np.ndarray): Array of distances from the points to the plane
    """
    a, b, c, d = plane_coeffs
    numerator = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    denominator = np.sqrt(a**2 + b**2 + c**2)
    distance = numerator / denominator
    return distance


def _project_points(points, plane_coeffs):
    """Project 3D points to a plane.

    Args:
        points (np.ndarray): Nx3 Array of point coordinates.
        plane_coeffs (tuple): Tuple of the plane coefficients (a, b, c, d).

    Returns:
        projected_points (np.ndarray): Nx3 Array of projected point coordinates.
    """
    distance = _point_to_plane_distance(points, plane_coeffs)
    a, b, c, d = plane_coeffs
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    projected_points = points - distance[:, np.newaxis] * normal
    return projected_points


@slurm_it(conda_env="iss-preprocess")
def transform_stack_to_ara(
    project,
    mouse,
    chamber,
    roi,
    prefix,
    channels,
    error_correction_ds_name,
    output_px_size,
    interpolate=True,
    output_folder=None,
):
    """Transform the registered stack to ARA coordinates.

    Will load the registered stack, the ARA coordinates and the rabies spots to find the
    main plane of the data. Then will rotate the ARA coordinates to the slice orientation
    and transform the stack to the ARA coordinates.

    Args:
        project (str): Project name
        mouse (str): Mouse name
        chamber (str): Chamber name
        roi (int): ROI number
        prefix (str): Prefix of the registered stack (e.g. "mCherry_1")
        channels (list): List of channels to use
        error_correction_ds_name (str): Name of the error correction dataset to load
            rabies spots and find the main ara plane
        output_px_size (int, optional): Output pixel size in um. Should be higher than
            that used for the registration. Defaults to 10.
        interpolate (bool, optional): If True, will interpolate the stack to the new
            coordinates, otherwise will use closest pixel. Defaults to True.
        output_folder (str, optional): Folder to save the rotated stack. If None, will
            not save the stack. Defaults to None.

    Returns:
        np.ndarray: Rotated stack
    """
    print(f"Transforming stack to ARA coordinates for {mouse}/{chamber}/{roi}")
    data_path = get_processed_path(f"{project}/{mouse}/{chamber}")
    print("    ... stitching")
    full_stack = stitch_registered(
        data_path,
        prefix=prefix,
        roi=roi,
        channels=channels,
    )
    ara_coords = load_coordinate_image(data_path, roi, full_scale=False)
    # downsample stack to ara coordinates shape
    stack = np.empty((ara_coords.shape[0], ara_coords.shape[1], full_stack.shape[-1]))
    for i in range(full_stack.shape[-1]):
        stack[..., i] = resize(full_stack[..., i], ara_coords.shape[:2])
    del full_stack

    # find the main ara plane of the data using rabies spots
    ara_info_folder = (
        get_processed_path(f"{project}/{mouse}") / "analysis" / "ara_infos"
    )
    target = (
        ara_info_folder
        / f"{error_correction_ds_name}_{chamber}_{roi}_rabies_spots_ara_info.pkl"
    )
    transform = get_ara_to_slice_rotation_matrix(spot_df=pd.read_pickle(target))

    # rotate the ara coordinates
    ara_coords_rot = ara_coords.reshape(-1, 3) @ transform
    ara_coords_rot = ara_coords_rot.reshape(ara_coords.shape)
    # the new ara coordinates are in mm, and are ~AP, ~DV, ~ML, with AP being hopefully
    # constant.

    # transform the stack to ara coordinates
    shapes = np.vstack([stack.shape[:2], ara_coords_rot.shape[:2]])
    if np.any(np.diff(shapes, axis=0)):
        raise ValueError("Stack and ara_coords_rot must have the same shape")

    # load the atlas
    atlas_name = "allen_mouse_10um"
    bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)
    # and find the limits of the ARA space
    ara_shape_mm = np.array(bg_atlas.shape_um) / 1000
    ara_shape_mm_rot = ara_shape_mm @ transform

    # height is DV, width is ML
    height, width = ara_shape_mm_rot[[1, 2]]
    w_px, h_px = (
        np.round(np.array([width, height]) * 1000 / output_px_size).astype(int) + 1
    )
    rotated_stack = np.zeros((h_px, w_px, 2))
    target_px = np.round(ara_coords_rot * 1000 / output_px_size).astype(int)
    target_px = target_px[..., [1, 2]]
    for i in range(2):
        target_px[..., i] = np.clip(target_px[..., i], 0, rotated_stack.shape[i] - 1)
    if interpolate:
        print("    ... interpolating")
        for axis in range(stack.shape[-1]):
            interp = NearestNDInterpolator(
                target_px.reshape((-1, 2)), stack[..., axis].ravel()
            )
            grid = np.meshgrid(np.arange(h_px), np.arange(w_px), indexing="ij")
            rotated_stack[..., axis] = interp(*grid)
    else:
        rotated_stack[target_px[..., 0], target_px[..., 1], :] = stack

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        target = output_folder / f"{prefix}_{chamber}_{roi}_rotated_stack.tif"
        write_stack(rotated_stack, target)
        print(f"Saved rotated stack to {target}")

    return rotated_stack
