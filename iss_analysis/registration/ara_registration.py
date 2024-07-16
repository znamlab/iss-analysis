import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import lstsq


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
    A = np.c_[fixed_ara_coords[in_brain, :2], np.ones(in_brain.sum())]
    coefs, _, _, _ = np.linalg.lstsq(A, fixed_ara_coords[in_brain, 2], rcond=None)
    a, b, d = coefs
    c = -1
    if verbose:
        print(f"Plane equation: z = {a:.2f}x + {b:.2f}y + {d:.2f}")

    # Project moving ara to fixed plane

    projected_ara = _project_points(moving_ara, (a, b, d))
    for i, col in enumerate(["ara_x_proj", "ara_y_proj", "ara_z_proj"]):
        moving_spots[col] = projected_ara[:, i]
    md = np.mean(_point_to_plane_distance(moving_ara, (a, b, d))) * 1000
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
        plane_coeffs (tuple): Tuple of the plane coefficients (a, b, d).

    Returns:
        distance (np.ndarray): Array of distances from the points to the plane
    """
    a, b, d = plane_coeffs
    numerator = a * points[:, 0] + b * points[:, 1] - points[:, 2] + d
    denominator = np.sqrt(a**2 + b**2 + 1)
    distance = numerator / denominator
    return distance


def _project_points(points, plane_coeffs):
    """Project 3D points to a plane.

    Args:
        points (np.ndarray): Nx3 Array of point coordinates.
        plane_coeffs (tuple): Tuple of the plane coefficients (a, b, d).

    Returns:
        projected_points (np.ndarray): Nx3 Array of projected point coordinates.
    """
    distance = _point_to_plane_distance(points, plane_coeffs)
    a, b, d = plane_coeffs
    normal = np.array([a, b, -1])
    normal /= np.linalg.norm(normal)
    projected_points = points - distance[:, np.newaxis] * normal
    return projected_points
