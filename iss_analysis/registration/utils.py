import numpy as np
from iss_analysis.io import get_sections_info


def get_surrounding_slices(
    ref_chamber,
    ref_roi,
    project=None,
    mouse=None,
    section_infos=None,
    include_ref=False,
    window=(-1, 1),
):
    """Returns info about the slices above and below the reference slice

    Args:
        ref_chamber (str): chamber name
        ref_roi (int): roi number
        project (str, optional): project name, required if section_infos is None.
            Defaults to None.
        mouse (str, optional): mouse name, required if section_infos is None.
            Defaults to None.
        section_infos (pd.DataFrame, optional): DataFrame with section positions,
            required if project and mouse are None. Defaults to None.
        include_ref (bool, optional): If True, will include the reference slice in the
            output. Defaults to False.
        window (tuple, optional): Tuple with the number of slices above and below the
            reference slice to include. Defaults to (-1, 1).

    Returns:
        surrounding_rois (pd.DataFrame): DataFrame with the surrounding slices
    """
    if section_infos is None:
        section_infos = get_sections_info(project, mouse)

    ref_sec_pos = section_infos.query(
        "chamber == @ref_chamber and roi == @ref_roi"
    ).iloc[0]
    window = np.array(window)
    window[-1] += 1  # for the range function
    surrounding_rois = list(
        range(*np.clip(window + ref_sec_pos.name, 0, len(section_infos)))
    )
    if include_ref:
        if ref_sec_pos.name not in surrounding_rois:
            surrounding_rois.append(ref_sec_pos.name)
    else:
        if ref_sec_pos.name in surrounding_rois:
            surrounding_rois.remove(ref_sec_pos.name)

    surrounding_rois = section_infos.loc[surrounding_rois].copy()
    return surrounding_rois


def fit_plane_to_points(points):
    """Fit a plane to the points

    Args:
        points (np.array): Array with the points to fit the plane. Each row is a point
            and each column is a coordinate.

    Returns:
        np.array: Array with the plane coefficients [a, b, c, d] for the equation
            ax + by + cz + d = 0
    """
    # Fit a plane, a x + b y + d = z to the fixed spots
    A = np.c_[points[:, :2], np.ones(points.shape[0])]
    B = -points[:, 2]
    x = np.linalg.lstsq(A, B, rcond=None)[0]
    c = 1
    a, b, d = x
    return np.array([a, b, c, d])
