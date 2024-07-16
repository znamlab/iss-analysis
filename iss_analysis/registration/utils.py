import numpy as np
from iss_analysis.io import get_section_info


def get_surrounding_slices(
    ref_chamber, ref_roi, project=None, mouse=None, section_infos=None
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

    Returns:
        surrounding_rois (pd.DataFrame): DataFrame with the surrounding slices
    """
    if section_infos is None:
        section_infos = get_section_info(project, mouse)

    ref_sec_pos = section_infos.query(
        "chamber == @ref_chamber and roi == @ref_roi"
    ).iloc[0]
    surrounding_rois = list(
        range(*np.clip(np.array([-1, 2]) + ref_sec_pos.name, 0, len(section_infos)))
    )
    surrounding_rois.remove(ref_sec_pos.name)
    surrounding_rois = section_infos.loc[surrounding_rois].copy()
    return surrounding_rois
