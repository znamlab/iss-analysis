import numpy as np
import pandas as pd
from tqdm import tqdm
from image_tools.registration.phase_correlation import phase_correlation
from iss_preprocess.segment.spots import make_spot_image

from . import ara_registration


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
        center_point (np.array): Center point in ARA coordinates
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
        out = np.empty(2) * np.nan, np.nan, np.nan
        if debug:
            out += (np.zeros((0, 2, 0, 0)), np.zeros((0, 0, 0)), pd.Index([]))
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
        for islice, (slice, slice_df) in enumerate(bc_df.groupby("slice")):
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
    sum_corr = phase_corrs.sum(axis=0)
    # find the max and the corresponding shift
    maxcorr = np.max(sum_corr)
    argmax = np.array(np.unravel_index(np.argmax(sum_corr), sum_corr.shape))
    # shift is relative to center of image
    shift = argmax - np.array(sum_corr.shape) // 2
    if verbose:
        print(f"Max correlation: {maxcorr} at shift {shift}")
    out = shift, maxcorr, len(best_barcodes)
    if debug:
        out += phase_corrs, spot_images, best_barcodes.index
    return out
