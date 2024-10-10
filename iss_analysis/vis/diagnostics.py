import numpy as np
import matplotlib.pyplot as plt
from ..registration import register_serial_sections


def check_serial_registration(
    cell_info,
    ref_slice,
    target_slice,
    rab_spot_df,
    rabies_cell_properties,
    window_size=300,
    min_spots=10,
    max_barcode_number=50,
    gaussian_width=30,
    spots_kwargs=None,
    shifts_to_use=None,
):
    """Plot the spots in the reference and target slice and the phase correlation
        between them.

    Args:
        cell_info (pd.Series): Series with the cell information. Must have the columns
            "ara_y_rot" and "ara_z_rot"
        ref_slice (str): Slice name of the reference slice
        target_slice (str): Slice name of the target slice
        rab_spot_df (pd.DataFrame): DataFrame with the spots
        rabies_cell_properties (pd.DataFrame): DataFrame with the cell properties
        window_size (int): Size of the window in microns
        min_spots (int): Minimum number of spots to consider a barcode
        max_barcode_number (int): Maximum number of barcodes to consider
        gaussian_width (int): Width of the gaussian filter to apply to the spots
        spots_kwargs (dict): Dictionary with the arguments to pass to the scatter plot
            Default is None
        shifts_to_use (np.ndarray): Array with the shifts to use. Default is None

    Returns:
        fig (plt.Figure): Figure with the plots
    """

    if "slice" not in rabies_cell_properties.columns:
        rabies_cell_properties["slice"] = (
            rabies_cell_properties.chamber
            + "_"
            + rabies_cell_properties.roi.map(lambda x: f"{x:02d}")
        )

    (
        shift,
        maxcorr,
        n_bcs,
        all_shifts,
        max_corrs,
        phase_corrs,
        spot_images,
        best_barcodes,
    ) = register_serial_sections.register_local_spots(
        center_point=(cell_info.ara_y_rot, cell_info.ara_z_rot),
        spot_df=rab_spot_df,
        ref_slice=ref_slice,
        target_slice=target_slice,
        window_size=window_size,
        min_spots=min_spots,
        max_barcode_number=max_barcode_number,
        gaussian_width=gaussian_width,
        verbose=True,
        debug=True,
    )
    # get spots with best_barcodes
    kw = dict(cmap="tab20", vmin=0, vmax=19, s=5, alpha=0.5)
    kw.update(spots_kwargs or {})
    nr = np.ceil(len(best_barcodes) / 2).astype(int)
    nc = 8 if len(best_barcodes) > 1 else 4
    fig, axes = plt.subplots(nr, nc, figsize=(2 * nc, 2 * nr), squeeze=False)
    ws = window_size / 1000
    spot_part = rab_spot_df.query(
        f"ara_y_rot > {cell_info.ara_y_rot - ws} "
        + f"and ara_y_rot < {cell_info.ara_y_rot + ws} "
        + f"and ara_z_rot > {cell_info.ara_z_rot - ws} "
        + f"and ara_z_rot < {cell_info.ara_z_rot + ws}"
    )
    midpoint = phase_corrs[0].shape[0] // 2
    # make an extent array around cell_info
    if shifts_to_use is None:
        sh = shift / 1000
    else:
        sh = np.array(shifts_to_use, dtype=float) / 1000
    extent = [
        cell_info.ara_y_rot - ws,
        cell_info.ara_y_rot + ws,
        cell_info.ara_z_rot - ws,
        cell_info.ara_z_rot + ws,
    ]
    for ibc, bc in enumerate(best_barcodes):
        spots_ref = spot_part.query(f"slice == @ref_slice and corrected_bases == @bc")
        spots_target = spot_part.query(
            f"slice == @target_slice and corrected_bases == @bc"
        )
        axes[ibc % nr, 0 + 4 * (ibc // nr)].imshow(
            spot_images[ibc][0], cmap="Greys", extent=extent, origin="lower"
        )
        axes[ibc % nr, 0 + 4 * (ibc // nr)].scatter(
            spots_ref.ara_y_rot, spots_ref.ara_z_rot, c=spots_ref.barcode_id % 20, **kw
        )
        axes[ibc % nr, 1 + 4 * (ibc // nr)].imshow(
            spot_images[ibc][0], cmap="Greys", extent=extent, origin="lower"
        )
        axes[ibc % nr, 1 + 4 * (ibc // nr)].scatter(
            spots_target.ara_y_rot,
            spots_target.ara_z_rot,
            c=spots_target.barcode_id % 20,
            **kw,
        )
        axes[ibc % nr, 2 + 4 * (ibc // nr)].imshow(
            spot_images[ibc][0], cmap="Greys", extent=extent, origin="lower"
        )
        axes[ibc % nr, 2 + 4 * (ibc // nr)].scatter(
            spots_target.ara_y_rot + sh[1],
            spots_target.ara_z_rot + sh[0],
            c=spots_target.barcode_id % 20,
            **kw,
        )
        axes[ibc % nr, 3 + 4 * (ibc // nr)].imshow(
            phase_corrs[ibc], cmap="viridis", origin="lower"
        )
        # find the row/col of the max of phase_corrs[ibc]
        i, j = np.unravel_index(np.argmax(phase_corrs[ibc]), phase_corrs[ibc].shape)
        axes[ibc % nr, 3 + 4 * (ibc // nr)].plot([midpoint, j], [midpoint, i], "k-o")
        axes[ibc % nr, 3 + 4 * (ibc // nr)].scatter(
            shift[1] + midpoint, shift[0] + midpoint, c="r"
        )

    for x in axes.flatten():
        x.axis("equal")
        x.set_xticks([])
        x.set_yticks([])
    plt.tight_layout()
    return fig


def plot_shifts_interpolation(res, threshold):
    kwargs = dict(
        cmap="cividis", clim=[0, threshold], angles="xy", scale_units="xy", scale=1000
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ampl = np.linalg.norm([res.shift_z, res.shift_y], axis=0)
    qu = axes[0].quiver(
        res.ara_z_rot, res.ara_y_rot, res.shift_z, res.shift_y, ampl, **kwargs
    )
    axes[0].set_title("Raw shifts")
    fig.colorbar(qu, ax=axes[0])

    ampl = np.linalg.norm([res.smooth_shift_z, res.smooth_shift_y], axis=0)
    qu = axes[1].quiver(
        res.ara_z_rot,
        res.ara_y_rot,
        res.smooth_shift_z,
        res.smooth_shift_y,
        ampl,
        **kwargs,
    )
    axes[1].set_title("Smooth shifts")
    fig.colorbar(qu, ax=axes[1])

    res["delta_z"] = res.smooth_shift_z - res.shift_z
    res["delta_y"] = res.smooth_shift_y - res.shift_y
    res["delta_ampl"] = np.linalg.norm([res.delta_z, res.delta_y], axis=0)
    kwargs.update(clim=[0, min(threshold, res.delta_ampl.max())])
    kwargs.update(cmap="cool")

    qu = axes[2].quiver(
        res.ara_z_rot, res.ara_y_rot, res.delta_z, res.delta_y, res.delta_ampl, **kwargs
    )
    axes[2].set_title("Smooth - Raw shifts")
    fig.colorbar(qu, ax=axes[2])
    for ax in axes:
        ax.set_aspect("equal")
        ax.invert_yaxis()
    return fig
