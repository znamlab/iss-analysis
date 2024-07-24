# define helper to plot resutls
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm


def get_stack_part(stack, xlim, ylim):
    ylim = sorted(ylim)
    xlim = sorted(xlim)
    return stack[ylim[0] : ylim[1], xlim[0] : xlim[1]]


def get_spot_part(df, xlim, ylim, return_mask=False):
    ylim = sorted(ylim)
    xlim = sorted(xlim)
    mask = (
        (df["x"] >= xlim[0])
        & (df["x"] < xlim[1])
        & (df["y"] >= ylim[0])
        & (df["y"] < ylim[1])
    )
    if return_mask:
        return df[mask], mask
    return df[mask]


def plot_bc_over_mask(
    ax,
    ma,
    bc,
    mask_assignment,
    xlim,
    ylim,
    nc=20,
    show_bg_barcodes=False,
    mask_alpha=0.5,
):
    # get the list of colors from tab20
    colors = matplotlib.cm.get_cmap("tab20", nc).colors
    im = ax.imshow(
        get_stack_part(ma, xlim, ylim) % nc,
        cmap="tab20",
        vmin=0,
        vmax=nc - 1,
        interpolation="none",
        zorder=-100,
        alpha=mask_alpha,
    )
    # centroids = get_spot_part()
    sp_col = mask_assignment % nc
    too_far = mask_assignment == -2
    background = mask_assignment == -1
    assigned = mask_assignment >= 0
    barcodes = list(bc.corrected_bases.unique())
    bc_color = np.array([barcodes.index(b) for b in bc.corrected_bases]).astype(int)
    ax.scatter(
        bc.x.values[too_far] - xlim[0],
        bc.y.values[too_far] - ylim[0],
        color="w",
        edgecolors="k",
        linewidths=0.2,
        s=5,
        alpha=1,
    )
    if show_bg_barcodes:
        ec = [colors[i] for i in bc_color[background] % nc]
        alpha = 1
        s = 20
    else:
        ec = "none"
        alpha = 0.5
        s = 5
    ax.scatter(
        bc.x.values[background] - xlim[0],
        bc.y.values[background] - ylim[0],
        color="k",
        edgecolors=ec,
        s=s,
        alpha=alpha,
        marker="o",
    )
    ec = [colors[i] for i in bc_color[assigned] % nc]
    fc = [colors[i] for i in sp_col[assigned]]
    ax.scatter(
        bc.x.values[assigned] - xlim[0],
        bc.y.values[assigned] - ylim[0],
        facecolors=fc,
        edgecolors=ec,
        s=30,
        alpha=1,
        marker="o",
    )
    ax.axis("off")
