import numpy as np
from tqdm import tqdm
import pandas as pd
from functools import partial
from multiprocessing import Pool
from warnings import warn
from numba import njit

# Sometimes the likelihoods are not exactly 0 but 10-15 or so, so we need to use a
# small number to consider them zero
EPSILON = 1e-6  # small number that is considered zero


def assign_barcodes_to_masks(
    spots,
    masks,
    p=0.9,
    m=0.1,
    background_spot_prior=0.0001,
    spot_distribution_sigma=50,
    max_iterations=100,
    max_distance_to_mask=200,
    inter_spot_distance_threshold=50,
    max_spot_group_size=5,
    max_total_combinations=1e6,
    verbose=1,
    base_column="bases",
    n_workers=1,
):
    """Assign barcodes to masks using a probabilistic model.

    Args:
        spots (pd.DataFrame): DataFrame with the spots. Must contain the columns 'x',
            'y', and `base_column`.
        mask_centres (pd.DataFrame): DataFrame with the mask centres. Must contain the
            columns 'x' and 'y'.
        p (float): Power of the spot count prior. Default is 0.9.
        m (float): Length scale of the spot count prior. Default is 0.1.
        background_spot_prior (float): Prior for the background spots. Default is 0.0001.
        spot_distribution_sigma (float): Sigma for the spot distribution. Default is 20.
        max_iterations (int): Maximum number of iterations. Default is 100.
        max_distance_to_mask (float): Threshold for the distance in pixels between spots
            and masks. Default is 200.
        inter_spot_distance_threshold (float): Threshold for the distance in pixels
            between spots that can be moved together. Default is 50.
        max_spot_group_size (int): Maximum number of spots in a grouped moved. Default
            is 5.
        max_total_combinations (int): Maximum number of combinations to consider. After
            reach this number, groups of larger number of spots will not be considered.
            Default is 1e6.
        verbose (int): Whether to print the progress. Default is False.
        base_column (str): Name of the column with the bases. Default is 'bases'.
        n_workers (int): Number of workers to use. 1 = no parallel processing Default is 1.

    Returns:
        np.ndarray: 1D array with the mask assignment if debug is False. Otherwise 2D
            array with the mask assignment for each iteration.
    """
    if not spots.index.is_unique:
        raise ValueError("Index of spots must be unique")
    if not masks.index.is_unique:
        raise ValueError("Index of masks must be unique")
    # get mask and spot positions
    mask_centers = masks[["x", "y"]].values

    barcodes = spots[base_column].unique().astype(str)
    if verbose > 0:
        print(f"Found {len(barcodes)} barcodes")
    spot_positions_per_bc = [
        spots.loc[spots[base_column].astype(str) == barcode, ["x", "y"]].values
        for barcode in barcodes
    ]

    _assign_by_barcode = partial(
        assign_single_barcode,
        mask_positions=mask_centers,
        max_distance_to_mask=max_distance_to_mask,
        inter_spot_distance_threshold=inter_spot_distance_threshold,
        background_spot_prior=background_spot_prior,
        p=p,
        m=m,
        spot_distribution_sigma=spot_distribution_sigma,
        max_iterations=max_iterations,
        verbose=max(0, verbose - 1),
        debug=False,
        max_spot_group_size=max_spot_group_size,
        max_total_combinations=max_total_combinations,
    )

    if n_workers > 1:
        print(f"Starting parpool with {n_workers} workers", flush=True)
        with Pool(n_workers) as pool:
            assignment_by_bc = list(
                tqdm(
                    pool.imap(_assign_by_barcode, spot_positions_per_bc),
                    total=len(barcodes),
                )
            )
    else:
        assignment_by_bc = list(
            tqdm(map(_assign_by_barcode, spot_positions_per_bc), total=len(barcodes))
        )
    assignments_id = pd.Series(index=spots.index, data=-2, dtype=int)
    for i_bar, barcode in enumerate(barcodes):
        assignments_id.loc[
            spots[base_column].astype(str) == barcode
        ] = assignment_by_bc[i_bar]

    # convert assignment index to actual mask value
    assignments = pd.Series(
        index=assignments_id.index, data=masks.index[assignments_id.values]
    )
    # put back the negative values
    assignments[assignments_id == -1] = -1
    if np.any(assignments_id == -2):
        warn("Some spots were not assigned")
        assignments[assignments_id == -2] = -2
    return assignments


def assign_single_barcode(
    spot_positions,
    mask_positions,
    max_distance_to_mask,
    inter_spot_distance_threshold,
    background_spot_prior,
    p,
    m,
    spot_distribution_sigma,
    max_spot_group_size=5,
    max_total_combinations=1e6,
    verbose=2,
    mask_assignments=None,
    max_iterations=100,
    debug=False,
):
    """Assign a single barcode to masks.

    Iteratively assign spots to masks until no spots are moving.

    Args:
        spot_positions (np.array): Nx2 array of spot positions.
        mask_positions (np.array): Mx2 array of mask positions.
        max_distance_to_mask (float): Maximum distance between spots and masks.
        inter_spot_distance_threshold (float): Maximum distance between spots.
        background_spot_prior (float): Prior for the background spots.
        p (float): Power of the spot count prior.
        m (float): Length scale of the spot count prior.
        spot_distribution_sigma (float): Sigma for the spot distribution.
        max_spot_group_size (int): Maximum number of spots in a group.
        max_total_combinations (int): Maximum number of combinations to consider.
            After this number, groups of larger number of spots will not be considered.
            Default is 1e6.
        verbose (int): 0 does not print anything, 1 prints progress, 2 list combinations
            number. Default 2.
        mask_assignments (np.array): Nx1 array of initial mask assignments. Default None.
        max_iterations (int): Maximum number of iterations. Default 100.
        debug (bool): Whether to return debug information. Default False.

    Returns:
        np.array: 1D array with the mask assignment if debug is False. Otherwise 2D
            array with the mask assignment for each iteration.
    """
    log_background_spot_prior = np.log(background_spot_prior)
    if not len(spot_positions):
        raise ValueError("No spots to assign")

    distances = np.linalg.norm(
        spot_positions[:, None, :] - mask_positions[None, :, :], axis=2
    )
    # no need to look at masks that are far from all spots
    putative_targets = np.min(distances, axis=0) < max_distance_to_mask
    if np.sum(putative_targets) == 0:
        if verbose > 0:
            print("No masks are close enough to the spots")
        return np.full(len(spot_positions), -1)

    distances = distances[:, putative_targets]
    mask_positions = mask_positions[putative_targets]
    log_dist_likelihood = -0.5 * (distances / spot_distribution_sigma) ** 2
    if mask_assignments is None:
        mask_assignments = np.argmin(distances, axis=1)
    else:
        # need to adapt mask assignment to putative targets
        assignment = mask_assignments.copy()
        target_ids = np.where(putative_targets)[0]
        nbg = assignment >= 0
        assert np.all(
            np.isin(assignment[nbg], target_ids)
        ), "Some spots are assigned to unreachable targets"
        assignment[nbg] = [
            np.argwhere(target_ids == mask)[0][0] for mask in assignment[nbg]
        ]

    combinations = valid_spot_combinations(
        spot_positions,
        inter_spot_distance_threshold,
        max_n=max_spot_group_size,
        max_total_combinations=max_total_combinations,
        verbose=verbose > 1,
    )
    new_assignments = mask_assignments.copy()
    if debug:
        all_assignments = [new_assignments.copy()]
    for i in range(max_iterations):
        if verbose > 0:
            print(f"---- Iteration {i} ----")
            # keep level 2 debuging onl for first iteration
            verb_inside = 1 if i else verbose
        else:
            verb_inside = 0
        new_assignments, spot_moved = assign_single_barcode_single_round(
            spot_positions=spot_positions,
            mask_positions=mask_positions,
            mask_assignments=new_assignments,
            max_distance_to_mask=max_distance_to_mask,
            log_background_spot_prior=log_background_spot_prior,
            p=p,
            m=m,
            spot_distribution_sigma=spot_distribution_sigma,
            distances=distances,
            log_dist_likelihood=log_dist_likelihood,
            combinations=combinations,
            verbose=verb_inside,
        )

        if verbose > 0:
            nsp = np.sum([len(combi) for combi in spot_moved])
            print(f"Moved {nsp} spots in {len(spot_moved)} groups")

        if len(spot_moved) == 0:
            if verbose > 0:
                print("Trying to move all spots to background")
            new_assignments, spot_moved = assign_single_barcode_all_to_background(
                spot_positions=spot_positions,
                mask_positions=mask_positions,
                mask_assignments=new_assignments,
                log_background_spot_prior=log_background_spot_prior,
                p=p,
                m=m,
                spot_distribution_sigma=spot_distribution_sigma,
                distances=distances,
                log_dist_likelihood=log_dist_likelihood,
            )
            if len(spot_moved) == 0:
                if debug:
                    all_assignments.append(new_assignments.copy())
                break
            if verbose > 0:
                nsp = np.sum([len(combi) for combi in spot_moved])
                print(f"Moved {nsp} spots in {len(spot_moved)} masks to background")
        if debug:
            all_assignments.append(new_assignments.copy())
    if debug:
        return np.vstack(all_assignments)
    # we looked only at subset of targets, so we need to put the results back
    target_ids = np.where(putative_targets)[0]
    assignments = new_assignments.copy()
    nbg = assignments >= 0
    assignments[nbg] = target_ids[new_assignments[nbg]]

    return new_assignments


def assign_single_barcode_single_round(
    spot_positions,
    mask_positions,
    mask_assignments,
    max_distance_to_mask,
    log_background_spot_prior,
    p,
    m,
    spot_distribution_sigma,
    inter_spot_distance_threshold=None,
    max_spot_group_size=5,
    max_total_combinations=1e6,
    distances=None,
    log_dist_likelihood=None,
    combinations=None,
    verbose=2,
):
    """Single round of mask assignment.

    Args:
        spot_positions (np.array): Nx2 array of spot positions.
        mask_positions (np.array): Mx2 array of mask positions.
        mask_assignments (np.array): Nx1 array of initial mask assignments.
        max_distance_to_mask (float): Maximum distance between spots and masks.
        log_background_spot_prior (float): Log background spot prior.
        p (float): Power of the spot count prior.
        m (float): Length scale of the spot count prior.
        spot_distribution_sigma (float): Sigma of the spot distribution.
        inter_spot_distance_threshold (float, optional): Maximum distance between spots.
            Required if combinations is None. Default None.
        max_spot_group_size (int, optional): Maximum number of spots in a group.
            Ignored if combinations is not None. Default 5.
        max_total_combinations (int, optional): Maximum number of combinations to
            consider. After this number, groups of larger number of spots will not be
            considered. Ignored if combinations is not None. Default is 1e6.
        distances (np.array, optional): NxM array of distances between spots and masks.
            Default None.
        log_dist_likelihood (np.array, optional): NxM array of log likelihoods of the
            distances. Default None.
        combinations (list, optional): List of combinations of spots that are all within
            distance_threshold of each other. Default None.
        verbose (int): 0 does not print anything, 1 prints progress, 2 list combinations
            number. Default 2.

    Returns:
        np.array: new mask assignments.
        list: list of combination of spots that were moved.
    """
    if verbose > 1:
        print(
            f"Assigning {len(spot_positions)} spots to {len(mask_positions)} masks",
            flush=True,
        )
    mask_counts = np.bincount(
        mask_assignments[mask_assignments >= 0], minlength=len(mask_positions)
    )
    if combinations is None:
        if inter_spot_distance_threshold is None:
            raise AttributeError(
                "inter_spot_distance_threshold must be provided "
                + "if combinations is None"
            )
        combinations = valid_spot_combinations(
            spot_positions,
            inter_spot_distance_threshold,
            max_n=max_spot_group_size,
            max_total_combinations=max_total_combinations,
            verbose=verbose > 1,
        )
    if verbose > 1:
        print(f"Found {len(combinations)} valid spot combinations")
    if distances is None:
        distances = np.linalg.norm(
            spot_positions[:, None, :] - mask_positions[None, :, :], axis=2
        )
    if log_dist_likelihood is None:
        log_dist_likelihood = -0.5 * (distances / spot_distribution_sigma) ** 2

    if len(combinations) == 0:
        return None

    # compute the likelihood of each combination
    likelihood_changes, best_targets = np.zeros((2, len(combinations)), dtype=float)
    for i_combi, combi in tqdm(
        enumerate(combinations), total=len(combinations), disable=verbose == 0
    ):
        bg_likelihood = likelihood_change_background_combination(
            combi,
            mask_assignments,
            mask_counts,
            log_dist_likelihood,
            log_background_spot_prior,
            p,
            m,
        )

        likelihood_changes[i_combi] = bg_likelihood
        best_targets[i_combi] = -1
        valid_targets = np.where(distances[combi].max(axis=0) < max_distance_to_mask)[0]
        current_assignments = np.unique(mask_assignments[combi])
        # don't move to a mask that is already assigned
        valid_targets = np.setdiff1d(valid_targets, current_assignments)
        if len(valid_targets) == 0:
            move_likelihood = -np.inf
        else:
            move_likelihood = likelihood_change_move_combination(
                combi,
                target_masks=valid_targets,
                mask_assignments=mask_assignments,
                mask_counts=mask_counts,
                log_dist_likelihood=log_dist_likelihood,
                log_background_spot_prior=log_background_spot_prior,
                p=p,
                m=m,
            )
        # keep only the best move
        if np.max(move_likelihood) > bg_likelihood:
            likelihood_changes[i_combi] = np.max(move_likelihood)
            best_targets[i_combi] = valid_targets[np.argmax(move_likelihood)]

    # sort by likelihood change
    order = np.argsort(likelihood_changes)[::-1]
    mask_changed = set()
    spot_moved = []
    new_assignments = mask_assignments.copy()
    for i_combi in order:
        if likelihood_changes[i_combi] <= EPSILON:
            # we reached the point where we make things worse
            break
        target = best_targets[i_combi]
        if target in mask_changed:
            continue
        spot_combi = combinations[i_combi]
        source_masks = mask_assignments[spot_combi]
        if any([m in mask_changed for m in source_masks]):
            continue
        new_assignments[spot_combi] = target
        spot_moved.append(spot_combi)
        if target != -1:
            mask_changed.add(target)
        mask_changed.update(source_masks[source_masks != -1])

    return new_assignments, spot_moved


def assign_single_barcode_all_to_background(
    spot_positions,
    mask_positions,
    mask_assignments,
    log_background_spot_prior,
    p,
    m,
    spot_distribution_sigma,
    distances=None,
    log_dist_likelihood=None,
):
    """Move all spots of each mask to the background if it increases the likelihood.

    Args:
        spot_positions (np.array): Nx2 array of spot positions.
        mask_positions (np.array): Mx2 array of mask positions.
        mask_assignments (np.array): Nx1 array of initial mask assignments.
        log_background_spot_prior (float): Log background spot prior.
        p (float): Power of the spot count prior.
        m (float): Length scale of the spot count prior.
        spot_distribution_sigma (float): Sigma for the spot distribution.
        distances (np.array, optional): NxM array of distances between spots and masks.
            Default None.
        log_dist_likelihood (np.array, optional): NxM array of log likelihoods of the
            distances. Default None.

    Returns:
        np.array: 1D array with the mask assignment.
        list: list of combination of spots that were moved.
    """
    mask_counts = np.bincount(
        mask_assignments[mask_assignments >= 0], minlength=len(mask_positions)
    )
    if log_dist_likelihood is None:
        if distances is None:
            distances = np.linalg.norm(
                spot_positions[:, None, :] - mask_positions[None, :, :], axis=2
            )
        log_dist_likelihood = -0.5 * (distances / spot_distribution_sigma) ** 2

    spot_moved = []
    for current_mask in np.unique(mask_assignments[mask_assignments >= 0]):
        combi = np.where(mask_assignments == current_mask)[0]
        bg_likelihood = likelihood_change_background_combination(
            combi,
            mask_assignments,
            mask_counts,
            log_dist_likelihood,
            log_background_spot_prior,
            p,
            m,
        )
        if bg_likelihood > EPSILON:
            mask_assignments[combi] = -1
            mask_counts[current_mask] = 0
            spot_moved.append(combi)
    return mask_assignments, spot_moved


def valid_spot_combinations(
    spot_positions: np.array,
    distance_threshold: float,
    max_n: int = 5,
    verbose=True,
    max_total_combinations: int = 1e6,
):
    """Combinations of spots that are all within distance_threshold of each other

    Args:
        spot_positions (np.array): Nx2 array of spot positions
        distance_threshold (float): maximum distance between spots
        max_n (int): maximum number of spots in the combination
        verbose (bool): print progress
        max_total_combinations (int): maximum number of combinations to consider
            Default is 1e6

    Returns:
        list of np.array: list of combinations of spots that are all within
            distance_threshold of each other
    """
    max_n = min(max_n, len(spot_positions))
    if max_n == 0:
        return []
    all_groups = [np.arange(len(spot_positions)).reshape(-1, 1)]
    npos = len(spot_positions)
    if (max_n == 1) or (npos < 2) or (npos >= max_total_combinations):
        return list(all_groups[0])

    close_enough = (
        np.linalg.norm(spot_positions[:, None, :] - spot_positions[None, :, :], axis=2)
        < distance_threshold
    )
    # keep the upper triangle
    close_enough = np.triu(close_enough, k=+1)
    pairs = np.array([(a, b) for a, b in zip(*np.where(close_enough))])
    all_groups.append(pairs)
    ntot = np.sum([len(o) for o in all_groups])
    if (max_n == 2) or (len(pairs) == 0) or (ntot >= max_total_combinations):
        out = []
        for gp in all_groups:
            out.extend(gp)
        return out

    g0 = all_groups[0].reshape(-1)
    for n_in_group in range(3, max_n + 1):
        g1 = all_groups[-1]
        if verbose:
            print(f"{len(g1)} combinations of {n_in_group-1} spots")

        valid = np.ones((len(g1), len(g0)), dtype=bool)
        for dim in range(g1.shape[1]):
            valid &= close_enough[g1[:, dim], :]
            valid &= g1[:, dim, None] != g0[None, :]
        if not valid.any():
            # no more valid combinations
            break
        all_groups.append(
            np.vstack([np.hstack([g1[i], g0[j]]) for i, j in zip(*np.where(valid))])
        )
        ntot += len(all_groups[-1])
        if ntot >= max_total_combinations:
            if verbose:
                print(f"Stopped at {ntot} combinations")
            break
        assert all_groups[-1].shape[1] == n_in_group

    out = []
    for gp in all_groups:
        out.extend(gp)
    return out


def likelihood_change_background_combination(
    spot_ids,
    mask_assignments,
    mask_counts,
    log_dist_likelihood,
    log_background_spot_prior,
    p,
    m,
):
    """Likelihood change for a combination of spots to all become background spots.

    Args:
        spot_ids (np.array): 1D array with the spot IDs.
        mask_assignments (np.array): 1D array with the current mask assignments.
        mask_counts (np.array): 1D array with the current mask counts.
        log_dist_likelihood (np.array): N spots x M masks array of distance likelihoods.
        log_background_spot_prior (float): Log background spot prior.
        p (float): Power of the spot count prior.
        m (float): Length scale of the spot count prior.

    Returns:
        float: Likelihood change for this spot combination to become a background spots.
    """
    # new likelihood is new spot to the background
    new_likelihood = log_background_spot_prior * len(spot_ids)
    # and new spot count prior for spots that were assigned
    was_assigned = mask_assignments[spot_ids] != -1
    changed_mask, changed_n = unique_counts(mask_assignments[spot_ids][was_assigned])
    new_counts = mask_counts.copy()
    new_counts[changed_mask] -= changed_n
    new_likelihood += _spot_count_prior(new_counts[changed_mask], p, m).sum()

    # old likelihood is old spot to the background * log_bg
    old_likelihood = log_background_spot_prior * (~was_assigned).sum()
    # + old spot count prior for spots that were assigned
    old_likelihood += _spot_count_prior(mask_counts[changed_mask], p, m).sum()
    # + old distance likelihood for spots that were assigned
    old_likelihood += log_dist_likelihood[
        spot_ids[was_assigned], mask_assignments[spot_ids][was_assigned]
    ].sum()
    return new_likelihood - old_likelihood


def likelihood_change_move_combination(
    spot_ids,
    target_masks,
    mask_assignments,
    mask_counts,
    log_dist_likelihood,
    log_background_spot_prior,
    p,
    m,
):
    """Likelihood change for a combination of spots to move to a new mask.

    Args:
        spot_ids (np.array): 1D array with the spot IDs.
        target_masks (np.array): 1D array with the target mask IDs.
        mask_assignments (np.array): 1D array with the current mask assignments.
        mask_counts (np.array): 1D array with the current mask counts.
        log_dist_likelihood (np.array): N spots x M masks array of distance likelihoods.
        log_background_spot_prior (float): Log background spot prior.
        p (float): Power of the spot count prior.
        m (float): Length scale of the spot count prior.

    Returns:
        np.array: 1D array with the likelihood change for each target."""
    spot_ids = np.asarray(spot_ids)
    target_masks = np.asarray(target_masks)
    if np.any(target_masks < 0):
        raise ValueError("Target mask must be >= 0")
    if len(np.unique(spot_ids)) < len(spot_ids):
        raise ValueError("Spot IDs must be unique")

    # New likelihood depends on the target mask
    new_counts = make_new_counts(target_masks, mask_counts, nspots=len(spot_ids))
    bg = mask_assignments[spot_ids] == -1
    changed_mask, changed_n = unique_counts(mask_assignments[spot_ids][~bg])
    new_counts[:, changed_mask] -= changed_n
    new_likelihood = _spot_count_prior(
        new_counts[np.arange(len(target_masks)), target_masks], p, m
    )
    # if changed_mask is not target, we need to add to the new likelihood
    masks2check = repeat_masks(target_masks, changed_mask)
    already_done = masks2check == target_masks[:, None]
    new_likelihood += _spot_count_prior(
        new_counts[:, changed_mask] * (~already_done), p, m
    ).sum(axis=1)
    # distance lieklihood between spots and target masks
    ids, targets = meshgrid_ij(spot_ids, target_masks)
    new_likelihood += log_dist_likelihood[ids, targets].sum(axis=0)
    # this includes also cases where the mask did not actually change

    old_likelihood = _spot_count_prior(mask_counts[target_masks], p, m)
    old_likelihood += log_dist_likelihood[
        spot_ids[~bg], mask_assignments[spot_ids[~bg]]
    ].sum()
    old_likelihood += log_background_spot_prior * bg.sum()
    # add spot count for the source mask
    old_likelihood += _spot_count_prior(
        mask_counts[changed_mask] * (~already_done), p, m
    ).sum(axis=1)

    return new_likelihood - old_likelihood


@njit
def _spot_count_prior(nspots: int, p: float, m: float):
    """Compute the prior for the number of spots in a mask.

    Args:
        nspots (int): Number of spots in the mask.
        p (float): Power of the spot count prior.
        m (float): Lambda of the spot count prior.

    Returns:
        float: Prior for the number of spots in the mask.

    """
    return -(nspots**p) / m


@njit
def unique_counts(arr):
    arr = np.ascontiguousarray(arr)
    arr = np.sort(arr)
    mask = np.empty(len(arr) + 1, dtype=np.bool_)
    mask[0] = True
    mask[1:-1] = arr[1:] != arr[:-1]
    mask[-1] = True
    unique = arr[mask[:-1]]
    counts = np.empty(unique.shape, dtype=np.int64)
    counts = np.diff(np.where(mask)[0])
    return unique, counts


@njit
# make numba compatible version of meshgrid for 2d arrays with ij indexing
def meshgrid_ij(x, y):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    xx = np.empty((len(x), len(y)), dtype=x.dtype)
    yy = np.empty((len(x), len(y)), dtype=y.dtype)
    for i in range(len(x)):
        xx[i] = x[i]
    for j in range(len(y)):
        yy[:, j] = y[j]
    return xx, yy


@njit
def make_new_counts(target_masks, mask_counts, nspots):
    new_counts = np.empty((len(target_masks), len(mask_counts)), dtype=np.int64)
    for i, mask in enumerate(target_masks):
        new_counts[i] = mask_counts.copy()
        new_counts[i][mask] += nspots
    return new_counts


@njit
def repeat_masks(target_masks, changed_mask):
    masks2check = np.empty((len(target_masks), len(changed_mask)), dtype=np.int64)
    for i, mask in enumerate(changed_mask):
        masks2check[:, i] = mask
    return masks2check
