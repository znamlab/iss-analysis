import numpy as np
from tqdm import tqdm
import pandas as pd
from functools import partial
from multiprocessing import Pool
from warnings import warn
from numba import njit
from scipy.special import digamma

# Sometimes the likelihoods are not exactly 0 but 10-15 or so, so we need to use a
# small number to consider them zero
EPSILON = 1e-6  # small number that is considered zero


def assign_barcodes_to_masks(
    spots,
    masks,
    method="spot_by_spot",
    parameters=None,
    verbose=1,
    base_column="bases",
    n_workers=1,
):
    """Assign barcodes to masks using a probabilistic model.

    Args:
        spots (pd.DataFrame): DataFrame with the spots. Must contain the columns 'x',
            'y', and `base_column`.
        mask (pd.DataFrame): DataFrame with the mask centres. Must contain the columns
            'x' and 'y'.
        method (str): Method to use for the assignment. Options are 'variational_gmm'
            and 'spot_by_spot'. Default is 'spot_by_spot'.
        p (float): Power of the spot count prior. Default is 0.9.
        m (float): Length scale of the spot count prior. Default is 0.1.
        background_spot_prior (float): Prior for the background spots. Default is 0.0001
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
        n_workers (int): Number of workers to use. 1 = no parallel processing Default is
            1.

    Returns:
        np.ndarray: 1D array with the mask assignment if debug is False. Otherwise 2D
            array with the mask assignment for each iteration.
    """
    if method == "variational_gmm":
        func = assign_single_barcode_variational_gmm
    elif method == "spot_by_spot":
        func = assign_single_barcode
    else:
        raise ValueError("Method must be 'variational_gmm' or 'spot_by_spot'")
    if parameters is None or parameters == []:
        parameters = dict()
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
    parameters.update(dict(verbose=max(0, verbose - 1)))

    _assign_by_barcode = partial(
        func,
        mask_positions=mask_centers,
        **parameters,
    )
    chunk_size = max(1, len(barcodes) // 10 // n_workers)
    if n_workers > 1:
        print(f"Starting parpool with {n_workers} workers", flush=True)

        with Pool(n_workers) as pool:
            assignment_by_bc = list(
                tqdm(
                    pool.imap(
                        _assign_by_barcode, spot_positions_per_bc, chunksize=chunk_size
                    ),
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


def assign_single_barcode_variational_gmm(
    spot_positions,
    mask_positions,
    alpha_background=None,
    alpha_cells=0.15,
    log_background_density=-11,
    max_iter=1000,
    tol=1e-4,
    sigma=50,
    max_distance_to_mask=300,
    verbose=True,
):
    """
    Assigns spots to cells using a variational inference algorithm for a Bayesian
    Gaussian mixture model.

    The model assumes that the spots are generated from a Gaussian mixture with
    components centered at the cell positions and a background component with uniform
    density. The prior over mixing coefficients follows a Direchlet distribution with
    the specified concentration parameters. The algorithm uses a variational inference
    algorithm to estimate the posterior distribution of the component assignments.

    The algorithm terminates when the change in the number of spots assigned to the
    background component is less than the specified tolerance or the maximum number of
    iterations is reached.

    Args:
        spot_positions (np.ndarray): Array of shape (n_spots, 2) containing the
            positions of the spots.
        cell_positions (np.ndarray): Array of shape (n_cells, 2) containing the
            positions of the cells.
        alpha_background (float): Concentration parameter of the Direchlet prior on the
            mixing coefficient for the background component. If None, it is set to the
            sum of the concentration parameters for the cell components.
        alpha_cells (float): Concentration parameter of the Direchlet prior on the
            mixing coefficients for the cell components. Default is 0.15.
        log_background_density (float): Log of the density of the background component.
            Default is -11.
        max_iter (int): Maximum number of iterations for the variational inference
            algorithm. Default is 1000.
        tol (float): Tolerance for convergence of the variational inference algorithm.
            Default is 1e-4.
        sigma (float): Standard deviation of the Gaussian components. Default is 50.
        max_distance_to_mask (float): Maximum distance to a mask for a cell to be
            considered for assignment. Default is 300.
        verbose (bool): Whether to print progress. Default is True.

    Returns:
        np.ndarray: Array of shape (n_spots,) containing the indices of the cells to
            which the spots are assigned.

    """
    # Concentration parameters on the Direchlet prior on mixing coefficients - one value
    # for background and cells
    num_cells = mask_positions.shape[0]
    mean_spot_position = spot_positions.mean(axis=0)
    precision_cells = 1 / sigma**2
    precisions = np.concatenate(([1], np.ones(num_cells) * precision_cells))
    mus = np.concatenate(([mean_spot_position], mask_positions))
    distances = np.linalg.norm(spot_positions[:, None] - mus[None], axis=2)
    idx = distances.min(axis=0) < max_distance_to_mask
    idx[0] = True  # always keep the background
    alpha_cells = np.ones(np.sum(idx) - 1) * alpha_cells
    if alpha_background is None or (alpha_background == []):
        alpha_background = np.sum(alpha_cells)
    alpha_0 = np.concatenate(([alpha_background], alpha_cells))
    distances = distances[:, idx]
    precisions = precisions[idx]
    u = -0.5 * (distances**2) * precisions[None]
    # background density is uniform
    u[:, 0] = log_background_density
    N = np.zeros_like(alpha_0)
    for iter in range(max_iter):
        old_N0 = N[0]
        # calculate E[log pi] - equivalent to the M step
        alphas = alpha_0 + N
        # calculate responsibilities - E step
        E_log_pi = digamma(alphas) - digamma(np.sum(alphas))
        rhos = np.exp(u + E_log_pi[None] + 0.5 * np.log(precisions[None]))
        rs = rhos / np.sum(rhos, axis=1)[:, None]
        # update N
        N = rs.sum(axis=0)
        if np.abs(N[0] - old_N0) < tol:
            break
    assignment = rs.argmax(axis=1)
    cells_used = np.nonzero(idx)[0]
    # Put the background to -1
    return cells_used[assignment] - 1


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
    run_by_groupsize=False,
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
        run_by_groupsize (bool): Whether to run the assignment by group size. This will
            first iteraton on combination of `max_total_combinations` spots only, then
            `max_total_combinations - 1` spots etc... Faster but might not be optimal.
            Default False.
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
    log_dist_likelihood = (-0.5 * (distances / spot_distribution_sigma) ** 2).ravel()
    # going via flexilims will replace None by []
    if mask_assignments is None or (not len(mask_assignments)):
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
        verbose=verbose,
    )
    if verbose > 1:
        print(f"Found {len(combinations)} valid spot combinations")
    n_per_combi = np.array([len(combi) for combi in combinations])
    combi_size_borders = [0, *np.where(np.diff(n_per_combi))[0] + 1, len(n_per_combi)]
    combi_sizes = np.sort(np.unique(n_per_combi))[::-1]
    if run_by_groupsize:
        combi_by_size = [
            combinations[combi_size_borders[c_size - 1] : combi_size_borders[c_size]]
            for c_size in combi_sizes
        ]
    else:
        combi_by_size = [combinations]
    new_assignments = mask_assignments.copy()
    if debug:
        all_assignments = [new_assignments.copy()]
    verb_inside = verbose - 1 if verbose else 0
    for i in range(max_iterations):
        if verbose > 0:
            print(f"---- Iteration {i} ----")
        if i > 30:
            print("Too many iterations")

        spot_moved = []
        for c_size, c_combi in enumerate(combi_by_size):
            if run_by_groupsize and (verbose > 1):
                print(f"Combination size {c_size}")
            new_assignments, move_this_size = assign_single_barcode_single_round(
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
                combinations=c_combi,
                verbose=verb_inside,
            )
            if move_this_size:
                spot_moved.extend(move_this_size)

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
    if verbose > 0:
        print(f"Assigned {np.sum(nbg)} spots to masks")
    return assignments


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
        log_dist_likelihood (np.array, optional): 1D array of len = NxM array of log
            likelihoods of the distances. Default None.
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
            f"\nAssigning {len(spot_positions)} spots to {len(mask_positions)} masks",
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

    if distances is None:
        distances = np.linalg.norm(
            spot_positions[:, None, :] - mask_positions[None, :, :], axis=2
        )
    if log_dist_likelihood is None:
        log_dist_likelihood = (
            -0.5 * (distances / spot_distribution_sigma) ** 2
        ).ravel()

    if len(combinations) == 0:
        return None

    # compute the likelihood of each combination
    likelihood_changes, best_targets = np.zeros((2, len(combinations)), dtype=float)
    for i_comb, combi in enumerate(combinations):
        (
            likelihood_changes[i_comb],
            best_targets[i_comb],
        ) = likelihood_change_single_combi(
            combi,
            mask_assignments,
            mask_counts,
            distances,
            log_dist_likelihood,
            log_background_spot_prior,
            p,
            m,
            max_distance_to_mask,
        )

    # sort by likelihood change
    order = np.argsort(likelihood_changes)[::-1]
    mask_changed = set()
    # mask_targeted = set()
    spot_moved = []
    new_assignments = mask_assignments.copy()
    for i_combi in order:
        if likelihood_changes[i_combi] <= EPSILON:
            # we reached the point where we make things worse
            break
        target = best_targets[i_combi]
        # we should not add spots to a mask where we removed some
        if target in mask_changed:
            continue
        spot_combi = combinations[i_combi]
        source_masks = mask_assignments[spot_combi]
        # we should not take spots from a mask where we added some
        if any([m in mask_changed for m in source_masks]):
            continue
        new_assignments[spot_combi] = target
        spot_moved.append(spot_combi)
        if target != -1:
            mask_changed.add(target)
        mask_changed.update(source_masks[source_masks != -1])

    return new_assignments, spot_moved


@njit
def likelihood_change_single_combi(
    combi,
    mask_assignments,
    mask_counts,
    distances,
    log_dist_likelihood,
    log_background_spot_prior,
    p,
    m,
    max_distance_to_mask,
):
    bg_likelihood = _likelihood_change_background_combination(
        combi,
        mask_assignments,
        mask_counts,
        log_dist_likelihood,
        log_background_spot_prior,
        p,
        m,
    )

    likelihood_change = bg_likelihood
    best_target = -1
    too_far = _max_along_0(distances[combi])

    valid_targets = np.where(too_far < max_distance_to_mask)[0]
    current_assignments = np.unique(mask_assignments[combi])
    current_assignments = current_assignments[current_assignments >= 0]
    # don't move to a mask that is already assigned
    unvalid = np.searchsorted(valid_targets, current_assignments)
    valid_targets = np.delete(valid_targets, unvalid)

    if len(valid_targets) == 0:
        return likelihood_change, best_target
    else:
        move_likelihood = _likelihood_change_move_combination(
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
    id_max_change = np.argmax(move_likelihood)
    if move_likelihood[id_max_change] > bg_likelihood:
        likelihood_change = move_likelihood[id_max_change]
        best_target = valid_targets[id_max_change]
    return likelihood_change, best_target


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
        log_dist_likelihood = (
            -0.5 * (distances / spot_distribution_sigma) ** 2
        ).ravel()

    spot_moved = []
    for current_mask in np.unique(mask_assignments[mask_assignments >= 0]):
        combi = np.where(mask_assignments == current_mask)[0]
        bg_likelihood = _likelihood_change_background_combination(
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
        if npos >= max_total_combinations:
            print("!!! Too many spots, runing only spot by spot !!!")
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
        if npos >= max_total_combinations:
            print("!!! Too many combination, runing by pairs !!!")
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
            if (npos >= max_total_combinations) and (n_in_group < max_n):
                print(f"!!! Too many combination, max group size: {n_in_group} !!!")
            if verbose:
                print(f"Stopped at {ntot} combinations")
            break
        assert all_groups[-1].shape[1] == n_in_group

    out = []
    for gp in all_groups:
        out.extend(gp)
    return out


@njit
def _likelihood_change_background_combination(
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
        log_dist_likelihood (np.array): 1D array of len N spots x M masks array of
            distance likelihoods.
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
    changed_mask, changed_n = _unique_counts(mask_assignments[spot_ids][was_assigned])
    new_counts = mask_counts.copy()
    new_counts[changed_mask] -= changed_n
    new_likelihood += _spot_count_prior(new_counts[changed_mask], p, m).sum()

    # old likelihood is old spot to the background * log_bg
    old_likelihood = log_background_spot_prior * (~was_assigned).sum()
    # + old spot count prior for spots that were assigned
    old_likelihood += _spot_count_prior(mask_counts[changed_mask], p, m).sum()
    # + old distance likelihood for spots that were assigned
    old_likelihood += log_dist_likelihood[
        _ravel_2d_index(
            spot_ids[was_assigned],
            mask_assignments[spot_ids][was_assigned],
            len(mask_counts),
        )
    ].sum()

    return new_likelihood - old_likelihood


@njit
def _likelihood_change_move_combination(
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
    new_counts = _make_new_counts(target_masks, mask_counts, nspots=len(spot_ids))
    bg = mask_assignments[spot_ids] == -1
    changed_mask, changed_n = _unique_counts(mask_assignments[spot_ids][~bg])
    new_counts[:, changed_mask] -= changed_n
    ravel_cnt = new_counts.reshape(-1)[
        _ravel_2d_index(np.arange(len(target_masks)), target_masks, len(mask_counts))
    ]
    new_likelihood = _spot_count_prior(ravel_cnt, p, m)

    # if changed_mask is not target, we need to add to the new likelihood
    masks2check = _repeat_masks(target_masks, changed_mask)
    already_done = masks2check == target_masks[:, None]
    new_likelihood += _spot_count_prior(
        new_counts[:, changed_mask] * (~already_done), p, m
    ).sum(axis=1)
    # distance likelihood between spots and target masks
    # tow ooption here
    # option1
    # ids, targets = _meshgrid_ij(spot_ids, target_masks)
    # new_likelihood += log_dist_likelihood[ids, targets].sum(axis=0)

    # option2
    for s_id in spot_ids:
        new_likelihood += log_dist_likelihood[
            _ravel_2d_index(s_id, target_masks, len(mask_counts))
        ]

    # this includes also cases where the mask did not actually change

    old_likelihood = _spot_count_prior(mask_counts[target_masks], p, m)
    old_likelihood += log_dist_likelihood[
        _ravel_2d_index(
            spot_ids[~bg], mask_assignments[spot_ids[~bg]], len(mask_counts)
        )
    ].sum()
    old_likelihood += log_background_spot_prior * bg.sum()
    # add spot count for the source mask
    old_likelihood += _spot_count_prior(
        mask_counts[changed_mask] * (~already_done), p, m
    ).sum(axis=1)

    return new_likelihood - old_likelihood


@njit
def _ravel_2d_index(i, j, n):
    """Convert 2D index to 1D index.

    Args:
        i (int): Row index.
        j (int): Column index.
        n (int): Number of columns.

    Returns:
        int: 1D index.
    """
    return i * n + j


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
def _unique_counts(arr):
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
def _meshgrid_ij(x, y):
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
def _make_new_counts(target_masks, mask_counts, nspots):
    new_counts = np.empty((len(target_masks), len(mask_counts)), dtype=np.int64)
    for i, mask in enumerate(target_masks):
        new_counts[i] = mask_counts.copy()
        new_counts[i][mask] += nspots
    return new_counts


@njit
def _repeat_masks(target_masks, changed_mask):
    masks2check = np.empty((len(target_masks), len(changed_mask)), dtype=np.int64)
    for i, mask in enumerate(changed_mask):
        masks2check[:, i] = mask
    return masks2check


@njit
def _max_along_0(arr):
    # def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
        result[i] = np.max(arr[:, i])
    return result
