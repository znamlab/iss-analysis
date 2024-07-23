import numpy as np
import itertools
from numba import njit, prange


def assign_barcodes_to_masks(
    spots,
    masks,
    p=0.9,
    m=0.1,
    background_spot_prior=0.0001,
    spot_distribution_sigma=50,
    max_iterations=100,
    distance_threshold=200,
    verbose=False,
    base_column="bases",
    debug=False,
    seed=123,
    number_of_random_initializations=10,
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
        distance_threshold (float): Threshold for the distance in pixels between spots
            and masks. Default is 200.
        verbose (bool): Whether to print the progress. Default is False.
        base_column (str): Name of the column with the bases. Default is 'bases'.
        debug (bool): Whether to return debug information. Default is False.
        seed (int): Seed for the random number generator. Default is 123.
        number_of_random_initializations (int): Number of random initializations to try
            after the initial "assign to closest". Default is 10.

    Returns:
        np.ndarray: 1D array with the mask assignment if debug is False. Otherwise 2D
            array with the mask assignment for each iteration.
    """
    np.random.seed(seed)

    # compute the distance between each spot and each mask
    mask_centers = masks[["x", "y"]].values
    spot_positions = spots[["x", "y"]].values
    distances = np.linalg.norm(
        spot_positions[:, None, :] - mask_centers[None, :, :], axis=2
    )
    # we will use only spots that are close to at least one mask.
    spots_in_range = np.any(distances < distance_threshold, axis=1)
    spots = spots[spots_in_range].copy().reset_index(drop=True)
    distances = distances[spots_in_range]

    if verbose:
        print(f"Using {spots_in_range.sum()}/{len(spots_in_range)} spots in range")
        print(f"Assigning {len(spots)} spots to {len(mask_centers)} masks")

    log_background_spot_prior = np.log(background_spot_prior)
    # compute the probability of the spot being in each mask given the distance
    log_spot_distribution = -0.5 * (distances / spot_distribution_sigma) ** 2
    barcodes = spots[base_column].unique().astype(str)
    spots_barcodes = spots[base_column].values.astype(str)
    if verbose:
        print(f"Found {len(barcodes)} unique barcodes")

    starting_conditions = []
    # first assign each spot to its nearest mask
    starting_conditions.append(np.argmax(log_spot_distribution, axis=1))

    for _ in range(number_of_random_initializations):
        # then assign all spots to nearby cells randomly depending on the distance
        assignment = np.zeros(len(spots), dtype=int)
        for sp in range(log_spot_distribution.shape[0]):
            dist = distances[sp]
            # keep only the 3 closest
            possible_masks = np.argsort(dist)[:3]
            # if they are closer than the threshold
            possible_masks = possible_masks[dist[possible_masks] < distance_threshold]
            if not possible_masks.size:
                assignment[sp] = -1
                continue
            prob = 1 / dist[possible_masks] ** 4
            prob /= prob.sum()
            assignment[sp] = np.random.choice(possible_masks, p=prob)
        starting_conditions.append(assignment)

    all_assignments = []
    all_log_likelihoods = []
    for mask_assignment in starting_conditions:
        mask_assignment = _assign_barcodes_to_masks(
            initial_mask_assignment=mask_assignment,
            barcodes=barcodes,
            spots_barcodes=spots_barcodes,
            spots_index=spots.index,
            mask_centers=mask_centers,
            distances=distances,
            log_background_spot_prior=log_background_spot_prior,
            p=p,
            m=m,
            log_spot_distribution=log_spot_distribution,
            max_iterations=max_iterations,
            distance_threshold=distance_threshold,
            verbose=verbose,
            debug=debug,
        )
        all_assignments.append(mask_assignment)
        # compute the likelihood of the assignment
        # for background spots it's easy
        log_likelihood = log_background_spot_prior * (mask_assignment == -1).sum()
        # we need to add the distance likelihood for each spot
        distance_likelihood = log_spot_distribution[
            np.arange(len(mask_assignment)), mask_assignment
        ]
        distance_likelihood[mask_assignment == -1] = 0
        log_likelihood += distance_likelihood.sum()
        # and finally the spot count prior, we need to split by barcode here
        for barcode in barcodes:
            spot_is_this_barcode = spots_barcodes == barcode
            this_barcode = mask_assignment[spot_is_this_barcode]
            mask_counts = np.bincount(
                this_barcode[this_barcode >= 0], minlength=len(mask_centers)
            )
            for mask_count in mask_counts[mask_counts > 0]:
                log_likelihood += _spot_count_prior(mask_count, p, m)
        all_log_likelihoods.append(log_likelihood)

    for i_assign in range(len(all_assignments)):
        # recreate a full assignment with -2 for spots that are not assigned
        mask_assignment = all_assignments[i_assign]
        if debug:
            assignment = []
            for m in range(len(mask_assignment)):
                assignment.append(
                    _recreate_full_assignment(mask_assignment[m], spots_in_range)
                )
            assignment = np.vstack(mask_assignment)
            all_assignments[i_assign] = assignment
        else:
            all_assignments[i_assign] = _recreate_full_assignment(
                mask_assignment, spots_in_range
            )
    return all_assignments, all_log_likelihoods


def valid_spot_combination(
    spot_positions: np.array, distance_threshold: float, max_n: int = 5
):
    """Combinations of spots that are all within distance_threshold of each other

    Args:
        spot_positions (np.array): Nx2 array of spot positions
        distance_threshold (float): maximum distance between spots
        max_n (int): maximum number of spots in the combination

    Returns:
        list of np.array: list of combinations of spots that are all within
            distance_threshold of each other
    """
    all_groups = []
    spot_distances = np.linalg.norm(
        spot_positions[:, None, :] - spot_positions[None, :, :], axis=2
    )
    # we won't have combination longer than the number of spots
    max_n = min(max_n, len(spot_positions))
    for n_in_group in range(1, max_n + 1):
        spots = np.array(
            list(itertools.combinations(np.arange(len(spot_positions)), n_in_group))
        )
        dim_pairs = np.array(list(itertools.combinations(np.arange(n_in_group), 2)))
        valid_spots = np.ones(len(spots), dtype=bool)
        for dim_pair in dim_pairs:
            valid_spots &= (
                spot_distances[spots[:, dim_pair[0]], spots[:, dim_pair[1]]]
                < distance_threshold
            )
        all_groups.extend(spots[valid_spots])
    return all_groups


def likelihood_change_background_combination(
    spot_ids,
    mask_assignments,
    mask_counts,
    log_dist_likelihood,
    log_background_spot_prior,
    p,
    m,
):
    """Compute the likelihood change for a combination of spots to all become background spots.

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
    changed_mask, changed_n = np.unique(
        mask_assignments[spot_ids][was_assigned], return_counts=True
    )
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


def _assign_barcodes_to_masks(
    initial_mask_assignment,
    barcodes,
    spots_barcodes,
    spots_index,
    mask_centers,
    distances,
    log_background_spot_prior,
    p,
    m,
    log_spot_distribution,
    max_iterations,
    distance_threshold,
    verbose,
    debug,
):
    mask_assignment = initial_mask_assignment
    if debug:
        output = [mask_assignment.copy()]
    for iter in range(max_iterations):
        spots_moved = 0
        for barcode in barcodes:
            # count the number of spots assigned to each mask
            spot_is_this_barcode = spots_barcodes == barcode
            this_barcode = mask_assignment[spot_is_this_barcode]
            mask_counts = np.bincount(
                this_barcode[this_barcode >= 0], minlength=len(mask_centers)
            )
            for spot_index in spots_index[spot_is_this_barcode]:
                mask_assignment, mask_counts, spots_moved = _reassign_spot(
                    spot_index=spot_index,
                    mask_assignment=mask_assignment,
                    mask_counts=mask_counts,
                    distances=distances,
                    distance_threshold=distance_threshold,
                    log_background_spot_prior=log_background_spot_prior,
                    p=p,
                    m=m,
                    log_spot_distribution=log_spot_distribution,
                    spots_moved=spots_moved,
                )
        if verbose:
            print(f"Iteration {iter}: {spots_moved} spots resassigned")
        if debug:
            output.append(mask_assignment.copy())
        if spots_moved == 0:
            # try to move all spots of each mask to the background
            mask_assignment, spots_moved = _move_all_spots_to_background(
                barcodes,
                mask_assignment,
                spots_barcodes,
                mask_centers,
                log_background_spot_prior,
                p,
                m,
                log_spot_distribution,
            )
            if debug:
                output.append(mask_assignment.copy())
            if spots_moved == 0:
                break
    if debug:
        return output
    return mask_assignment


def _recreate_full_assignment(mask_assignment, spots_in_range):
    full_assignment = np.full(len(spots_in_range), -2)
    full_assignment[spots_in_range] = mask_assignment
    return full_assignment


def _move_all_spots_to_background(
    barcodes: np.array,
    mask_assignment: np.array,
    barcode_list: np.array,
    mask_centers: np.array,
    log_background_spot_prior: float,
    p: float,
    m: float,
    log_spot_distribution: np.array,
):
    """Move all spots of each mask to the background if it increases the likelihood.

    Args:
        barcodes (np.array): Array of unique barcodes.
        mask_assignment (np.array): 1D array with the mask assignment.
        barcode_list (np.array): 1D array with the barcode list.
        mask_centers (np.array): 2D array with the mask centers.
        log_background_spot_prior (float): Log background spot prior.
        p (float): Power of the spot count prior.
        m (float): Length scale of the spot count prior.
        log_spot_distribution (np.array): N spots x M masks array of distance
            likelihoods.

    Returns:
        np.array: 1D array with the mask assignment.
        int: Number of spots moved.
    """
    spots_moved = 0
    for barcode in barcodes:
        # count the number of spots assigned to each mask
        valid_spots = barcode_list == barcode
        this_barcode = mask_assignment[valid_spots]
        mask_counts = np.bincount(
            this_barcode[this_barcode >= 0], minlength=len(mask_centers)
        )
        for current_mask in range(len(mask_centers)):
            if mask_counts[current_mask] > 0:
                # likelihood change if all spots in the mask are background spots
                log_likelihood_change_background = (
                    log_background_spot_prior * mask_counts[current_mask]
                    - _spot_count_prior(mask_counts[current_mask], p, m)
                    - log_spot_distribution[
                        (mask_assignment == current_mask) & valid_spots,
                        current_mask,
                    ].sum()
                )
                if log_likelihood_change_background > 0:
                    mask_assignment[
                        (mask_assignment == current_mask) & valid_spots
                    ] = -1
                    spots_moved += mask_counts[current_mask]
                    mask_counts[current_mask] = 0
    return mask_assignment, spots_moved


@njit
def _reassign_spot(
    spot_index: int,
    mask_assignment: np.array,
    mask_counts: np.array,
    distances: np.array,
    distance_threshold: float,
    log_background_spot_prior: float,
    p: float,
    m: float,
    log_spot_distribution: np.array,
    spots_moved: int,
):
    current_mask = mask_assignment[spot_index]
    mask_is_closeby = distances[spot_index] < distance_threshold
    masks_to_check = np.where(mask_is_closeby)[0]
    # compute likelihood for switching to a different mask
    log_likelihood_change = _likelihood_change_move_spot(
        current_spot=spot_index,
        current_mask=current_mask,
        masks_to_check=masks_to_check,
        log_background_spot_prior=log_background_spot_prior,
        p=p,
        m=m,
        mask_counts=mask_counts,
        log_spot_distribution=log_spot_distribution,
    )
    # compute likelihood for switching to a background spot
    log_likelihood_change_background = _likelihood_change_background(
        current_mask=current_mask,
        current_counts=mask_counts[current_mask],
        current_distance_likelihood=log_spot_distribution[spot_index, current_mask],
        log_background_spot_prior=log_background_spot_prior,
        p=p,
        m=m,
    )
    # if max of log_likelihood_change is higher than log_likelihood_change_background
    # then assign the spot to the mask that gives the highest increase in likelihood
    if np.max(log_likelihood_change) > log_likelihood_change_background:
        new_mask_assignment = np.argmax(log_likelihood_change)
        if new_mask_assignment != current_mask:
            mask_assignment[spot_index] = new_mask_assignment
            if current_mask != -1:
                assert mask_counts[current_mask] > 0
                mask_counts[current_mask] -= 1
            mask_counts[new_mask_assignment] += 1
            spots_moved += 1
    elif current_mask != -1:
        mask_assignment[spot_index] = -1
        assert mask_counts[current_mask] > 0
        mask_counts[current_mask] -= 1
        spots_moved += 1
    return mask_assignment, mask_counts, spots_moved


@njit(parallel=True)
def _likelihood_change_move_spot(
    current_spot: int,
    current_mask: int,
    masks_to_check: np.array,
    log_background_spot_prior: float,
    p: float,
    m: float,
    mask_counts: np.array,
    log_spot_distribution: np.array,
):
    """Compute the likelihood change for a spot to move to new masks.

    Args:
        current_spot (int): The current spot.
        current_mask (int): The current mask.
        masks_to_check (np.array): 1-D array of m < M mask IDs to check.
        log_background_spot_prior (float): The log background spot prior.
        p (float): Power of the spot count prior.
        m (float): Length scale of the spot count prior.
        mask_counts (np.array): 1-D array of the M mask counts for all Mmasks
        log_spot_distribution (np.array): N spots x M masks array of distance
            likelihoods.

    Returns:
        np.array: The likelihood change for the spot to move to a new mask.
    """
    log_likelihood_change = np.zeros((len(mask_counts)), dtype=float) - np.inf
    for i_mask in prange(len(masks_to_check)):
        new_mask = masks_to_check[i_mask]
        current_likelihood, new_likelihood = _calc_log_likelihoods(
            current_mask,
            new_mask,
            log_background_spot_prior,
            p,
            m,
            mask_counts,
            current_spot,
            log_spot_distribution,
        )
        log_likelihood_change[new_mask] = new_likelihood - current_likelihood
    return log_likelihood_change


@njit
def _calc_log_likelihoods(
    current_mask: int,
    new_mask: int,
    log_background_spot_prior: float,
    p: float,
    m: float,
    mask_counts: np.ndarray,
    spot_index: int,
    log_spot_distribution: np.ndarray,
):
    """Inner function of assign_barcodes_to_masks.

    Calculate the likelihoods for the current and new mask.

    Args:
        current_mask (int): The current mask.
        new_mask (int): The new mask.
        log_background_spot_prior (float): The log background spot prior.
        p (float): The power of the spot count prior.
        m (float): The length scale of the spot count prior.
        mask_counts (np.ndarray): The mask counts.
        spot_index (int): The spot index.
        log_spot_distribution (np.ndarray): The log spot distribution.

    Returns:
        tuple: Tuple with the current and new likelihoods.
    """
    if current_mask == -1:
        # changing from a background spot to a mask
        current_likelihood = log_background_spot_prior + _spot_count_prior(
            mask_counts[new_mask], p, m
        )
        new_likelihood = log_spot_distribution[
            spot_index, new_mask
        ] + _spot_count_prior(mask_counts[new_mask] + 1, p, m)
    else:
        current_likelihood = (
            log_spot_distribution[spot_index, current_mask]
            + _spot_count_prior(mask_counts[current_mask], p, m)
            + _spot_count_prior(mask_counts[new_mask], p, m)
        )
        new_likelihood = (
            log_spot_distribution[spot_index, new_mask]
            + _spot_count_prior(mask_counts[current_mask] - 1, p, m)
            + _spot_count_prior(mask_counts[new_mask] + 1, p, m)
        )
    return current_likelihood, new_likelihood


@njit
def _likelihood_change_background(
    current_mask: int,
    current_counts: int,
    current_distance_likelihood: float,
    log_background_spot_prior: float,
    p: float,
    m: float,
):
    """Compute the likelihood change for a spot to become a background spot.

    Args:
        current_mask (int): Current mask assignment.
        current_counts (int): Current rolonie counts for that mask.
        current_distance_likelihood (float): Current distance  likelihood for the spot.
        log_background_spot_prior (float): Log background spot prior.
        p (float): Power of the spot count prior.
        m (float): Length scale of the spot count prior.

    Returns:
        float: Likelihood change for this spot to become a background spot.
    """
    if current_mask == -1:
        log_likelihood_change_background = 0
    else:
        log_likelihood_change_background = (
            log_background_spot_prior
            + _spot_count_prior(current_counts - 1, p, m)
            - _spot_count_prior(current_counts, p, m)
            - current_distance_likelihood
        )
    return log_likelihood_change_background


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
