import pytest
import numpy as np
import pandas as pd
from iss_analysis.barcodes import probabilistic_assignment as pa

# Test the mask assignment functions


def test_assign_barcodes_to_masks():
    spots = pd.DataFrame(
        index=[10, 11, 12], data=dict(x=[0, 0, 0], y=[0, 0, 0], bases=["A", "A", "A"])
    )
    masks = pd.DataFrame(index=[7, 10, 3], data=dict(x=[0, 10, 100], y=[0, 0, 0]))
    parameters = dict(
        p=0.9,
        m=0.1,
        background_spot_prior=0.0001,
        spot_distribution_sigma=50,
        max_iterations=100,
        max_distance_to_mask=200,
        inter_spot_distance_threshold=50,
        max_spot_group_size=5,
        max_total_combinations=1e6,
    )

    assignment = pa.assign_barcodes_to_masks(
        spots,
        masks,
        parameters=parameters,
        verbose=0,
        base_column="bases",
    )
    assert np.all(assignment == 7)
    assert np.all(assignment.index == spots.index)
    parameters["p"] = 0.8
    parameters["m"] = 0.08
    assignment = pa.assign_barcodes_to_masks(
        spots,
        masks,
        parameters=parameters,
        verbose=0,
        base_column="bases",
    )
    assert np.all(assignment == -1)
    assert np.all(assignment.index == spots.index)

    # check that assignment works with and without parallel
    # slightly more complex case, 10 masks, with 1 to 10 spots
    masks = pd.DataFrame(
        index=np.arange(20) * 70 + 1,
        data=dict(x=np.arange(1, 21) * 1000, y=np.arange(1, 21) * 1000),
    )
    nspots = np.arange(1, 11).sum()
    pos = np.vstack([[n + 10, n + 10] for n in np.arange(1, 11) for i in range(n)])
    spots = pd.DataFrame(
        index=np.arange(nspots) * 5 + 10,
        data=dict(x=pos[:, 0] * 1000, y=pos[:, 1] * 1000, bases=["A"] * nspots),
    )
    expected = []
    # the blob with less than 5 spots should be background
    for i in range(5):
        expected.extend([-1] * i)
    # the rest should be assigned to the closest mask
    for i in range(5, 11):
        expected.extend([(i + 9) * 70 + 1] * (i))

    for nw in [1, 5]:
        assignment = pa.assign_barcodes_to_masks(
            spots,
            masks,
            parameters=parameters,
            verbose=0,
            base_column="bases",
            n_workers=nw,
        )
        assert np.all(assignment.index == spots.index)
        assert np.all(assignment == expected)


def test_valid_spot_combination():
    distance_threshold = 10
    # set of spots too far from each other
    spot_positions = np.vstack([(0, 0), (100, 100), (200, 200)])
    valid = pa.valid_spot_combinations(
        spot_positions, distance_threshold, verbose=False
    )
    assert len(valid) == len(spot_positions)

    # set of spots very close to each other
    spot_positions = np.vstack([(0, 0), (1, 1), (2, 2)])
    valid = pa.valid_spot_combinations(
        spot_positions, distance_threshold, verbose=False
    )
    assert len(valid) == 7

    # set of spots very close to each other, with one far away
    spot_positions = np.vstack([(0, 0), (1, 1), (2, 2), (100, 100)])
    valid = pa.valid_spot_combinations(
        spot_positions, distance_threshold, verbose=False
    )
    assert len(valid) == 8

    # spots close to each other by pairs but not triples
    spot_positions = np.vstack([(0, 0), (0, 7), (0, 14)])
    valid = pa.valid_spot_combinations(
        spot_positions, distance_threshold, verbose=False
    )
    assert len(valid) == 5


def test_likelihood_change_background_combination():
    (
        mask_assignments,
        mask_counts,
        log_dist_likelihood,
        log_background_spot_prior,
        p,
        m,
    ) = create_data()
    params = dict(
        mask_assignments=mask_assignments,
        mask_counts=mask_counts,
        log_dist_likelihood=log_dist_likelihood.ravel(),
        log_background_spot_prior=log_background_spot_prior,
        p=p,
        m=m,
    )
    # moving a background spots should not change the likelihood
    spot_ids = np.array([1], dtype=int)
    out = pa._likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, 0)

    # even if it's multiple background spots
    spot_ids = np.array([1, 2], dtype=int)
    out = pa._likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, 0)

    # moving just spot 0 should change the likelihood
    # spot 0 is assigned to mask 0 with a mask_counts of 3
    # the distance likelihood of this spot is 0
    # with p=m=1 the prior is - spot_count
    expected = log_background_spot_prior + (-2) - (-3)
    log_dist_likelihood[0, 0]
    out = pa._likelihood_change_background_combination(
        spot_ids=np.array([0], dtype=int), **params
    )
    assert np.allclose(out, expected)

    # test moving 2 spots from the same mask
    spot_ids = np.array([4, 5], dtype=int)
    # they are both in mask 4 with count 3, so cnt likelihood is -3 before, -1 after
    expected = (
        log_background_spot_prior * 2
        + (-1)
        - (-3 + log_dist_likelihood[spot_ids, 4].sum())
    )
    out = pa._likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, expected)

    # test moving 2 spots from different masks
    spot_ids = np.array([3, 5], dtype=int)
    # they are from mask 2 and 4, with counts 1 and 3 respectively
    expected = (
        log_background_spot_prior * 2
        + (-0 - 2)
        - (-1 + -3 + log_dist_likelihood[spot_ids, mask_assignments[spot_ids]].sum())
    )
    out = pa._likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, expected)

    # †ry to move 0 spots
    spot_ids = np.array([], dtype=int)
    out = pa._likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, 0)

    # try to move all spots
    spot_ids = np.arange(len(mask_assignments))
    expected = log_background_spot_prior * len(mask_assignments)
    bg = mask_assignments == -1
    expected -= log_background_spot_prior * bg.sum()
    expected -= (-mask_counts).sum()
    expected -= log_dist_likelihood[
        spot_ids[~bg], mask_assignments[spot_ids[~bg]]
    ].sum()
    out = pa._likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, expected)


def test_likelihood_change_move_combination():
    (
        mask_assignments,
        mask_counts,
        log_dist_likelihood,
        log_background_spot_prior,
        p,
        m,
    ) = create_data()
    params = dict(
        mask_assignments=mask_assignments,
        mask_counts=mask_counts,
        log_dist_likelihood=log_dist_likelihood.ravel(),
        log_background_spot_prior=log_background_spot_prior,
        p=p,
        m=m,
    )

    # moving a mask to where it is assigned should not change the likelihood
    out = pa._likelihood_change_move_combination(
        spot_ids=np.array([0]), target_masks=np.array([0]), **params
    )
    assert out == 0

    # moving a background spot, spot 1
    # we go to mask 0, which has 2 spots, so the count likelihood after is -3
    # mask 0 means no distance likelihood
    expected1 = -3 - (log_background_spot_prior + (-2))
    out = pa._likelihood_change_move_combination(
        spot_ids=np.array([1]), target_masks=np.array([0]), **params
    )
    assert np.allclose(out, expected1)

    # moving a background spot, spot 1
    # we go to mask 4, which has 3 spots, so the count likelihood after is -4
    # mask 3 means we need to look at distance likelihood
    expected2 = -4 + log_dist_likelihood[1, 4] - (log_background_spot_prior + (-3))
    out = pa._likelihood_change_move_combination(
        spot_ids=np.array([1]), target_masks=np.array([4]), **params
    )
    assert np.allclose(out, expected2)

    # we can look at both targets at once
    out = pa._likelihood_change_move_combination(
        spot_ids=np.array([1]), target_masks=np.array([0, 4]), **params
    )
    assert np.allclose(out, np.array([expected1, expected2]))

    # moving all spots to the same mask
    out = pa._likelihood_change_move_combination(
        spot_ids=np.arange(len(mask_assignments)), target_masks=np.array([3]), **params
    )
    expected = -len(mask_assignments)
    expected += log_dist_likelihood[np.arange(len(mask_assignments)), 3].sum()
    expected -= -mask_counts.sum()
    expected -= log_background_spot_prior * (mask_assignments == -1).sum()
    expected -= log_dist_likelihood[
        np.arange(len(mask_assignments))[mask_assignments != -1],
        mask_assignments[mask_assignments != -1],
    ].sum()

    # trying an illegal move
    with pytest.raises(ValueError):
        out = pa._likelihood_change_move_combination(
            spot_ids=np.array([1, 1]), target_masks=np.array([0]), **params
        )
    with pytest.raises(ValueError):
        out = pa._likelihood_change_move_combination(
            spot_ids=np.array([1]), target_masks=np.array([-1]), **params
        )


def test_unique_count():
    arr = np.array([10, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5, 5, 5, 10])
    val, cnt = pa._unique_counts(arr)
    v, c = np.unique(arr, return_counts=True)
    assert np.array_equal(val, v), f"Value issue {val} != {v}"
    assert np.array_equal(cnt, c), f"Count issue {cnt} != {c}"

    arr = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    val, cnt = pa._unique_counts(arr)
    v, c = np.unique(arr, return_counts=True)
    assert np.array_equal(val, v), f"Value issue {val} != {v}"
    assert np.array_equal(cnt, c), f"Count issue {cnt} != {c}"


def test_assign_single_round():
    spot_positions = np.array(
        [
            [1, 1],
            [0, 1],
            [1, 0],
            [2, 1],
            [1, 2],
            [2, 2],
            [0, 0],
            [100, 100],
            [101, 101],
            [102, 102],
        ]
    )
    mask_positions = np.array([[1, 1], [2, 2], [100, 102], [400, 400]])
    mask_assignments = np.array([0, 1, 0, 2, 1, -1, 1, 3, -1, -1])
    mask_counts = np.bincount(
        mask_assignments[mask_assignments >= 0], minlength=len(mask_positions)
    )
    mask_distance_threshold = 100
    spot_distance_threshold = 10
    log_background_spot_prior = 1
    spot_distribution_sigma = 1
    p = 1
    m = 1
    spot_moved = [1]
    new_assignment = mask_assignments.copy()
    niter = 0
    all_assignments = [new_assignment]
    while len(spot_moved):
        new_assignment, spot_moved = pa.assign_single_barcode_single_round(
            spot_positions=spot_positions,
            mask_positions=mask_positions,
            mask_assignments=new_assignment,
            max_distance_to_mask=mask_distance_threshold,
            inter_spot_distance_threshold=spot_distance_threshold,
            log_background_spot_prior=log_background_spot_prior,
            p=p,
            m=m,
            spot_distribution_sigma=spot_distribution_sigma,
            max_spot_group_size=11,
            verbose=0,
        )
        all_assignments.append(new_assignment)
        niter += 1
        if niter > 10:
            raise ValueError("Too many iterations")
    # with the weird parameters, background is always the best choice
    # (but it might take a few iterations)
    assert np.all(new_assignment == -1)

    # better params and new spots
    p = 0.9
    m = 0.1
    background_spot_prior = 0.0001
    spread = 10 / 0.2
    cov = [[spread, 0], [0, spread]]
    rng = np.random.default_rng(seed=12)
    x, y = rng.multivariate_normal([10, 10], cov, 10).T
    spot_positions = np.vstack([x, y]).T
    mask_positions = np.array([[10, 10], [20, 20], [400, 402], [1000, 1000]])
    mask_assignments = rng.choice([0, 1, 2, 3], len(spot_positions))
    mask_distance_threshold = 600
    spot_distance_threshold = 100
    spot_distribution_sigma = 60
    spot_moved = [1]
    new_assignment, spot_moved = pa.assign_single_barcode_single_round(
        spot_positions=spot_positions,
        mask_positions=mask_positions,
        mask_assignments=mask_assignments,
        max_distance_to_mask=mask_distance_threshold,
        inter_spot_distance_threshold=spot_distance_threshold,
        log_background_spot_prior=np.log(background_spot_prior),
        p=p,
        m=m,
        spot_distribution_sigma=spot_distribution_sigma,
        max_spot_group_size=11,
        verbose=0,
    )
    # now the spots should be assigned to the closest mask
    assert np.all(new_assignment == 0)

    # with lower max_spot_group_size, cannot move all spots at once
    iter = 0
    new_assignment = mask_assignments.copy()
    spot_moved = [1]
    while len(spot_moved):
        iter += 1
        new_assignment, spot_moved = pa.assign_single_barcode_single_round(
            spot_positions=spot_positions,
            mask_positions=mask_positions,
            mask_assignments=new_assignment,
            max_distance_to_mask=mask_distance_threshold,
            inter_spot_distance_threshold=spot_distance_threshold,
            log_background_spot_prior=np.log(background_spot_prior),
            p=p,
            m=m,
            spot_distribution_sigma=spot_distribution_sigma,
            max_spot_group_size=5,
            verbose=0,
        )
        if iter > 10:
            raise ValueError("Too many iterations")
    # with the seed = 12, we have an initial assignment with 2 spots in mask 1 and 1
    # spot in mask 0, which means that it's easier to put everything in mask 1 given
    # that the distance isn't big
    assert np.all(new_assignment == 1)

    # finally, 3 blobs of 5, 10, and 20 spots around masks 2, 0 and 3
    x, y = rng.multivariate_normal([400, 402], cov, 5).T
    spot_positions = np.vstack([spot_positions, np.vstack([x, y]).T])
    x, y = rng.multivariate_normal([1000, 1000], cov, 20).T
    spot_positions = np.vstack([spot_positions, np.vstack([x, y]).T])
    closest = [0] * 10 + [2] * 5 + [3] * 20
    new_assignment = np.array(closest)
    # move 2 spots of 0 to 1 to give so
    new_assignment[[0, 1]] = 1
    # and a couple of others
    new_assignment[[5, 6]] = 2
    new_assignment[[7, 8]] = 3
    new_assignment[20:23] = 1
    new_assignment[24] = 0
    new_assignment[30:35] = -1
    iter = 0
    spot_moved = [1]
    while len(spot_moved):
        if iter == 3:
            print("last round")
        new_assignment, spot_moved = pa.assign_single_barcode_single_round(
            spot_positions=spot_positions,
            mask_positions=mask_positions,
            mask_assignments=new_assignment,
            max_distance_to_mask=mask_distance_threshold,
            inter_spot_distance_threshold=spot_distance_threshold,
            log_background_spot_prior=np.log(background_spot_prior),
            p=p,
            m=m,
            spot_distribution_sigma=spot_distribution_sigma,
            max_spot_group_size=5,
            verbose=0,
        )
        iter += 1
        if iter > 50:
            raise ValueError("Too many iterations")
    # now the spots should be assigned to the closest mask
    assert np.all(new_assignment == closest)


def test_assign_single_barcode_all_to_background():
    p = 0.9
    m = 0.1
    background_spot_prior = 0.0001
    spread = 10 / 0.2
    cov = [[spread, 0], [0, spread]]
    rng = np.random.default_rng(seed=12)
    mask_positions = np.array([[10, 10], [20, 20], [400, 402], [1000, 1000]])
    spot_distribution_sigma = 60
    for nspots in range(5):
        x, y = rng.multivariate_normal([10, 10], cov, nspots).T
        spot_positions = np.vstack([x, y]).T
        mask_assignments = np.zeros(len(spot_positions), dtype=int)
        new_ass, spot_moved = pa.assign_single_barcode_all_to_background(
            spot_positions,
            mask_positions,
            mask_assignments,
            log_background_spot_prior=np.log(background_spot_prior),
            p=p,
            m=m,
            spot_distribution_sigma=spot_distribution_sigma,
            distances=None,
            log_dist_likelihood=None,
        )
        if nspots == 0:
            assert not len(spot_moved)
            assert not len(new_ass)
            continue
        # with default param we need 3 spots to be a cell
        expected = -1 if nspots < 3 else 0
        assert len(spot_moved) == (expected == -1)
        assert np.all(new_ass == expected)


def create_data():
    # to make the number easy to compute, take weird parameters
    log_background_spot_prior = 1
    p = 1
    m = 1

    n_rolonies = 8
    n_masks = 5
    # make a fake mask assignment
    mask_assignments = np.zeros(n_rolonies, dtype=int)
    mask_assignments[1] = -1
    mask_assignments[2] = -1
    mask_assignments[3] = 2
    mask_assignments[4:7] = 4
    mask_counts = np.bincount(
        mask_assignments[mask_assignments >= 0], minlength=n_masks
    )
    log_dist_likelihood = (
        np.arange(n_rolonies)[:, np.newaxis] * np.arange(n_masks)[np.newaxis, :]
    )
    return (
        mask_assignments,
        mask_counts,
        log_dist_likelihood,
        log_background_spot_prior,
        p,
        m,
    )


if __name__ == "__main__":
    test_assign_barcodes_to_masks()
    test_assign_single_barcode_all_to_background()
    test_assign_single_round()
    test_likelihood_change_move_combination()
    test_valid_spot_combination()
    test_likelihood_change_background_combination()
