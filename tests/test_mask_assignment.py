import numpy as np
from iss_analysis.barcodes import probabilistic_assignment as pa

# Test the mask assignment functions


def test_valid_spot_combination():
    distance_threshold = 10
    # set of spots too far from each other
    spot_positions = np.vstack([(0, 0), (100, 100), (200, 200)])
    valid = pa.valid_spot_combination(spot_positions, distance_threshold)
    assert len(valid) == len(spot_positions)

    # set of spots very close to each other
    spot_positions = np.vstack([(0, 0), (1, 1), (2, 2)])
    valid = pa.valid_spot_combination(spot_positions, distance_threshold)
    assert len(valid) == 7

    # set of spots very close to each other, with one far away
    spot_positions = np.vstack([(0, 0), (1, 1), (2, 2), (100, 100)])
    valid = pa.valid_spot_combination(spot_positions, distance_threshold)
    assert len(valid) == 8

    # spots close to each other by pairs but not triples
    spot_positions = np.vstack([(0, 0), (0, 7), (0, 14)])
    valid = pa.valid_spot_combination(spot_positions, distance_threshold)
    assert len(valid) == 5


def test_likelihood_change_background_combination():
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
    params = dict(
        mask_assignments=mask_assignments,
        mask_counts=mask_counts,
        log_dist_likelihood=log_dist_likelihood,
        log_background_spot_prior=log_background_spot_prior,
        p=p,
        m=m,
    )
    # moving a background spots should not change the likelihood
    spot_ids = np.array([1], dtype=int)
    out = pa.likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, 0)

    # even if it's multiple background spots
    spot_ids = np.array([1, 2], dtype=int)
    out = pa.likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, 0)

    # moving just spot 0 should change the likelihood
    # spot 0 is assigned to mask 0 with a mask_counts of 3
    # the distance likelihood of this spot is 0
    # with p=m=1 the prior is - spot_count
    expected = log_background_spot_prior + (-2) - (-3)
    log_dist_likelihood[0, 0]
    out = pa.likelihood_change_background_combination(
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
    out = pa.likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, expected)

    # test moving 2 spots from different masks
    spot_ids = np.array([3, 5], dtype=int)
    # they are from mask 2 and 4, with counts 1 and 3 respectively
    expected = (
        log_background_spot_prior * 2
        + (-0 - 2)
        - (-1 + -3 + log_dist_likelihood[spot_ids, mask_assignments[spot_ids]].sum())
    )
    out = pa.likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, expected)

    # †ry to move 0 spots
    spot_ids = np.array([], dtype=int)
    out = pa.likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, 0)

    # try to move all spots
    spot_ids = np.arange(n_rolonies)
    expected = log_background_spot_prior * n_rolonies
    bg = mask_assignments == -1
    expected -= log_background_spot_prior * bg.sum()
    expected -= (-mask_counts).sum()
    expected -= log_dist_likelihood[
        spot_ids[~bg], mask_assignments[spot_ids[~bg]]
    ].sum()
    out = pa.likelihood_change_background_combination(spot_ids=spot_ids, **params)
    assert np.allclose(out, expected)


if __name__ == "__main__":
    test_valid_spot_combination()
    test_likelihood_change_background_combination()
