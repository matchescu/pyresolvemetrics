import pytest

from pyresolvemetrics.algebraic import rand_index


def test_rand_empty_returns_zero():
    assert rand_index([], []) == 0


def test_rand_singleton_returns_zero():
    assert rand_index([{1}], [{1}]) == 0


def test_rand_same_single_pair_returns_one():
    assert rand_index([{1, 2}], [{1, 2}]) == 1


def test_rand_two_identical_clusters_returns_one():
    assert rand_index([{1, 2}, {3, 4, 5}], [{1, 2}, {3, 4, 5}]) == 1


@pytest.mark.parametrize(
    "ground_truth, result, expected_score",
    [
        ([{1, 2}, {3, 4}, {5, 6}], [{1, 2, 3}, {4, 5}, {6}], 0.66666666666666667),
        ([{1, 2}, {3, 4, 5, 6}], [{4, 6}, {1, 2, 3, 5}], 0.46666666666666667),
    ]
)
def test_rand_simple_different_clusters(
    ground_truth, result, expected_score
):
    assert rand_index(ground_truth, result) == expected_score
