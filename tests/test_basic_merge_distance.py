import pytest

from abstractions.data_structures import Clustering
from entity_resolution_measurements import basic_merge_distance


@pytest.fixture
def gold_standard():
    return Clustering(
        feature_info=[],
        clustered_rows=[
            (("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")),
            (("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")),
        ],
    )


def result_one():
    return Clustering(
        feature_info=[],
        clustered_rows=[
            (("a"), ("b"), ("c"), ("d"), ("e", "f", "g", "h", "i", "j"))
        ]
    )


def result_two():
    return Clustering(
        feature_info=[],
        clustered_rows=[
            (("a", "b"), ("c", "d"), ("e", "f", "g"), ("h", "i", "j"))
        ]
    )


@pytest.mark.parametrize(
    "actual_result, expected_bmd", [(result_one(), 2.0), (result_two(), 1.0)]
)
def test_basic_merge_distance(gold_standard, actual_result, expected_bmd):
    result = basic_merge_distance(gold_standard, actual_result)

    assert result == expected_bmd


def test_basic_merge_distance_sums_pairwise_distances(gold_standard):
    entity_resolution_result = Clustering(
        feature_info=[],
        clustered_rows=[
            (("a"), ("b"), ("c"), ("d"), ("e", "f", "g", "h", "i", "j")),
            (("a", "b"), ("c", "d"), ("e", "f", "g"), ("h", "i", "j")),
        ]
    )

    result = basic_merge_distance(entity_resolution_result, gold_standard)

    assert result == 3.0
