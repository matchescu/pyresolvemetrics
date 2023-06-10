import pytest

from abstractions.data_structures import Clustering
from entity_resolution_measurements import basic_merge_distance, gmd_slice


def one(*_):
    return 1


@pytest.mark.parametrize(
    "actual_result, expected_bmd",
    [
        ([("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")], 0.0),  # nothing required
        ([["a"], ["b"], ["c"], ["d"], ["e", "f", "g", "h", "i", "j"]], 2.0),  # 2 merges required
        ([["a", "b", "c", "d"], ["e", "f", "g", "h", "i", "j"]], 1.0),  # 1 split required
        ([["a", "b", "c", "d"], ["e", "f", "g"], ["h", "i", "j"]], 2.0),  # 1 split + 1 merge required
    ]
)
def test_gmd_split_happy_flow(gold_standard, actual_result, expected_bmd):
    result = gmd_slice(actual_result, gold_standard, one, one)

    assert result == expected_bmd


def test_gmd_split_item_in_result_not_in_standard(gold_standard):
    with pytest.raises(ValueError) as err_proxy:
        gmd_slice([("f", "g"), "m", ("a", "b")], gold_standard, one, one)

    assert str(err_proxy.value) == "the token 'c' is in the standard, but not in the result"


def test_basic_merge_distance_sums_pairwise_distances(gold_standard):
    standard = Clustering(feature_info=[], clustered_rows=[gold_standard, gold_standard])
    result = Clustering(
        feature_info=[],
        clustered_rows=[
            ("a", "b", "c", "d", ("e", "f", "g", "h", "i", "j")),
            (("a", "b"), ("c", "d"), ("e", "f", "g"), ("h", "i", "j")),
        ]
    )

    result = basic_merge_distance(result, standard)

    assert result == 3.0
