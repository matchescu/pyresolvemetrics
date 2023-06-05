from abstractions.data_structures import Clustering
from entity_resolution_measurements import pairwise_f1


def test_pairwise_f1_score_on_single_record(gold_standard):
    er_result = Clustering(
        feature_info=[],
        clustered_rows=[
            ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"),
            ("a", ("b", "c"), ("d", "e"), ("f", "g"), ("h", "i"), "j"),
            (("a", "b"), ("c", "d", "e"), ("f", "g", "h", "i", "j")),
            (("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")),
        ]
    )
    result = pairwise_f1(er_result, gold_standard)

    assert result == [
        0.0,
        0.19047619047619052,
        0.7741935483870968,
        1.0,
    ]
