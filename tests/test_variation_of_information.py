from abstractions.data_structures import Clustering
from entity_resolution_measurements import variation_of_information


def test_vi(gold_standard):
    er_result = Clustering(
        feature_info=[],
        clustered_rows=[
            (("a", "b"), ("c", "d"), ("e", "f", "g"), ("h", "i", "j"))
        ]
    )
    vi = variation_of_information(er_result, gold_standard)

    assert vi == [0.4158883083359672]
