from abstractions.data_structures import Clustering
from entity_resolution_measurements import variation_of_information


def test_vi(gold_standard):
    standard = Clustering(feature_info=[], clustered_rows=[gold_standard] * 4)
    er_result = Clustering(
        feature_info=[],
        clustered_rows=[
            (("a", "b"), ("c", "d"), ("e", "f", "g"), ("h", "i", "j"))
        ]
    )
    vi = variation_of_information(er_result, standard)

    assert vi == [0.4158883083359672]
