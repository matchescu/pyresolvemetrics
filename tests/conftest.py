import pytest

from abstractions.data_structures import Clustering


@pytest.fixture
def gold_standard():
    return Clustering(
        feature_info=[],
        clustered_rows=[
            (("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")),
            (("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")),
            (("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")),
            (("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")),
        ],
    )
