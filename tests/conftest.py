import pytest


@pytest.fixture
def gold_standard():
    return [("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")]
