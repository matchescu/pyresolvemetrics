import pytest


@pytest.fixture
def algebraic_ground_truth():
    return {("a", "b"), ("c", "d"), ("e", "f", "g", "h", "i", "j")}
