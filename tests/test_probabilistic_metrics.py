import pytest

from pyresolvemetrics._probabilistic import precision, recall, f1


@pytest.fixture
def ground_truth(request):
    if hasattr(request, "param"):
        return request.param
    return {("a", "b"), ("c", "d")}


@pytest.fixture
def result(request):
    if hasattr(request, "param"):
        return request.param
    return {("a", "b"), ("c", "d")}


@pytest.mark.parametrize(
    "ground_truth, result, expected",
    [
        ({("a", "b")}, {("a", "b")}, 1),
        ({("a", "b")}, {("b", "a")}, 1),
        ({("a", "b"), ("c", "d")}, {("b", "a")}, 1),
        ({("a", "b"), ("c", "d")}, {("a", "c")}, 0),
        ({("a", "b"), ("c", "d")}, {("b", "a"), ("d", "e")}, 0.5),
        ({("a", "b"), ("c", "d")}, set(), 0),
    ],
)
def test_precision(ground_truth, result, expected):
    actual = precision(ground_truth, result)
    assert actual == expected


@pytest.mark.parametrize(
    "ground_truth, result, expected",
    [
        ({("a", "b")}, {("a", "b")}, 1),
        ({("a", "b")}, {("b", "a")}, 1),
        ({("a", "b")}, {("b", "a"), ("c", "d")}, 1),
        ({("a", "b"), ("c", "d")}, {("a", "b")}, 0.5),
        ({("a", "b"), ("c", "d")}, {("d", "e")}, 0),
        ({("a", "b"), ("c", "d")}, set(), 0),
    ],
)
def test_recall(ground_truth, result, expected):
    actual = recall(ground_truth, result)
    assert actual == expected


@pytest.mark.parametrize(
    "ground_truth, result, expected",
    [
        ({("a", "b")}, {("a", "b")}, 1),
        ({("a", "b")}, {("b", "a")}, 1),
        ({("a", "b"), ("c", "d")}, {("a", "b")}, 0.67),
        ({("a", "b"), ("c", "d")}, {("d", "e")}, 0),
        ({("a", "b"), ("c", "d")}, set(), 0),
    ],
)
def test_f1(ground_truth, result, expected):
    actual = f1(ground_truth, result)
    assert round(actual, 2) == expected
