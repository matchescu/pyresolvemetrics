import pytest

from matchescu.metrics.algebraic import twi


def test_empty_set():
    assert twi([set()], [set()]) == 0


def test_single_item():
    assert twi([{1}], [{1}]) == 1


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (
            [
                {1, 2, 3, 4, 5, 6},
                {7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                {17, 18, 19},
                {20, 21, 22, 23, 24, 25}
            ],
            [
                {1, 2},
                {3, 4, 5, 6},
                {7, 8, 9, 10, 12, 13, 14, 15, 16, 25},
                {17, 20},
                {18, 19, 22, 24},
                {11, 21, 23}
            ],
            0.24,
        ),
        ([{1, 2}, {3, 4, 5, 6}], [{1, 2}, {3, 4}, {5, 6}], 0.666666666666666667)
    ]
)
def test_basic_cases(x, y, expected):
    assert twi(x, y) == expected
