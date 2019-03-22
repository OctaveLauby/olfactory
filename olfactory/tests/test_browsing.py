import numpy as np
import pytest

from olfactory import browsing


def test_chunk():
    assert list(browsing.chunk([1, 2, 3, 0, -1], 2)) == [[1, 2], [3, 0], [-1]]
    assert list(browsing.chunk(range(10), 4)) == [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9]
    ]

    with pytest.raises(TypeError):
        browsing.chunk([1, 2, 3, 4, 5], 2.)

    with pytest.raises(ValueError):
        browsing.chunk([1, 2, 3, 4, 5], 0)


def test_sliding_window():

    with pytest.raises(ValueError):
        browsing.sliding_window(np.array([1]), 2)

    np.testing.assert_equal(
        browsing.sliding_window(np.array([0, 1]), 2),
        np.array([[0, 1]])
    )
    np.testing.assert_equal(
        browsing.sliding_window(np.array([0, 1, 2, 4]), 2),
        np.array([[0, 1], [1, 2], [2, 4]])
    )
