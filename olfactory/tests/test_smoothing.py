import numpy as np
import pytest

from olfactory import smoothing


def test_x_to_i_window():

    assert smoothing.x_to_i_window([0, 1], 3) == 4
    assert smoothing.x_to_i_window([0, 1], 2) == 3
    assert smoothing.x_to_i_window([1, 2], 0.3) == 1
    assert smoothing.x_to_i_window([1, 3, 5], 4) == 3

    with pytest.raises(ValueError):
        smoothing.x_to_i_window([1, 3, 4], 2)


def test_odd_window():

    assert smoothing._odd_window(list(range(5)), 3) == 3
    assert smoothing._odd_window(list(range(5)), 5) == 5
    assert smoothing._odd_window(list(range(5)), 2) == 3
    assert smoothing._odd_window(list(range(5)), 4) == 5

    assert smoothing._odd_window(list(range(1)), 100) == 1
    assert smoothing._odd_window(list(range(2)), 100) == 1
    assert smoothing._odd_window(list(range(3)), 100) == 3
    assert smoothing._odd_window(list(range(4)), 100) == 3
    assert smoothing._odd_window(list(range(5)), 100) == 5
    assert smoothing._odd_window(list(range(6)), 100) == 5

    assert smoothing._odd_window([0, 1], 2, xwindow=True) == 1
    assert smoothing._odd_window([0, 1, 2], 2, xwindow=True) == 3
    assert smoothing._odd_window([0, 1, 2, 3], 3, xwindow=True) == 3
    assert smoothing._odd_window([1, 2], 0.5, xwindow=True) == 1
    assert smoothing._odd_window([1, 3, 5], 4, xwindow=True) == 3


def test_savgol_smooth():

    X = [1, 2, 3, 4]
    Y = [0, 2, 2, 3]
    np.testing.assert_almost_equal(
        smoothing.savgol_smooth(X, Y, window=3, polyorder=1),
        [1/3, 4/3, 7/3, 17/6]
    )

    X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    Y = np.array([0, 3, 0, 3, 0, 3, 0, 3])
    res = [0.3428571, 1.6285714, 2.0571429, 0.9428571, 2.0571429, 0.9428571, 1.3714286, 2.6571429]
    np.testing.assert_almost_equal(
        smoothing.savgol_smooth(X, Y, window=5, polyorder=2), res
    )
    np.testing.assert_almost_equal(  # Auto increment window when even
        smoothing.savgol_smooth(X, Y, window=4, polyorder=2), res
    )
    np.testing.assert_almost_equal(
        smoothing.savgol_smooth(2*X, Y, window=8, polyorder=2, xwindow=True),
        res
    )

    np.testing.assert_almost_equal(
        smoothing.savgol_smooth([0, 1, 2], [0, 10, 20], 4, 1),
        [0, 10, 20]
    )

    with pytest.raises(ValueError):  # Irregular step
        smoothing.savgol_smooth([0, 1, 3], [0, 10, 30], 3, 1, xwindow=True)


def test_window_smooth():

    X = np.array([0, 2, 4, 6])
    Y = np.array([0, 2, 2, 3])
    np.testing.assert_almost_equal(
        smoothing.window_smooth(X, Y, window=3),
        [1, 4/3, 7/3, 2.5]
    )
    np.testing.assert_almost_equal(
        smoothing.window_smooth(X, Y, window=2),
        [1, 4/3, 7/3, 2.5]
    )
    np.testing.assert_almost_equal(
        smoothing.window_smooth(X, Y, window=3, wfading=1),
        Y
    )
    np.testing.assert_almost_equal(
        smoothing.window_smooth(X, Y, window=3, wfading=0.5),
        [1/1.5, 3/2, 4.5/2, 4/1.5]
    )

    X = [1, 2, 3, 4, 5, 6, 7, 8]
    Y = [0, 3, 0, 3, 0, 3, 0, 3]
    np.testing.assert_almost_equal(
        smoothing.window_smooth(X, Y, window=5),
        [1, 1.5, 1.2, 1.8, 1.2, 1.8, 1.5, 2]
    )
