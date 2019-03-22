import numpy as np

from olfactory.detection import yregularity


def test_detect_elbow():

    Y = np.array([0, 1, 2, 3, 4, 6, 8, 10])
    assert yregularity.detect_elbow(Y) == 4
    assert yregularity.detect_elbow(Y, mthd="singleline") == 4
    assert yregularity.detect_elbow(Y, mthd="doubleline") == 4

    Y = np.array([0.35, 0.36, 0.37, 0.40, 0.42, 0.49, 0.53, 0.65, 0.95, 1.35])
    assert yregularity.detect_elbow(Y) == 6
    assert yregularity.detect_elbow(Y, mthd="singleline") == 6
    assert yregularity.detect_elbow(Y, mthd="doubleline") == 7


def test_detect_iso():
    Y = np.array([10000, 2950, 3000, 2900, 2200, 3000, 2800, 2850, 2200, 1500])
    np.testing.assert_equal(
        yregularity.detect_iso(Y),
        [0, 4, 9]
    )
    np.testing.assert_equal(
        yregularity.detect_iso(Y, delta_r=0.099, lvlref=2000),
        [0, 4, 5, 9]
    )

    Y = np.array([3025, 3000, 2900, 3100, 2200, 3000, 2850, 2200, 2000])
    np.testing.assert_equal(yregularity.detect_iso(Y), [4])

    np.testing.assert_equal(yregularity.detect_iso(np.array([1, 10000])), [])
    np.testing.assert_equal(yregularity.detect_iso(np.array([11, 12, 13])), [])
    np.testing.assert_equal(
        yregularity.detect_iso(np.array([1, 10, 1])),
        [0, 1, 2]
    )


def test_detect_leap():

    X = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    Y = np.array([6, 6, 4, 8, 8, 1, 1, 6])

    assert yregularity.detect_leap(None, Y, thld=3) == [3, 7]
    assert yregularity.detect_leap(None, Y, thld=4) == [3, 7]
    assert yregularity.detect_leap(None, Y, thld=3, lvl_thld=7) == [3]
    assert yregularity.detect_leap(X, Y, thld=3, onspan=1) == [3, 7]
    assert yregularity.detect_leap(X, Y, thld=4, onspan=1) == [7]
    assert yregularity.detect_leap(X, Y, thld=3, onspan=1, wfading=0.5) == [3, 7]

    assert yregularity.detect_leap(None, Y, thld=-3) == [5]
    assert yregularity.detect_leap(None, Y, thld=-3, lvl_thld=0) == []
    assert yregularity.detect_leap(None, Y, thld=-3, lvl_thld=1) == [5]
    assert yregularity.detect_leap(None, Y, thld=-3, lvl_thld=2) == [5]
    assert yregularity.detect_leap(X, Y, thld=-3, onspan=1) == [5]
    assert yregularity.detect_leap(X, Y, thld=-3, onspan=2) == [5]
