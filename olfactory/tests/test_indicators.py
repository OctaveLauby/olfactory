import numpy as np

from olfactory import indicators


# ----------------------------------- #
# Labels

def test_count_within():
    count_within = indicators.count_within

    np.testing.assert_equal(
        count_within([0, 0, 1, 2, 5, 1, 6], [(1, 3), (5, 7)]),
        [3, 2],
    )
    np.testing.assert_equal(
        count_within([0, 0, 1, 2, 5, 1, 6], [(1, 3), (1, 3)]),
        [3, 3],
    )


def test_count_within_bkd():
    count_within_bkd = indicators.count_within_bkd

    assert count_within_bkd([0, 0, 1, 2, 5, 1, 6], [(1, 3), [5, 7]]) == {
        (1, 3): 3,
        (5, 7): 2,
    }


def test_index_lvl():
    index_lvl = indicators.index_lvl

    np.testing.assert_equal(index_lvl(np.array([0, 10, 20]), [10]), [0, 1, 1])
    thlds = [1, 2, 3, 4]
    a = np.array([0, 1, 1.5, 2, 2.5, 3.5, 4, 1.5, 1.6])
    np.testing.assert_equal(index_lvl(a, thlds), [0, 1, 1, 2, 2, 3, 4, 1, 1])
    assert index_lvl(0, thlds) == 0
    assert index_lvl(1, thlds) == 1
    assert index_lvl(2.5, thlds) == 2
    assert index_lvl(5, thlds) == 4


def test_label_lvl():
    label_lvl = indicators.label_lvl

    np.testing.assert_equal(
        label_lvl(np.array([0, 10, 20, 5, 7]), [10], ['low', 'high']),
        ['low', 'high', 'high', 'low', 'low']
    )
    thlds = [1, 2, 3, 4]
    labels = ["VL", "L", "M", "H", "VH"]
    a = np.array([0, 1, 1.5, 2, 2.5, 3.5, 4, 1.5, 1.6])
    np.testing.assert_equal(
        label_lvl(a, thlds, labels),
        ['VL', 'L', 'L', 'M', 'M', 'H', 'VH', 'L', 'L']
    )
    assert label_lvl(0, thlds, labels) == 'VL'
    assert label_lvl(1, thlds, labels) == 'L'
    assert label_lvl(2.5, thlds, labels) == 'M'
    assert label_lvl(5, thlds, labels) == 'VH'


def test_brkd_lvl():
    brkd_lvl = indicators.bkd_lvl

    np.testing.assert_equal(
        brkd_lvl(np.array([0, 10, 20, 5, 7]), [10], ['low', 'high']),
        {'low': [(0, 0), (3, 4)], 'high': [(1, 2)]}
    )
    a = np.array([0, 1, 1.5, 2, 2.5, 3.5, 4, 1.5, 1.6])
    thlds = [1, 2, 3, 4]
    labels = ["VL", "L", "M", "H", "VH"]
    assert brkd_lvl(a, thlds, labels) == {
        'VL': [(0, 0)],
        'L': [(1, 2), (7, 8)],
        'M': [(3, 4)],
        'H': [(5, 5)],
        'VH': [(6, 6)],
    }


# ----------------------------------- #
# In-between

def test_inbetweens():
    T, F = True, False
    refticks = np.array([5, 10, 20, 50, 100, 200, 500])
    c1 = np.array([3, 8, 23, 30, 77])
    c2 = np.array([0, 100, 1000, 10000])
    c3 = np.array([5, 51, 501])

    res = indicators.count_inbetweens(refticks, [c1, c2, c3])
    np.testing.assert_equal(res, [
        [1, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
    ])
    res = indicators.has_inbetweens(refticks, [c1, c2, c3])
    np.testing.assert_equal(res, [
        [T, F, T, T, F, F, F],
        [F, F, F, F, T, F, F],
        [T, F, F, T, F, F, F],
    ])
    res = indicators.count_recoveries(refticks, [c1, c2, c3])
    np.testing.assert_equal(res, [3, 1, 2])

    res = indicators.count_inbetweens(refticks, [c1, c2, c3], end=np.inf)
    np.testing.assert_equal(res, [
        [1, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 2],
        [1, 0, 0, 1, 0, 0, 1],
    ])
    res = indicators.has_inbetweens(refticks, [c1, c2, c3], end=np.inf)
    np.testing.assert_equal(res, [
        [T, F, T, T, F, F, F],
        [F, F, F, F, T, F, T],
        [T, F, F, T, F, F, T],
    ])

    # Pile of misses
    refticks = np.array([5, 10, 10, 20, 20])
    c1 = np.array([5, 10, 20])
    c2 = np.array([0, 15, 25])
    res = indicators.has_inbetweens(refticks, [c1, c2])
    np.testing.assert_equal(res, [
        [T, F, T, F, F],
        [F, F, T, F, F],
    ])

    # Pile of misses and recoveries
    refticks = np.array([10, 10, 10, 10, 15])
    c1 = np.array([10, 10, 10, 10])
    res = indicators.has_inbetweens(refticks, [c1])
    np.testing.assert_equal(res, [
        [F, F, F, T, F],
    ])
    refticks = np.array([10, 10, 10, 10, 15])
    c1 = np.array([10, 10, 10, 10])
    res = indicators.count_recoveries(refticks, [c1])
    np.testing.assert_equal(res, [1])


# ----------------------------------- #
# Noise


def test_monotonic_dev():

    Y = np.array([1, 2, 3])
    assert indicators.monotonic_dev(None, Y) == 0
    assert indicators.monotonic_dev(None, Y, gradsign=-1) == 2*2
    assert indicators.monotonic_dev(None, Y, gradsign=0) == 2
    assert indicators.monotonic_dev(None, [0, 3, 5, 6, 6]) == 0
    assert indicators.monotonic_dev(None, [0, 3, 0, 6]) == 2*3

    assert indicators.monotonic_dev(None, [0, 1, -1, 0]) == 4
    assert indicators.monotonic_dev(None, [0, 1, -1, 0], gradsign=0) == 4
    assert indicators.monotonic_dev(None, [0, 2, -2, 0]) == 8
    assert indicators.monotonic_dev(None, [0, 2, -2, 0], gradsign=0) == 8

    assert indicators.monotonic_dev(
        None, [1, 0, 2, 0, 3, 0, 4]
    ) == 2*6
    assert indicators.monotonic_dev(
        None, [1, 0, 2, 0, 3, 0, 4], amplification=lambda x: x**2
    ) == 2*14
    assert indicators.monotonic_dev(
        None, [1, 0, 2, 0, 3, 0, 4], amplification=lambda x: x[x > 2]
    ) == 2*3


def test_monotonic_devr():

    Y = np.array([1, 2, 3])
    assert indicators.monotonic_devr(None, Y) == 0
    assert indicators.monotonic_devr(None, Y, gradsign=-1) == 4 / 3
    assert indicators.monotonic_devr(None, Y, gradsign=0) == 2 / 3
    assert indicators.monotonic_devr(None, [0, 3, 5, 6, 6]) == 0
    assert indicators.monotonic_devr(None, [0, 3, 0, 6]) == 6 / 4

    assert indicators.monotonic_devr(None, [0, 1, -1, 0]) == 4 / 4
    assert indicators.monotonic_devr(None, [0, 1, -1, 0], gradsign=0) == 4 / 4
    assert indicators.monotonic_devr(None, [0, 2, -2, 0]) == 8 / 4
    assert indicators.monotonic_devr(None, [0, 2, -2, 0], gradsign=0) == 8 / 4
