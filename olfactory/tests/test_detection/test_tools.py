import numpy as np

from olfactory.detection import tools


def test_group_consecutives():
    np.testing.assert_equal(
        tools.group_consecutives([0, 1, 3, 4, 5, 7, 9, 10]),
        [[0, 1], [3, 4, 5], [7], [9, 10]]
    )
    np.testing.assert_equal(
        tools.group_consecutives([0, 1, 3, 4, 5, 7, 9, 10], 2),
        [[0], [1, 3], [4], [5, 7, 9], [10]]
    )
    np.testing.assert_equal(
        tools.group_consecutives([.5, 1, 2, 3.5, 4.5, 5]),
        [[.5], [1, 2], [3.5, 4.5], [5]]
    )


def test_diff():
    assert tools.diff(
        [1, 2, "salut", "bye"], [3, "bye", "aurevoir", 2]
    ) == {
        'common': {2, "bye"},
        'minus': {1, "salut"},
        'plus': {3, "aurevoir"},
    }


def test_linearize():
    np.testing.assert_equal(
        tools.linearize([0, 5, 6, 13, 16]),
        [0, 4, 8, 12, 16],
    )
    np.testing.assert_equal(
        tools.linearize([0, 5, 6, 13, 16], index=2),
        [0, 3, 6, 11, 16],
    )
    np.testing.assert_equal(
        tools.linearize([0, 5, 6, 13, 16, 21, 2, 0]),
        [0] * 8,
    )
    np.testing.assert_equal(
        tools.linearize([0, 5, 6, 13, 16, 21, 2], index=4),
        [0, 4, 8, 12, 16, 9, 2],
    )
