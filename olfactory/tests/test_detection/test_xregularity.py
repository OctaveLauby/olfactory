import numpy as np

from olfactory.detection import xregularity


def test_reg_bounds():
    dX = np.array([0, 1, 2, 3, 4, 1, 0, 5, 6])
    assert xregularity.reg_bounds(dX, bot_thld=2, top_thld=4) == [
        2, 5, 7
    ]


def test_stepreg_bounds():
    X = np.array([0, 1, 2, 4, 6, 8, 11, 12, 16, 20, 22])
    assert xregularity.stepreg_bounds(X, bot_thld=2, top_thld=3) == [
        2, 6, 7, 9
    ]
    assert xregularity.stepreg_bounds(X, top_thld=1) == [2, 6, 7]
    assert xregularity.stepreg_bounds(X, top_thld=10) == []
