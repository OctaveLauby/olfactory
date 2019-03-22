import numpy as np

from olfactory.detection import drop
from olfactory.operations import grad


def test_drop():
    Y = np.array([100, 110, 90, 60, 50, 200, 170, 150, 110])
    G = np.concatenate([[Y[1] - Y[0]], np.diff(Y)])
    X = np.zeros(9) + np.concatenate([[0], np.diff(Y)])
    H = grad(X, G)
    XYGH = (X, Y, G, H)
    np.testing.assert_equal(drop.detect_drop(
        XYGH, y0=55,
    ), [False, False, False, False, True, False, False, False, False])
    np.testing.assert_equal(drop.detect_drop(
        XYGH, y0=55, y1=101, g1=-29,
    ), [False, False, False, True, True, False, False, False, False])
    np.testing.assert_equal(drop.detect_drop(
        XYGH, y0=55, y1=101, g1=-29, y2=200, g2=-19
    ), [False, False, True, True, True, False, True, True, True])
