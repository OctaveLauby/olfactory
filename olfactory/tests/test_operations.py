import pytest
import numpy as np

from olfactory import operations


def test_grad():
    grad = operations.grad

    np.testing.assert_equal(grad([1, 2], [1, 3]), [2, 2])

    X = np.array([0, 1, 2, 3, 4])
    Y = np.array([0, 2, 4, 6, 8])
    np.testing.assert_equal(grad(X, Y), [2, 2, 2, 2, 2])

    X = np.array([0, 1, 2, 3, 4])
    Y = np.array([0, 2, 3, 2, 0])
    np.testing.assert_equal(grad(X, Y), [2, 1.5, 0, -1.5, -2])

    X = np.array([0, 1, 3, 5, 6])
    Y = np.array([0, 6, 12, 18, 24])
    np.testing.assert_equal(grad(X, Y), [6, 4, 3, 4, 6])

    with pytest.raises(ValueError):
        grad([1], [2])


def test_isnull():

    assert operations.isnull(None)
    assert operations.isnull(np.nan)

    assert not operations.isnull(1)
    assert not operations.isnull(0.)
    assert not operations.isnull(np.array([1, 2]))
    assert not operations.isnull(np.array([None, None]))
    assert not operations.isnull(np.array([np.nan, np.nan]))
    assert not operations.isnull(np.array([]))
    assert not operations.isnull("string")
    assert not operations.isnull("")
