import numpy as np


def isnull(obj):
    """Return whether object is None or np.nan"""
    if obj is None:
        return True
    try:
        return bool(np.isnan(obj))
    except (ValueError, TypeError):
        return False


def grad(X, Y):
    """Compute Y derivative on X ticks

    Works better if X is regular.
    """
    with np.errstate(divide='raise'):
        try:
            return np.gradient(Y) / np.gradient(X)
        except FloatingPointError:
            raise ValueError(
                "Insufficient X to build gradient: %s" % X
            ) from None


def xygh(X, Y):
    """Compute Y derivative G and 2nd derivative H on X ticks, return XYGH tuple

    Works better if X is regular.
    """
    try:
        G = grad(X, Y)
        H = grad(X, G)
    except ValueError:
        G = None
        H = None
    return (X, Y, G, H)
