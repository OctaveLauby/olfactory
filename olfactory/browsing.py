"""Functions to browse structures"""
import numpy as np

from olutils import check_type


def chunk(iterable, size):
    """Chunk iterable into batch of size-elements (last one can be shorter)"""
    check_type("size", size, int)
    if size < 1:
        raise ValueError("Chunk size must >= 1, got %s" % size)

    # Must put yield in sub-function so that arg checking is made right away
    def biterator():
        batch = []
        for elem in iterable:
            batch.append(elem)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    return biterator()


def sliding_window(a, window):
    """Build a sliding window on array

    Args:
        a (n-np.ndarray): array to build sliding window on
        window (int)    : size of window

    Return:
        (np.ndarray) len = len(a) - window + 1
    """
    if len(a.shape) == 1 and len(a) < window:
        raise ValueError("Window must be smaller than length of array")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
