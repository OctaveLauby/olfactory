import numpy as np


def diff(list1, list2):
    """Return diff b/w lists in a dictionary

    About:
        Because 0 == False and 1 == True, diff may not work as wanted with
        list mixing booleans and integers.
    """
    s1 = set(list1)
    s2 = set(list2)
    common = s1.intersection(s2)
    return {
        'common': common,
        'minus': s1 - common,
        'plus': s2 - common,
    }


def group_consecutives(a, step=1):
    """Group step-consecutive elements in a list of arrays

    Example:
        >> group_consecutives([1, 2, 4, 5, 6, 9], step=1)
        [[1, 2], [4, 5, 6], [9]]
    """
    if len(a) == 0:
        return []
    return np.split(a, np.where(np.diff(a) != step)[0] + 1)


def linearize(a, index=-1):
    """Linearize vector in 2 linear segments

    Assumption: a is based on regular step

    Args:
        a (np.ndarray)
        index (int): index where to split linearization
            if index out of bounds, return one segment

    Return:
        (np.ndarray)
    """
    if index <= 0 or index >= (len(a) - 1):
        return ((a[-1] - a[0]) / (len(a)-1)) * np.arange(0, len(a)) + a[0]

    y = a[index]
    fst_seg = ((y - a[0]) / index) * np.arange(index+1) + a[0]
    rindex = len(a) - index - 1
    lst_seg = ((a[-1] - y) / rindex) * np.arange(rindex+1) + y
    return np.concatenate([fst_seg, lst_seg[1:]])
