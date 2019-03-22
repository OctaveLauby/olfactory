import numpy as np
from bisect import bisect_right
from collections import defaultdict, OrderedDict


# ----------------------------------- #
# Level


def count_within(values, intervals):
    """Count number of values within each interval

    Args:
        values (n-list of nb)
        intervals (m-list of 2-nb-tuple)

    Return:
        (m-np.ndarray) count for each interval
    """
    rvals = np.reshape(values, [-1, 1])
    intervals_m = np.array(intervals)
    rlow = intervals_m[:, 0].reshape(1, -1)
    rhigh = intervals_m[:, 1].reshape(1, -1)

    flags = (rlow <= rvals) * (rvals < rhigh)
    return np.sum(flags, axis=0)


def count_within_bkd(values, intervals):
    """Return number of values within each interval in a dictionary

    Args:
        values (n-list of nb)
        intervals (m-list of 2-nb-tuple)

    Return:
        (OrderedDict) key=interv value=nb of values within interv
    """
    return OrderedDict([
        (tuple(i), c)
        for i, c in zip(intervals, count_within(values, intervals))
     ])


def index_lvl(a, thlds):
    """Index level(s) of value(s) given thlds b/w those levels

    Args:
        a (nb or n-np.ndarray)  : value or array of values
        thlds (list)            : list of sorted thresholds b/w levels

    Return:
        level index(es) (int or n-int-np.ndarray)
            res_i = index of level regarding thlds
            res_i = k means thld_(k-1) <= a_i < thld_k

    Example:
        >> index_lvl(np.array([0, 10, 20]), [10])
        [0, 1, 1]
    """
    if isinstance(a, (float, int)):
        return bisect_right(thlds, a)

    res = np.zeros(len(a))
    for thld in thlds:
        res += thld <= a
    return res.astype(int)


def label_lvl(a, thlds, labels):
    """Label level(s) of value(s) given thlds b/w those levels

    Args:
        a (nb or n-np.ndarray)  : value or array of values
        thlds (list)            : list of sorted thresholds b/w levels
        labels (m+1-list)       : level names
            label_k = level b/w thld_(k-1) & thld_k

    Return:
        level label(s) (int or n-int-np.ndarray)
            res_i = label_k if thld_(k-1) <= a_i < thld_k

    Example:
        >> label_lvl(np.array([0, 10, 20, 5, 7]), [10], ['low', 'high'])
        ['low', 'high', 'high', 'low', 'low']
    """
    if len(labels) != len(thlds) + 1:
        raise ValueError("Must be one more label than number of thresholds")
    lvl_indexes = index_lvl(a, thlds)
    return np.take(labels, lvl_indexes)


def bkd_lvl(a, thlds, labels):
    """Breakdown level spans per label given thlds b/w levels

    Args:
        a (n-np.ndarray)    : array of values
        thlds (list)        : list of sorted thresholds b/w levels
        labels (m+1-list)   : level names
            label_k = level b/w thld_(k-1) & thld_k

    Return:
        (dict)
        {
            <label>: (2-int-tuple-list)
        }

    Example:
        >> bkd_lvl(np.array([0, 10, 20, 5, 7]), [10], ['low', 'high'])
        {'low': [(0, 0), (3, 4)], 'high': [(1, 2)]}
    """
    res = defaultdict(list)
    flags = label_lvl(a, thlds, labels)
    lsti = 0
    case = flags[lsti]
    for i, flag in enumerate(flags[1:], 1):
        if flag == case:
            continue
        else:
            res[case].append((lsti, i-1))
            case = flag
            lsti = i
    if lsti != i:
        res[case].append((lsti, i))
    return dict(res)


# ----------------------------------- #
# In-betweens

def count_inbetweens(refticks, contenders, end=None):
    """Count number of intervals with a tick of contender in-b/w"""
    end = refticks[-1] if end is None else end

    refticks_a = np.reshape(refticks, (-1, 1))
    refticks_b = np.reshape(np.concatenate([refticks[1:], [end]]), (-1, 1))

    res = []
    for contender in contenders:
        inbetweens = np.sum(  # rtick b/w 2 cons. mtick (TCyc recovered in time)
            (contender - refticks_a >= 0) * (refticks_b - contender > 0),
            axis=1
        )
        res.append(inbetweens)
    return np.array(res)


def has_inbetweens(*args, **kwargs):
    return count_inbetweens(*args, **kwargs) > 0


def count_recoveries(*args, **kwargs):
    return np.sum(has_inbetweens(*args, **kwargs), axis=1)


# ----------------------------------- #
# Noise

def monotonic_dev(X, Y, gradsign=None, gradwindow=5, amplification=None):
    """Return total deviation from a monotonic behavior

    Gives and idea of total noise of signal.

    Args:
        X (n-np.ndarray)
        Y (n-np.ndarray)
        gradsign (float): sign of expected monotony
        gradwindow (int): size of window to smart compute gradsign
        amplification (callable): noise amplification
            (np.ndarray)-> (np.ndarray)

    Return:
        (float) monotonic deviation
    """
    def totdev(dY):
        """Return total deviation"""
        ampl = (lambda x: x) if amplification is None else amplification
        return np.sum(ampl(np.abs(dY)))

    if len(Y) == 0:
        return np.nan
    elif len(Y) == 1:
        return 0

    if gradsign is None:
        gradwindow = min(gradwindow, (len(Y) + 1) // 2)
        yfst = np.mean(Y[:gradwindow])
        ylst = np.mean(Y[-gradwindow:])
        gradsign = (ylst - yfst) if gradsign is None else gradsign

    dY = np.diff(Y)
    if gradsign == 0:
        return totdev(dY)
    elif gradsign < 0:
        return 2 * totdev(dY[dY > 0])
    else:  # gradsign > 0
        return 2 * totdev(dY[dY < 0])


def monotonic_devr(X, Y, **kwargs):
    """Return deviation ratio from a monotonic behavior

    Gives an idea of mean noise in signal.

    Args:
        X (n-np.ndarray)
        Y (n-np.ndarray)
        **kwargs: @see monotonic_dev

    Return:
        (float) monotonic deviation / n
    """
    return monotonic_dev(X, Y, **kwargs) / len(Y)
