import numpy as np

from .tools import linearize


def detect_elbow(Y, mthd='singleline'):
    """Return index where elbow occurs

    Args:
        Y (np.ndarray)
        mthd (str): method to find elbow
            singleline> find farthest pt from fst-pt to lst-pt line
            doubleline> find pt where dist(fst-pt to pt to lst-pt, Y) is min

    Return:
        index (int)
    """
    if mthd == 'singleline':
        line = linearize(Y)
        return np.argmax(np.sqrt((Y-line)**2))
    elif mthd == 'doubleline':
        bst_index, bst_dist = None, np.inf
        for index, y in enumerate(Y[1:-1], 1):
            curve = linearize(Y, index)
            dist = np.linalg.norm(Y - curve)
            if dist <= bst_dist:
                bst_index, bst_dist = index, dist
        return bst_index
    else:
        raise ValueError("Unknown detection method '%s' % method")


def detect_iso(Y, delta_r=0.1, lvlref=None):
    """Return indexes of isolated points

    About:
        First Y is shifted so each value is >=0
        Yj is isolated if [i=j-1 and k=j+1]:
            not (b/w Yi and Yk)
            and min(|Yi - Yj / lvlref|, |Yk - Yj / lvlref|) > delta_r
        Borders can be isolated if ratio with closest Y is > delta_r

        Won't work properly iY has negative values

    Args:
        Y (float np.ndarray): list of values
        delta_r (float): max factor b/w lvlref and neighbors
        lvlref (float or callable): lvlref or func to compute lvlref from Y
            default is 9th percentile

    Return:
        (int-np.ndarray) indexes of isolated points
    """
    if len(Y) <= 2:
        return np.array([])

    lvlref = (
        lvlref if isinstance(lvlref, (float, int))
        else (
            lvlref(Y) if lvlref
            else np.percentile(Y, 90, interpolation="lower")
        )
    )
    if lvlref <= 0:
        raise ValueError("lvlref=%s <= 0" % lvlref)

    # Compute isolated points in center of Y
    dY = np.diff(Y)
    dY_l = -dY[:-1]
    dY_r = dY[1:]

    inbetween = dY_l * dY_r < 0
    delta = np.min([np.abs(dY_l) / lvlref, np.abs(dY_r) / lvlref], axis=0)

    # Add borders
    inbetween = np.concatenate([[False], inbetween, [False]])
    delta = np.concatenate(
        [[np.abs(dY[0]) / lvlref], delta, [np.abs(dY[-1]) / lvlref]]
    )

    return np.where((1 - inbetween) * (delta > delta_r))[0]


def detect_leap(X, Y, thld, lvl_thld=None, onspan=None, wfading=None):
    """Return indexes where leap is detected on Y

    Args:
        X (n-numpy.ndarray)
        Y (n-numpy.ndarray)
        thld (float)        : min diff b/w consecutive values to consider leap
            if thld is neg, thld considered as max diff b/w consec. values
        lvl_thld (float)    : min new value to consider leap
            if thld is neg, lvl_thld considered as max new value
        onspan (float)      : given a detected leap at x,
            compute prev_y on ticks b/w prev_x - onspan & prev_x
            compute next_y on ticks b/w x & x + onspan
        wfading (float): when computing prev_y or next_y, apply weight to
            each selected y_val [weight = 1 - wfading * |x-x_ref| / onspan]

    Return:
        indexes (list) where y_i - y_i-1 >= deltaU_thld and y_i >= U_thld
    """
    indexes = []

    def flag(py, ny):
        res = ((ny - py) >= thld) if thld >= 0 else ((ny - py) <= thld)
        if lvl_thld is not None:
            res *= (ny >= lvl_thld) if thld >= 0 else (ny <= lvl_thld)
        return res
    indexes = list(np.argwhere(flag(Y[:-1], Y[1:])).flatten() + 1)

    if not onspan:
        return indexes

    findexes = []
    wfading = 0 if wfading is None else wfading
    if not 0 <= wfading <= 1:
        raise ValueError("wfading must be b/w 0 and 1")

    def weight(x, ref):
        return 1 - wfading * np.abs(x - ref) / onspan

    for i in indexes:

        ref_x = X[i-1]
        j = i - 1
        sl = []
        while j >= 0 and X[j] >= ref_x - onspan:
            sl.insert(0, j)
            j -= 1
        prevX, prevY = X[sl], Y[sl]
        prevW = weight(prevX, ref_x)

        ref_x = X[i]
        j = i
        sl = []
        while j < len(Y) and X[j] <= ref_x + onspan:
            sl.append(j)
            j += 1
        nextX, nextY = X[sl], Y[sl]
        nextW = weight(nextX, ref_x)

        prev_y = sum(prevY * prevW) / sum(prevW)
        next_y = sum(nextY * nextW) / sum(nextW)

        if flag(prev_y, next_y):
            findexes.append(i)

    return findexes
