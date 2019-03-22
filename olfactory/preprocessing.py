import numpy as np
from bisect import bisect_left


def pipe_filter(elems, pipe, fastbreak=True, w_rm=False, r_flags=False):
    """Return element given a filter pipe

    Args:
        elems (iterable) : list to filter from
        pipe (list of callable) : a callable returns True on element to remove
        fastbreak (bool)        : stop pipe when a callable return True
            set it to False so all callable are called on object
        w_rm (bool)             : also return list of removed elements
        r_flags (bool)          : return flags (True=ToRM ; False=ToKeep)

    Return:
        (list) if fastbreak else (2-list-tuple)
    """
    flags = []
    for elem in elems:
        flag = False
        if fastbreak:
            for block in pipe:
                flag = block(elem)
                if flag:
                    break
        else:
            flag = sum([block(elem) for block in pipe])
        flags.append(flag)

    if r_flags:
        return flags

    rm_list = []
    select_list = []
    for flag, elem in zip(flags, elems):
        if flag:
            rm_list.append(elem)
        else:
            select_list.append(elem)

    if w_rm:
        return select_list, rm_list
    else:
        return select_list


def resample(X, Y, n_pts=None, step=None):
    """Resample X with regular ticks (given n_pts or step)

    When using step, X[-1] wont be included unless X[-1] = X[0] + k * step

    Examples:
        >> resample([0, 2, 4], [1, 3, 5], step=1)
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
        >> resample([0, 2, 4], [1, 3, 5], step=3)
        ([0, 3], [1, 4])
    """
    newX = []
    if not n_pts and not step:
        raise ValueError("n_pts or step argument is required")
    elif n_pts and step:
        raise ValueError("n_pts and step arguments can't be both given")
    elif n_pts:
        if len(X) <= 1:
            return X, Y
        elif n_pts <= 1:
            raise ValueError("n_pts must be greater than 1")
        newX = [
            X[0] + i * (X[-1] - X[0]) / (n_pts - 1)
            for i in range(n_pts)
        ]
    elif step:
        tick = X[0]
        while tick <= X[-1]:
            newX.append(tick)
            tick += step

    newY = []
    for tick in newX:
        k = bisect_left(X, tick)
        if k is 0:
            yval = Y[0]
        elif k == len(X):
            yval = Y[-1]
        else:
            a, b = k-1, k
            yval = (
                ((X[b] - tick) * Y[a] + (tick - X[a]) * Y[b])
                / (X[b] - X[a])
            )
        newY.append(yval)

    return newX, newY


def rescale(a, bounds=(0, 1), batch=None, adapt=False, is_sorted=False, is_reversed=False):
    """Return rescaled array that fits between bounds

    Args:
        a (n-numpy.ndarray) : array to be rescaled
        bounds (2-tuple)    : bounds for rescaled array
        batch (int)         : apply bounds on <batch> values
            For instance, if batch=2, mean(2 lowest vals) = low_bound
        adapt (bool)        : ensure batch is not greater than len // 2
        is_sorted (bool)    : consider a sorted
            meaning lowest / greatest values are taken from start / end of a

    Examples:


    Return:
        (n-numpy.ndarray): rescaled array
    """

    if len(bounds) != 2 or (bounds[1] - bounds[0]) <= 0:
        raise ValueError("bounds must be like (a, b) where b > a")

    if len(a) == 1:
        return np.array([bounds[0]])

    a_s = a if is_sorted else np.sort(a)
    a_min, a_max = a_s[0], a_s[-1]
    if batch:
        if adapt:
            batch = min(len(a) // 2, batch)
        if batch >= len(a):
            raise ValueError(
                "batch must be strictly lower than array length:"
                "batch=%s >= %s=len(a)"
                % (batch, len(a))
            )
        a_min = np.mean(a_s[:batch])
        a_max = np.mean(a_s[-batch:])
    if is_reversed:
        a_min, a_max = a_max, a_min

    if a_min == a_max:
        return (a - a_min)
    else:
        span = bounds[1] - bounds[0]
        return span * (a - a_min) / (a_max - a_min) + bounds[0]


def xy_merge(xy1, xy2, raise_err=True, twin_mean=False):
    """Merge (x1, y1) with (x2, y2)

    Args:
        xy1 (tuple of n-list)   : (X1, Y1) where X is sorted with no duplicate
        xy2 (tuple of p-list)   : (X2, Y2) where X is sorted with no duplicate
        raise_err (bool): raise error on following cases
            x1 = x2 but y1 != y2
        twin_mean (bool): compute mean of y when x1 = x2, else keep duplicate

    Returns:
        (X, Y) where
            X is sorted
            X contains X1 and X2
            Y contains Y1 and Y2
            Y_i = (
                Y1_k if X_i = X1_k
                or Y2_k if X_i = X1_k
            )
    """
    # TODO : pretty sure it can't be done with numpy
    x = []
    y = []

    iter1 = iter(zip(*xy1))
    iter2 = iter(zip(*xy2))
    x1, y1 = next(iter1)
    x2, y2 = next(iter2)
    cont = True

    def cnext(iterable):
        try:
            return (True, *next(iterable))
        except StopIteration:
            return False, None, None

    while cont:
        if x1 < x2:
            x.append(x1)
            y.append(y1)
            cont, x1, y1 = cnext(iter1)
        elif x2 < x1:
            x.append(x2)
            y.append(y2)
            cont, x2, y2 = cnext(iter2)
        else:
            if raise_err:
                if y1 != y2:
                    raise ValueError(
                        "Trying to merge xy with diff on y at x=%s (%s!=%s)"
                        % (x1, y1, y2)
                    )
            if twin_mean:
                x.append(x1)
                y.append(np.mean([y1, y2]))
            else:
                x.append(x1)
                y.append(y1)
                x.append(x2)
                y.append(y2)
            cont1, x1, y1 = cnext(iter1)
            cont2, x2, y2 = cnext(iter2)
            cont = cont1 and cont2
    if x1:
        x.append(x1)
        y.append(y1)
    if x2:
        x.append(x2)
        y.append(y2)
    for x1, y1 in iter1:
        x.append(x1)
        y.append(y1)
    for x2, y2 in iter2:
        x.append(x2)
        y.append(y2)

    return x, y
