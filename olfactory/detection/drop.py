import numpy as np
import re
from collections import defaultdict

from olfactory.browsing import sliding_window


# --------------------------------------------------------------------------- #
# Utils


def create_drop_lvlmarker(yi=None, gi=None, hi=None):
    """Return function to flag (y, h, g) given thresholds

    Args:
        yi (float)  : value threshold (None for no thld)
        gi (float)  : gradient threshold (None for no thld)
        hi (float)  : grad2 threshold (None for no thld)

    Return:
        (callable)
            y, g, h -> False                            if all None
            y, g, h -> y < Yi and g < Gi and h < Hi     if one is not None
                Yi = yi if yi is not None else np.inf
                Gi = gi if gi is not None else np.inf
                Hi = hi if hi is not None else np.inf
            where y, g, h is float_tuple or n-np.ndarray_tuple
            when y, g and h are arrays, return 1d-array of boolean
(float or n-np.ndarray)
    """
    # All None
    if yi is None and hi is None and gi is None:
        return lambda y, g, h: False

    # 2 None
    elif yi is None and gi is None:
        return lambda y, g, h: h < hi
    elif yi is None and hi is None:
        return lambda y, g, h: g < gi
    elif gi is None and hi is None:
        return lambda y, g, h: y < yi

    # 1 None
    elif yi is None:
        return lambda y, g, h: (h < hi) * (g < gi)
    elif hi is None:
        return lambda y, g, h: (y < yi) * (g < gi)
    elif gi is None:
        return lambda y, g, h: (y < yi) * (h < hi)

    else:
        return lambda y, g, h: (y < yi) * (g < gi) * (h < hi)


def create_drop_marker(**kwargs):
    """Return function to flag (y, h, g) given level thresholds

    Args:
        **kwargs: (name_frmt: r'^[yhg]\\d$', val: float)
            Define each level i of thresholds:
                yi (float)  : value threshold (None for no thld)
                gi (float)  : gradient threshold (None for no thld)
                hi (float)  : grad2 threshold (None for no thld)
            flag raised if (y<yi or g<gi or h<hi) or (y<yk or g<gk or h<hk) ..

    Example:
        >> func = create_drop_marker(y0=10, y1=17, g1=2, h2=0)
        >> func(8, 3, 1)
            True
        >> func(11, 3, 1)
            False
        >> func(11, 3, 0)
            False
        >> func(11, 3, -1)
            True
        >> func(11, 1, 1)
            True
        >> func(
        >      np.array([8, 11, 11, 11, 11]),
        >      np.array([3, 3, 3, 3, 1]),
        >      np.array([1, 1, 0, -1, 1])
        >  )
            np.array([True, False, False, True, True])

    Return:
        (callable)
            function(y, g, h)
                (bool) if y, g, h are floats
                    True drop detected
                    False if not
                (n-bool-np.ndarray) if y, g and h are arrays
                    True where drop is detected
                    False where no drop detected
    """
    # Pack parameters by level_index for create_drop_lvlmarker
    if not all([bool(re.match(r"^[yhg]\d$", param)) for param in kwargs]):
        raise ValueError(
            "Params must match r'^[yhg]\\d$', got %s" % list(kwargs.keys())
        )
    lvlthlds = defaultdict(dict)
    for param, value in kwargs.items():
        letter, level_index = param[0], param[1]
        lvlthlds[level_index][letter + "i"] = value

    # Build level functions
    lvlfuncs = []
    for thlds in lvlthlds.values():
        lvlfuncs.append(create_drop_lvlmarker(**thlds))

    # Build result
    def drop_marker(y, g, h):
        """Return flag(s) for drop detection

        Args:
            y (float or n-np.ndarray): value(s)
            g (float or n-np.ndarray): associated gradient
            h (float or n-np.ndarray): associated gradient2

        Return:
            (bool) if y, g, h are floats
                True drop detected
                False if not
            (n-bool-np.ndarray) if y, g and h are arrays
                True where drop is detected
                False where no drop detected
        """
        return np.sum(
            [lfunc(y, g, h) for lfunc in lvlfuncs], axis=0
        ).astype(bool)

    return drop_marker


# -------------------------------------------------------------------------- #
# Algorithms


def detect_drop(XYGH, window=1, crit_thld=1, output="flags", **flag_kwargs):
    """Return tick(s) where y-drop was detected

    Args:
        XYGH (ndarray-tuple): x-ticks, y-values, gradient, gradient2
        window (int)        : slicing window to look for flags
        crit_thld (int)     : nb of flags expected in window to raise detection
        output (str)        : required output (@see Return)
        **flag_kwargs: parameters to set flag function. 2 possibilities
            a. flag_kwargs are kwargs for create_drop_marker function
            b. flag_kwargs = {'drop_marker': drop_marker (callable)}
                where drop_marker takes (y, g, h) as args and return (bool)
                    True when drop detected
                    False if not

    Return:
        None if gradient could not be computed
        else output depends on output-arg:
            flags       return list of flags related        (n-bool-np.ndarray)
            indexes     return list of flag-indexes         (int-np.ndarray)
            all         return all x where flag is raised   (float-np.ndarray)
            fst         return fst x where flag is raised   (float or np.nan)
            lst         return lst x where flag is raised   (float or np.nan)
            ---         raise ValueError
    """
    # Read params
    X, Y, G, H = XYGH
    if G is None or H is None:
        return None

    drop_marker = flag_kwargs.get('drop_marker', None)
    if drop_marker is None:
        drop_marker = create_drop_marker(**flag_kwargs)
    flags = drop_marker(Y, G, H)

    if window < crit_thld:
        raise ValueError("crit_thld must be equal or greater than window")

    # Apply window / crit_thld
    if crit_thld > 1:
        border = np.zeros(window - 1, dtype=bool)
        eflags = np.concatenate((border, flags))
        flags = np.sum(sliding_window(eflags, window), -1) >= crit_thld

    # Build result
    if output in ["fst", "lst"]:
        index = 0 if output == "fst" else -1
        try:
            return X[flags][index]
        except IndexError:
            return np.nan
    elif output == "flags":
        return flags
    elif output == "indexes":
        return np.where(flags)[0]
    elif output == "all":
        return X[flags]
    else:
        raise ValueError("Unexpected 'output' arg %s" % output)
