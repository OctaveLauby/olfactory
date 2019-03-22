"""Function to smooth curves

One can compare result with simple example:
    ```
    import matplotlib.pyplot as plot
    import numpy as np

    Y = np.array([1, 2, 6, 7, 6, 8, 7, 2, 1, 3, 4, 2, 0, 1, 2, 1, 2, -2, -5])
    X = np.array([i for i in range(len(Y))])

    plt.plot(X, Y, label="root")
    plt.plot(X, window_smooth(X, Y, 11, wfading=0.9), label="window_smooth")
    plt.plot(X, savgol_smooth(X, Y, 11, polyorder=3), label="savgol_smooth")
    plt.legend()
    plt.show()
    ```
"""
import numpy as np
from scipy.signal import savgol_filter


def x_to_i_window(X, window):
    """Return window on index given window on X assuming X is regular

    Args:
        X (n-numpy.ndarray) : regular ticks
        window (float)      : window size on x

    Return:
        (odd-int) window size on index

    Raises:
        ValueError if X not regular
    """
    if not np.all(np.isclose(np.diff(np.diff(X)), 0)):
        raise ValueError("X must have regular step for this method")
    step = (X[-1] - X[0]) / (len(X) - 1)
    return int((window + step) / step)


def _odd_window(X, window, xwindow=False):
    """Return biggest odd-window on index"""
    if xwindow:
        window = x_to_i_window(X, window)
    return min(2 * (window // 2) + 1, 2 * ((len(X) + 1) // 2) - 1)


def savgol_smooth(X, Y, window, polyorder=3, xwindow=False, **kwargs):
    """Run savgol filter to smooth y

    Pros & Cons:
        ++ Keep shape of Y (like border-drops)
        -- Creates bumps on irregularities

    Args:
        X (n-numpy.ndarray) : xticks
        Y (n-numpy.ndarray) : associated values
        window (int)        : size of filter window (on index or x)
            if xwindow is True, requires X regular
            else: requires window odd && 0 < window
        xwindow (bool)      : if window is given on x, not on index
        polyorder (int)     : order of polynomial used to fit the samples
        **kwargs: @see scipy.signal.savgol_filter
            polyorder (3 works fine)

    Return:
        (n-numpy.ndarray) smoothen y
    """
    if len(X) < 2:
        return Y
    window = _odd_window(X, window, xwindow=xwindow)
    polyorder = min(window-1, polyorder)
    return savgol_filter(Y, window, polyorder, **kwargs)


def window_smooth(X, Y, window, wfading=None, xwindow=False):
    """Smooth y using slicing window

    Pros & Cons:
        ++ Shaving fluctuations
        -- Erase border fluctuations (bad when studying drops)

    Args:
        X (n-numpy.ndarray) : xticks
        Y (n-numpy.ndarray) : associated values
        window (int)        : size of filter window (on index or x)
            if xwindow is True, requires window < span(X) && X regular
            else: requires window odd && 0 < window < len(Y)
        xwindow (bool)      : if window is given on x, not on index
        wfading (float)     : when computing y, apply a weight to surrounding-y
            for yi : weight_yk = (
                    1 - wfading * (|xk - xi| / max_|xj-xi|_'j in i window')
                )

    Return:
        (n-numpy.ndarray) smoothen y
    """
    if wfading is not None and not 0 <= wfading <= 1:
        raise ValueError("wfading must be b/w 0 and 1")

    window = _odd_window(X, window, xwindow=xwindow)
    fading_weight = wfading if window > 1 else None

    halfw = window // 2
    new_y = []
    for i, x in enumerate(X, 0):
        s = slice(max(0, i-halfw), i+halfw+1, 1)
        wY = Y[s]
        if fading_weight:
            distances = np.abs(X[s] - x)
            fading_ratio = wfading / max(distances)
            weights = 1 - distances * fading_ratio
        else:
            weights = np.ones(len(wY))
        new_y.append(
            sum(wY * weights)
            / sum(weights)
        )
    return np.array(new_y)
