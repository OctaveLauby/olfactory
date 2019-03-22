"""Collection of signal generators"""
import numpy as np
import scipy.stats as stats

DFT_SCALE_R = 0.2  # Dft ratio b/w noise-scale & noise-vals window


def _compute_scale(window, scale_r=None, scale=None):
    """Compute scale of noise given a window for noise values

    This computed value is by default computed so that it gives a nice
    distribution (small scale ratio)

    Args:
        window (float) : window for noise values
        scale_r (float): scale / window
        scale (float)  : force scale value (prevails on scale_r)

    Return:
        (float)
    """
    return (
        window * (DFT_SCALE_R if scale_r is None else scale_r)
        if scale is None
        else scale
    )


def truncnorm(size, center=0, window=1, **scale_params):
    """Return truncated values of normal distribution

    Args:
        size (int): number of pts
        center (float): center of distribution
        scale (float): width of truncation
        **scale_params:
            scale_r: ratio b/w noise-scale and noise-values window
            scale (float): noise-scale (prevails on scale_r)

    Return:
        (np.ndarray) normal distribution
    """
    scale = _compute_scale(window, **scale_params)
    return stats.truncnorm(
            - window / scale, window / scale, loc=center, scale=scale
    ).rvs(size)


def boundnorm(size, low=-1, high=1, center='middle', **scale_params):
    """Return values of normal distribution within bound

    Args:
        size (int): number of pts
        low (float): lowest value
        high (float): highest value
        center (float or str): center of distribution
            left>low / middle / right>high
        **scale_params:
            scale_r (float): ratio b/w noise-scale and noise-values window
                window = max(center-low, high-center)
            scale (float): noise-scale (prevails on scale_r)

    Return:
        (np.ndarray) normal distribution
    """
    if center == 'middle':
        center = (high + low) / 2
    elif center == 'left':
        center = low
    elif center == 'right':
        center = high
    elif not isinstance(center, (int, float)):
        raise ValueError(
            "Unknown center value %s: expect float or 'left'/'middle'/'right'"
            % center
        )

    assert low <= center <= high
    scale = _compute_scale(max(high-center, center-low), **scale_params)
    return stats.truncnorm(
            (low-center)/scale, (high-center)/scale, loc=center, scale=scale
    ).rvs(size)


def stepsignals(n_pts, centers, scales, flatten=False, shuffle=False):
    """Build a list of arrays where each correspond to a noisy step signal

    Args:
        n_pts (int or n-int-list): total nb of pts or nb of pts per step-signal
        centers (n-float-list): center of each step
        scales (float or n-float-list): scale of noise
        flatten (bool): flatten groups in single array
        shuffle (bool): shuffle output (if flattened)

    Return:
        (list or np.ndarray)
            if flatten: np.ndarray else list of step_signals
    """
    n_grps = len(centers)

    sizes = n_pts
    if isinstance(n_pts, int):
        if n_pts < n_grps:
            raise ValueError(
                "Can't build %s groups with %s pts" % (n_grps, n_pts)
            )
        sizes = np.ones(n_grps).astype(int)
        for index in np.random.randint(0, n_grps, n_pts - n_grps):
            sizes[index] += 1
        # print("> n_pts distribution:", sizes)

    if isinstance(scales, (int, float)):
        scales = [scales] * n_grps

    assert len(sizes) == n_grps
    assert len(scales) == n_grps

    grps = [
        truncnorm(size, center, scale)
        for center, scale, size in zip(centers, scales, sizes)
    ]

    if flatten is False:
        return grps
    pts = np.array([pt for gp in grps for pt in gp])
    if shuffle:
        return np.shuffle(pts)
    else:
        return pts
