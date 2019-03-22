import numpy as np

from .tools import group_consecutives


def reg_bounds(X, bot_thld=0, top_thld=np.inf):
    """Detect regularity and return boundaries for consistent splitting

    Args:
        X (np.ndarray)  : array to work with
        bot_thld (float): bot threshold for x
        top_thld (float): top threshold for x

    Returns:
        (int-list)
            indexes where to split X so each span has x-values with either
                with bot_thld <= x <= top_thld
                with x < bot_thld
                with top_thld < x

    Example:
        >> values = np.array([0, 1, 2, 3, 4, 1, 0, 5, 6])
        >> detection.reg_bounds(values, bot_thld=2, top_thld=4)
        [2, 5, 7]
    """
    if bot_thld > top_thld:
        raise ValueError("bot_thld must be smaller or equal to top_thld")

    if len(X) == 0:
        return []

    indexes_forsplit = set()

    # Small steps
    if bot_thld > 0:
        indexes = np.where(X < bot_thld)[0]
        igroups = group_consecutives(indexes)
        for igroup in igroups:
            indexes_forsplit.add(igroup[0])
            indexes_forsplit.add(igroup[-1] + 1)

    # Wide steps
    if top_thld != np.inf:
        indexes = np.where(X > top_thld)[0]
        igroups = group_consecutives(indexes)
        for igroup in igroups:
            indexes_forsplit.add(igroup[0])
            indexes_forsplit.add(igroup[-1] + 1)

    try:
        indexes_forsplit.remove(0)
    except KeyError:
        pass
    try:
        indexes_forsplit.remove(len(X))
    except KeyError:
        pass
    return sorted(indexes_forsplit)


def stepreg_bounds(X, *args, **kwargs):
    """Detect step regularity and return boundaries for consistent splitting

    Args:
        X (np.ndarray)  : array to work with
        bot_thld (float): bot threshold for step
        top_thld (float): top threshold for step

    Returns:
        (int-list)
            indexes where to split X so each span has x-diffs with either:
                with bot_thld <= x_i+1 - x_i <= top_thld
                with x_i+1 - x_i < bot_thld
                with top_thld < x_i+1 - x_i
    """
    if len(X) <= 1:
        return []
    steps = np.diff(X)  # step_i refers to X[i], X[i+1]
    return reg_bounds(steps, *args, **kwargs)
