import functools
import numpy as np
from prettytable import PrettyTable
from time import time as clock


class Call(object):

    def __init__(self, func, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        t = clock()
        self.output = func(*args, **kwargs)
        self.time = clock() - t


class CallTracker(object):
    """Func calls tracking class, goes along trackedfunc decorator"""

    trackers = {}

    def __init__(self, func):
        """Init a tracker (to use for a given function)

        Tracker can be find at CallTracker.trackers[func.__qualname__]
        """

        # Add func to tracked functions
        funcname = func.__qualname__
        cls = self.__class__
        # # Should be checked but not convenient when using notebook
        # if funcname in cls.trackers:
        #     raise ValueError(
        #         "Can't track a function already tracked: %s" % funcname
        #     )
        cls.trackers[funcname] = self

        # Init instance
        self.func = func
        self._n_calls = 0
        self._exc_time = 0

    @property
    def n_calls(self):
        """Number of calls to tracked functions"""
        return self._n_calls

    @property
    def exc_time(self):
        """Total running time of tracked function"""
        return self._exc_time

    def mean_exc_time(self):
        """Return mean execution time by call"""
        try:
            return self.exc_time / self.n_calls
        except ZeroDivisionError:
            return np.nan

    def call(self, *args, **kwargs):
        """Call func and track execution time"""
        call = Call(self.func, *args, **kwargs)
        self._n_calls += 1
        self._exc_time += call.time
        return call.output

    def reset(self):
        """Reset call-results, n_calls and total exc time"""
        self._n_calls = 0
        self._exc_time = 0

    # *********************************************************************** #
    # Class method

    @classmethod
    def display_stats(cls, w_uncalled=False, sortby=None, reverse=False):
        """Return pretty table of called function stats

        Args:
            w_uncalled (bool): display uncalled tracked functions below table
            sortby (str): sort res table by given header
                should be in "function", "n_calls", "exc_time", "mean_exc_time"
            reverse (bool): reverse sorting
        """
        table = PrettyTable()
        table.field_names = ["function", "n_calls", "exc_time", "mean_exc_time"]

        # Gather called / uncalled
        called = []
        uncalled_funcs = []
        for funcname, tracker in cls.trackers.items():
            if tracker.n_calls > 0:
                called.append((funcname, tracker))
            else:
                uncalled_funcs.append(funcname)

        # Sort
        if sortby is None:
            pass
        elif sortby in ["name", "funcname", "function"]:
            called.sort(key=lambda item: item[0], reverse=reverse)
        elif sortby == "n_calls":
            called.sort(key=lambda item: item[1].n_calls, reverse=reverse)
        elif sortby == "exc_time":
            called.sort(key=lambda item: item[1].exc_time, reverse=reverse)
        elif sortby == "mean_exc_time":
            called.sort(key=lambda item: item[1].mean_exc_time(), reverse=reverse)
        else:
            raise ValueError(
                "sortby unknown value %s, must be picked in %s"
                % (sortby, table.field_names)
            )

        # Fills table
        for funcname, tracker in called:
            row = [
                funcname,
                tracker.n_calls,
                "%.3e" % tracker.exc_time,
                "%.3e" % tracker.mean_exc_time(),
            ]
            table.add_row(row)

        # Display
        print(table)
        if w_uncalled:
            print("*", ", ".join(uncalled_funcs))

    @classmethod
    def reset_all(cls):
        """Reset trackers stats"""
        for tracker in cls.trackers.values():
            tracker.reset()


def trackedfunc(func):
    """Decorator to track calls to func.

    In a class, decorator must be bellow @clsmethod or @staticmethod decorator
    """

    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):
        return func_wrapper.tracker.call(*args, **kwargs)

    func_wrapper.tracker = CallTracker(func)
    func_wrapper.__doc__ = func.__doc__ if func.__doc__ else ""
    func_wrapper.__doc__ += "\n@about: tracked function\n"

    return func_wrapper
