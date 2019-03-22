import numpy as np
from datetime import timedelta

from olutils import float2dt
from .span import SpanBase


class Fragment(np.ndarray, SpanBase):
    """Fragment of series descriptor"""

    def __new__(cls, a):
        obj = np.asarray(a).astype(float).view(cls)
        return obj

    def __init__(self, x):
        """Init a fragment instance of x series

        Args:
            x (np.array)
        """
        self._is_empty = len(x) == 0

    # ----------------------------------------------------------------------- #
    # Quick access

    @property
    def is_empty(self):
        return self._is_empty

    @property
    def fst(self):
        return self[0] if not self.is_empty else np.nan

    @property
    def lst(self):
        return self[-1] if not self.is_empty else np.nan

    @property
    def n_pts(self):
        return len(self)

    @property
    def span(self):
        return self.lst - self.fst

    @property
    def step(self):
        with np.errstate(divide='raise'):
            try:
                return self.span / (self.n_pts-1)
            except (ZeroDivisionError, FloatingPointError):
                return np.nan

    # ----------------------------------------------------------------------- #
    # Processing

    def split(self, indexes, share=False):
        result = []

        lst_i = 0
        for i in sorted(set(indexes)):
            if i == 0:
                continue
            result.append(Fragment(self[lst_i:(i+share)]))
            lst_i = i
        if lst_i != self.n_pts - 1:
            result.append(Fragment(self[lst_i:]))
        return result

    # ----------------------------------------------------------------------- #
    # Utils

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.is_empty:
            return "<Empty Frag>"
        return "<Frag | {fst} to {lst} | {n_pts} pts | step={step}>".format(
            fst=float2dt(self.fst),
            lst=float2dt(self.lst),
            n_pts=self.n_pts,
            step=timedelta(seconds=int(self.step)),
        )
