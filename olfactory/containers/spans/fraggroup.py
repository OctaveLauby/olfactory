import numpy as np
from datetime import timedelta

from olutils import float2dt
from .span import SpanBase


class FragGroup(SpanBase):
    """Container for list of Fragments for consistent step computation"""

    def __init__(self, majors=[], minors=[]):
        """Init group of fragments

        Args:
            majors (Fragment-list): main frags for step computation
            minors (Fragment-list): ignored frags for step computation
        """
        self._frags = []
        self._major_frags = []
        self._minor_frags = []

        for frag in majors:
            self.append(frag, label="major")
        for frag in minors:
            self.append(frag, label="minor")

    # ----------------------------------------------------------------------- #
    # Properties

    @property
    def x(self):
        try:
            return np.concatenate(self.frags)
        except ValueError:
            return np.array([])

    @property
    def frags(self):
        return self._frags

    @property
    def step(self):
        """Computed step"""
        if not self.has_major():
            return np.nan
        weights = np.array([frag.n_pts for frag in self._major_frags])
        steps = np.array([frag.step for frag in self._major_frags])
        return np.sum(weights * steps) / np.sum(weights)

    # ----------------------------------------------------------------------- #
    # Frag like properties

    @property
    def fst(self):
        if self.is_empty():
            return np.nan
        return self.frags[0].fst

    @property
    def lst(self):
        if self.is_empty():
            return np.nan
        return self.frags[-1].lst

    @property
    def n_pts(self):
        return len(set(self.x))

    @property
    def span(self):
        return self.lst - self.fst

    # ----------------------------------------------------------------------- #
    # Content management

    def append(self, frag, label="major"):
        """Add a frag given its label (dft is major)"""
        self._frags.append(frag)
        if label == "major":
            self._major_frags.append(frag)
        else:
            self._minor_frags.append(frag)

    def is_empty(self):
        """Return whether instance contains at least one frag"""
        return len(self.frags) == 0

    def has_major(self):
        """Return whether instance contains at least one major frag"""
        return len(self._major_frags) > 0

    # ----------------------------------------------------------------------- #
    # Utils

    def __iter__(self):
        return self.frags.__iter__()

    def __getitem__(self, index):
        return self.frags.__getitem__(index)

    def __setitem__(self, index, value):
        return self.frags.__setitem__(index, value)

    def __len__(self):
        return self.frags.__len__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.is_empty():
            return "<Empty Group>"
        label = "Group" if self.has_major() else "Inconsistent Group"
        step = (
            timedelta(seconds=int(self.step))
            if self.has_major() else self.step
        )
        return (
            "<{label} : {n_frags} frags ({n_majors} majors)"
            " | {fst} to {lst}"
            " | {n_pts} pts"
            " | step={step}>"
        ).format(
            label=label,
            n_frags=len(self.frags),
            n_majors=len(self._major_frags),
            fst=float2dt(self.fst),
            lst=float2dt(self.lst),
            n_pts=self.n_pts,
            step=step,
        )
