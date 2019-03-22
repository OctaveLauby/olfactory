import matplotlib.pyplot as plt
from olutils import convert_ts

from olfactory.operations import isnull


class SpanBase(object):
    """Skeleton for span classes"""

    # TODO : implement a metaclass to build a real Skeleton
    # # This metaclass must ensure that both fst and lst are implemented

    @property
    def fst(self):
        raise NotImplementedError

    @property
    def lst(self):
        raise NotImplementedError

    @property
    def span(self):
        return self.lst - self.fst

    def plot(self, label=None, color=None, alpha=0.5, ts_unit=None,
             w_borders=True):
        """Plot span

        Args:
            label (str): label of span
            color (str): color of span
            alpha (str): alpha of span color
            ts_unit (str): specify unit to convert fst and lst
                in that case original fst and lst must be timestamps
            w_borders (bool): also plot borders of span (no alpha)
        """
        if isnull(self.fst) or isnull(self.lst):
            return
        if ts_unit:
            start, end = convert_ts([self.fst, self.lst], unit=ts_unit)
        else:
            start, end = self.fst, self.lst
        if w_borders:
            line = plt.axvline(start, c=color)
            color = line.get_color()
            plt.axvline(end, c=color)
        plt.axvspan(start, end, facecolor=color, alpha=alpha, label=label)

    def __repr__(self):
        return (
            "<%s from %s to %s>"
            % (self.__class__.__name__, self.fst, self.lst)
        )

    def __str__(self):
        return repr(self)


class Span(SpanBase):
    """Simple span

    Example:
        >> span = Span(10, 20)
        >> span.plot(label="span", color="red", ts_unit=xunit, w_borders=False)
        >> plt.xticks(rotation=45)
        >> plt.show()
    """

    def __init__(self, fst, lst):
        if fst > lst:
            raise ValueError("fst can't be greater than lst")
        self._fst = fst
        self._lst = lst

    @property
    def fst(self):
        return self._fst

    @property
    def lst(self):
        return self._lst
