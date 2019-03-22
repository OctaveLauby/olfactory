import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from olutils import display, read_params


from olfactory.detection import stepreg_bounds
from olfactory.containers.spans import FragGroup, Fragment


def reldiff(a, b):
    """Return relative difference"""
    return 2 * np.abs(b - a) / (a + b)


def fcolor(step, max_step):
    """Return color to use for step value (from blue to red)"""
    if np.isnan(step) or step > max_step:
        return "black"
    else:
        return (step / max_step, 0, 1 - step / max_step)  # rgb


def track(func):
    """Decorator for methods we want to store result

    Args:
        func (function): method return self of class with .copy mthd
    """
    def func_wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        self._calls[func.__name__] = {
            'return': res,
            'args': args,
            'kwargs': kwargs,
        }
        return res
    func_wrapper.__name__ = func.__name__
    func_wrapper.__doc__ = func.__doc__
    return func_wrapper


# *************************************************************************** #
# StepEstimator
# *************************************************************************** #

class StepEstimator(object):

    dft_fit_params = {'reg_factor': 2}
    dft_pred_params = {'npts_m': 10}
    dft_spred_params = {'step_reldiff_M': 0.1, 'npts_m': 10}

    def __init__(self, x, v=False):
        self.v = v
        self._x = Fragment(x)
        self._calls = {}

    def log(self, *args, **kwargs):
        kwargs['v'] = self.v
        display(*args, **kwargs)

    # ----------------------------------------------------------------------- #
    # Properties

    @property
    def x(self):
        return self._x

    @property
    def calls(self):
        return self._calls

    @property
    def frags(self):
        return self.calls.get('fit', {'return': None})['return']

    @property
    def pred(self):
        return self.calls.get('predict', {'return': None})['return']

    @property
    def dpred(self):
        return self.calls.get('dummy_predict', {'return': None})['return']

    @property
    def spred(self):
        return self.calls.get('smart_predict', {'return': None})['return']

    # ----------------------------------------------------------------------- #
    # Computation

    @track
    def fit(self, **params):
        """Build step-consistent fragments from x ticks"""
        return StepEstimator.xfit(self.x, **params)

    @track
    def predict(self, **params):
        """Predict most likely step on the entire span"""
        if self.frags is None:
            raise Exception("Must fit before prediction")
        params = read_params(params, StepEstimator.dft_pred_params)
        weights = np.array([
            frag.n_pts**2 for frag in self.frags
            if frag.n_pts >= params.npts_m
        ])
        steps = np.array([
            frag.step for frag in self.frags
            if frag.n_pts >= params.npts_m
        ])
        if len(weights) == 0:
            return np.nan
        return np.sum(weights * steps) / np.sum(weights)

    @track
    def dummy_predict(self):
        """Return flat step calculation (span / n_pts)"""
        return self.x.step

    @track
    def smart_predict(self, **params):
        """Predict most likely step on frags of smart segmentation"""
        if self.frags is None:
            raise Exception("Must fit before smart-prediction")
        params = read_params(params, StepEstimator.dft_spred_params)
        groups = []
        group = FragGroup()
        for frag in self.frags:
            if frag.n_pts < params.npts_m:
                group.append(frag, label="minor")
            elif not group.has_major():
                group.append(frag, label="major")
            elif reldiff(frag.step, group.step) > params.step_reldiff_M:
                groups.append(group)
                group = FragGroup(majors=[frag])
            else:
                group.append(frag)
        groups.append(group)
        return groups

    def compute(self, **params):
        """Launch a full computation (fit and predictions)

        Args:
            x (list)        : list of ticks
            **params   :
                fit_params      @see StepComputation.fit
                pred_params     @see StepComputation.fit
                spred_params    @see StepComputation.fit

        Return:
            list of Fragment instances where a frag has a consistent step
        """
        # Read params
        params = read_params(params, {
            'fit_params': {},
            'pred_params': {},
            'spred_params': {},
        })
        fit_params = read_params(
            params.fit_params, StepEstimator.dft_fit_params
        )
        pred_params = read_params(
            params.pred_params, StepEstimator.dft_pred_params
        )
        spred_params = read_params(
            params.spred_params, StepEstimator.dft_spred_params
        )

        self.log("> Fit with params %s" % fit_params)
        self.fit(**fit_params)

        pred = self.dummy_predict()
        pred = pred if np.isnan(pred) else timedelta(seconds=pred)
        self.log("> Dummy prediction:", pred)

        pred = self.predict(**pred_params)
        pred = pred if np.isnan(pred) else timedelta(seconds=pred)
        self.log("> Predicted step with %s: %s" % (pred_params, pred))
        self.log("> Computed fragments:")
        for frag in self.frags:
            self.log("\t-", frag)

        # Smart prediction
        self.log("> Smart prediction with params %s" % spred_params)
        spred = self.smart_predict(**spred_params)
        for group in spred:
            self.log("\t- %s" % group)

        return self

    @staticmethod
    def xfit(x, **params):
        """Compute step-consistent fragments from x ticks

        Vocabulary:
            Tick span           tspan_i         x[i] - x[i-1]

        Args:
            x (list)            : list of ticks
            reg_factor (float)  : max factor b/w tspan_i and apparent step
                                  to consider tspan_i as regular

        Return:
            list of Fragment instances where a frag has a consistent step
        """
        params = read_params(params, StepEstimator.dft_fit_params)
        frag = Fragment(x)

        if frag.n_pts in [0, 1]:
            return []
        elif frag.n_pts == 2:
            return [frag]

        indexes_forsplit = stepreg_bounds(
            frag,
            bot_thld=frag.step / params.reg_factor,
            top_thld=frag.step * params.reg_factor,
        )
        if not indexes_forsplit:
            return [frag]
        frags = frag.split(indexes_forsplit, share=True)
        return sum([StepEstimator.xfit(f, **params) for f in frags], [])

    # ----------------------------------------------------------------------- #
    # Utils

    def _plot(self, case, requires, **params):
        """Plot fit frags or smart prediction groups regarding case"""
        spans = getattr(self, case)
        if spans is None:
            raise Exception("Must %s before %s plot" % (requires, case))
        return StepEstimator.plot_spans(spans, **params)

    def plot_frags(self, **params):
        """Plot fit frags

        Args:
            **params: @see StepEstimator.plot_spans
        """
        return self._plot(
            case="frags", requires="run fit", **params
        )

    def plot_spred(self, **params):
        """Plot smart prediction groups

        Args:
            **params: @see StepEstimator.plot_spans
        """
        return self._plot(
            case="spred", requires="run smart_prediction", **params
        )

    @staticmethod
    def plot_spans(spans, wlabel=True, max_step=None, xunit="sec",
                   curve=None):
        """Plot list of spans

        Args:
            spans (list)        : Fragment or FragGroup instances
            wlabel (bool)       : smart display of span labels
            max_step (int)      : Max acceptable step (dft is max step)
            xunit (str)         : x-axis unit (sec, min, day, year, dt, td)
            curve (ECurve)      : add curve plot to spans plot
        """
        if curve:
            label = "label" if wlabel else None
            curve.plot(xunit=xunit, label=label)

        if max_step is None:
            max_step = max([
                elem.step
                for elem in spans
                if elem.step != np.inf and not np.isnan(elem.step)
            ])
        labels = set()
        sorted_elems = sorted(spans, key=lambda elem: elem.step)
        for elem in sorted_elems:
            color = fcolor(elem.step, max_step)
            label = (
                "Gap or no-span"
                if color == "black"
                else "Step=%s" % timedelta(seconds=int(elem.step))
            )
            if (not wlabel) or label in labels:
                label = None
            else:
                labels.add(label)
            elem.plot(label=label, color=color, xunit=xunit)
        if wlabel:
            plt.legend()
