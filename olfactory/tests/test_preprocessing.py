import numpy as np
import pytest

from olfactory import preprocessing


def test_pipe_filter():

    class Elem(dict):

        def __init__(self, dic):
            super().__init__(dic)
            self.id = dic['id']
            self.flags = set()

        def __getattr__(self, attribute):
            try:
                return self[attribute]
            except KeyError:
                raise AttributeError(
                    "%s is neither a param attribute nor a field" % attribute
                ) from None

        def __eq__(self, other):
            if isinstance(other, int):
                return self.id == other
            else:
                super().__eq__(other)

    def block(func, flag=None):
        def nfunc(elem):
            if func(elem):
                if flag:
                    elem.flags.add(flag)
                return True
            return False
        return nfunc

    elems = [
        Elem({'id': 0, 'active': True, 'hidden_key': "haveit"}),
        Elem({'id': 1, 'active': True, 'hidden_key': "haveit"}),
        Elem({'id': 2, 'active': False, 'hidden_key': "haveit"}),
        Elem({'id': 3, 'active': True}),
        Elem({'id': 4, 'active': True}),
        Elem({'id': 5, 'active': False}),
    ]

    # Filter inactive
    pipe = [block(lambda e: not e.active, "inactive")]
    assert preprocessing.pipe_filter(elems, pipe) == [0, 1, 3, 4]
    for elem in elems:
        if elem.id in [2, 5]:
            assert "inactive" in elem.flags

    # Filter inactive and with hidden key
    pipe = [
        block(lambda e: not e.active, "inactive"),
        block(lambda e: 'hidden_key' not in e, "nohiddenkey")
    ]
    assert preprocessing.pipe_filter(elems, pipe) == [0, 1]
    for elem in elems:
        if elem.id in [3, 4]:
            assert "nohiddenkey" in elem.flags
        if elem.id in [5]:  # Fast break so no double flag
            assert "nohiddenkey" not in elem.flags

    assert preprocessing.pipe_filter(elems, pipe, fastbreak=False) == [0, 1]
    for elem in elems:
        if elem.id in [3, 4, 5]:
            assert "nohiddenkey" in elem.flags

    # Filter elems with flag
    pipe = [block(lambda e: e.flags)]
    kept, rm = preprocessing.pipe_filter(elems, pipe, w_rm=True)
    assert kept == [0, 1]
    assert rm == [2, 3, 4, 5]
    flags = preprocessing.pipe_filter(elems, pipe, r_flags=True)
    assert flags == [False, False, True, True, True, True]


def test_xy_merge():
    xy1 = ([1, 2, 3, 4], [10, 20, 30, 4])
    xy2 = ([0, 3.5, 4, 5, 6], [0, 0.35, 4, 0.5, 0.6])
    assert preprocessing.xy_merge(xy1, xy2) == (
        [0, 1, 2, 3, 3.5, 4, 4, 5, 6],
        [0, 10, 20, 30, 0.35, 4, 4, 0.5, 0.6]
    )

    x1 = np.array([1, 2, 4, 5, 8, 9])
    y1 = 10 * x1
    x2 = np.array([0, 2, 3, 10])
    y2 = 10 * x2
    assert preprocessing.xy_merge((x1, y1), (x2, y2)) == (
        [0, 1, 2, 2, 3, 4, 5, 8, 9, 10],
        [0, 10, 20, 20, 30, 40, 50, 80, 90, 100]
    )

    with pytest.raises(ValueError):
        preprocessing.xy_merge(([1], [1]), ([1], [10]))

    res = preprocessing.xy_merge(
        ([1], [1]), ([1], [10]), raise_err=False, twin_mean=True
    )
    assert res == ([1], [5.5])
    res = preprocessing.xy_merge(
        ([1], [1]), ([1], [10]), raise_err=False, twin_mean=False
    )
    assert res == ([1, 1], [1, 10])


def test_resample():

    X = np.array([30, 50,  85, 90])
    Y = np.array([.3, .5, .85, .9])

    assert preprocessing.resample(X, Y, step=10) == (
        [30, 40, 50, 60, 70, 80, 90],
        [.3, .4, .5, .6, .7, .8, .9]
    )
    assert preprocessing.resample(X, Y, step=30) == (
        [30, 60, 90],
        [.3, .6, .9]
    )
    assert preprocessing.resample(X, Y, step=40) == (
        [30, 70],
        [.3, .7]
    )
    assert preprocessing.resample(X, Y, n_pts=7) == (
        [30, 40, 50, 60, 70, 80, 90],
        [.3, .4, .5, .6, .7, .8, .9]
        )

    with pytest.raises(ValueError):
        preprocessing.resample(X, Y)

    with pytest.raises(ValueError):
        preprocessing.resample(X, Y, step=5, n_pts=5)

    with pytest.raises(ValueError):
        preprocessing.resample(X, Y, n_pts=1)


def test_rescale():

    # Classic use
    a = np.array([3, 10, 0, 5, 9])
    np.testing.assert_equal(preprocessing.rescale(a), [0.3, 1, 0, 0.5, 0.9])
    np.testing.assert_equal(
        preprocessing.rescale(a, bounds=(-20, 20)),
        [-8, 20, -20, 0, 16]
    )
    np.testing.assert_equal(
        preprocessing.rescale(a, batch=2),
        [1.5 / 8, 8.5 / 8, -1.5 / 8, 3.5 / 8, 7.5 / 8]
    )
    np.testing.assert_equal(
        preprocessing.rescale(a, bounds=(0, 8), batch=2),
        [1.5, 8.5, -1.5, 3.5, 7.5]
    )

    # Using is_sorted
    s_a = np.sort(a)
    np.testing.assert_equal(
        preprocessing.rescale(s_a, is_sorted=True),
        [0, 0.3, 0.5, 0.9, 1]
    )
    np.testing.assert_equal(
        preprocessing.rescale(np.flip(s_a, axis=0), is_sorted=True, is_reversed=True),
        [1, 0.9, 0.5, 0.3, 0]
    )
    np.testing.assert_equal(
        preprocessing.rescale(np.flip(s_a, axis=0), is_sorted=True),
        [0, 0.1, 0.5, 0.7, 1]
    )

    # Using is_sorted when it is not
    np.testing.assert_equal(
        preprocessing.rescale(a, bounds=(0, 6), is_sorted=True),
        [0, 7, -3, 2, 6]
    )
    np.testing.assert_equal(
        preprocessing.rescale(np.flip(a, axis=0), bounds=(0, 6), is_sorted=True),
        [0, 4, 9, -1, 6]
    )

    # Batch greater than len(a)
    with pytest.raises(ValueError):
        preprocessing.rescale(a, batch=5)
    np.testing.assert_equal(
        preprocessing.rescale(a, batch=5, bounds=(0, 8), adapt=True),
        [1.5, 8.5, -1.5, 3.5, 7.5]
    )
