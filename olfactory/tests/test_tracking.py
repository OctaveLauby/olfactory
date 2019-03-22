from time import sleep

from olfactory import tracking


def test_all():

    wait = 1 / 10

    class A():
        @staticmethod
        @tracking.trackedfunc
        def func1():
            sleep(wait)
            return 1

    @tracking.trackedfunc
    def func2(n):
        sleep(wait)
        return sum([A.func1() for _ in range(n)])

    tracker1 = tracking.CallTracker.trackers['test_all.<locals>.A.func1']
    tracker2 = tracking.CallTracker.trackers['test_all.<locals>.func2']

    assert func2(2) == 2
    assert tracker1.n_calls == 2
    assert tracker2.n_calls == 1

    assert func2(3) == 3
    assert tracker1.n_calls == 5
    assert tracker2.n_calls == 2

    assert int(tracker1.exc_time/wait) == tracker1.n_calls
    assert int(tracker2.exc_time/wait) == tracker1.n_calls + tracker2.n_calls
