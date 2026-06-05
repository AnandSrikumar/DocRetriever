import functools
import time


def timeit(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        s = time.perf_counter()
        res = func(*args, **kwargs)
        e = time.perf_counter()
        print(f"{func.__name__} took {(e - s):.4f} seconds")
        return res

    return inner
