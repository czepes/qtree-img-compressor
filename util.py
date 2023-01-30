"""Module for utility functions"""

from time import perf_counter
from typing import Any, Callable


def fn_to_str(fn: Callable, *args: Any, **kwargs: Any):
    sep = ', ' if args and kwargs else ''

    return fn.__name__ + \
        '(' + ', '.join([str(arg) for arg in args]) + \
        sep + \
        ', '.join([f'{k} = {v}' for k, v in kwargs.items()]) + ')'


def time_func(fn: Callable, print_args: bool = True):
    def wrapper(*args: Any, **kwargs: Any):
        start = perf_counter()

        result = fn(*args, **kwargs)

        exec_time = perf_counter() - start

        fn_str = fn_to_str(fn, *args, **kwargs) if print_args else fn.__name__
        print(f'{fn_str} executed in {round(exec_time, 4)}s')

        return result
    return wrapper
