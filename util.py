"""Module for utility functions"""

from time import perf_counter

def fn_to_str(fn, *args, **kwargs):
    sep = ', ' if args and kwargs else ''

    return fn.__name__ + \
        '(' + ', '.join([str(arg) for arg in args]) + \
        sep + \
        ', '.join([f'{k} = {v}' for k, v in kwargs.items()]) + ')'


def time_func(fn, print_args=True):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        
        result = fn(*args, **kwargs)

        exec_time = perf_counter() - start

        fn_str = fn_to_str(fn, *args, **kwargs) if print_args else fn.__name__
        print(f'{fn_str} executed in {round(exec_time, 4)}s')

        return result
    return wrapper
