"""Module of utility functions"""

from time import perf_counter
from typing import Any, Callable


def func_to_str(func: Callable, *args: Any, **kwargs: Any) -> str:
    """
    Represent function as a string.

    Args:
        func (Callable): Function.

    Returns:
        str: Function string representation.
    """
    sep = ', ' if args and kwargs else ''

    return func.__name__ + \
        '(' + ', '.join([str(arg) for arg in args]) + \
        sep + \
        ', '.join([f'{k} = {v}' for k, v in kwargs.items()]) + ')'


def time_func(func: Callable, print_args: bool = False) -> Callable:
    """
    Decorate function to display its execution time.

    Args:
        func (Callable): Function.
        print_args (bool, optional): Flag to print function call arguments. Defaults to True.

    Returns:
        Callable: Time measuring wrapper function.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Time measuring wrapper function.

        Returns:
            Any: Function return value.
        """
        start = perf_counter()

        result = func(*args, **kwargs)

        exec_time = perf_counter() - start

        fn_str = func_to_str(func, *args, **kwargs) if print_args else func.__name__
        print(f'{fn_str} executed in {round(exec_time, 4)}s')

        return result
    return wrapper
