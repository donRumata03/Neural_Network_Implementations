from gradient import *
from typing import Callable


def approximate_by_function(xs : list, ys : list, function : Callable, initial_values : tuple):
    """
    For line:
    xs : [(1,), (2,), ...];
    ys = [(-1,), (-3,)];
    function : (2,) -> (-3,);
    initial_values : (3,)
    :param xs:
    :param ys:
    :param function: 
    :param initial_values: 
    :return: 
    """
    input_space_dimension = len(xs[0])
    # Check, that all the points have the same dimension:
    if len(initial_values) != input_space_dimension:
        raise ValueError("")
    for x in xs:
        if len(x) != input_space_dimension:
            raise ValueError("Bad points dimensions!")


    def func_to_optimize(*args):
        res = 0
        for i in range(len(args)):
            res += (points[i] - function(args[i])) ** 2
        return res

    gradient_optimize(func_to_optimize, 0.0000001, 0.001, 1000)


if __name__ == '__main__':
    print("\xbef")