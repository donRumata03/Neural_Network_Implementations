from typing import Union
import random
import numpy as np

def count_gradient(function, variables_point : Union[np.ndarray, list, tuple], variables_delta : Union[float, tuple, list]):
    if type(variables_delta) == float:
        temp = variables_delta
        variables_delta = [temp for _ in range(len(variables_point))]
    res = []
    f_x0 = function(*variables_point)
    for index, var in enumerate(variables_point):
        this_new_val = variables_delta[index] + var
        this_new_vars = tuple(list(variables_point[:index]) + [this_new_val] + list(variables_point[index + 1:]))

        this_new_func = function(*this_new_vars)

        res.append((this_new_func - f_x0) / (this_new_val - var))

    return res


def gradient_optimize(function, point : tuple, diff_delta : float, learning_rate : float, iterations : int, debug = False):
    solution : list = list(point)
    for iteration in range(iterations):
        this_grad = count_gradient(function, tuple(solution), tuple([diff_delta for _ in range(len(point))]))
        this_alpha = learning_rate * (1 - iteration / iterations) ** (1 / 3) # (math.sqrt())
        this_increment = [_i * this_alpha for _i in this_grad]

        solution = [solution[_i] - this_increment[_i] for _i in range(len(solution))]

        if debug and random.random() < (5 / iterations):
            print(this_grad, this_alpha, this_increment, solution)

    return function(*solution), solution



def count_oneparam_gradient(function, variables_point : Union[np.ndarray, list, tuple], variables_delta : Union[float, tuple, list], args):
    if type(variables_delta) == float:
        temp = variables_delta
        variables_delta = [temp for _ in range(len(variables_point))]
    res = []
    f_x0 = function(variables_point, args)
    for index, var in enumerate(variables_point):
        this_new_val = variables_delta[index] + var
        this_new_vars = tuple(list(variables_point[:index]) + [this_new_val] + list(variables_point[index + 1:]))

        this_new_func = function(this_new_vars, args)

        res.append((this_new_func - f_x0) / (this_new_val - var))

    return res

