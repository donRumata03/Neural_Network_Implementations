import random
from matplotlib import pyplot as plt
import numpy as np
from typing import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def known_sigmoid_derivative(s_val):
    return s_val * (1 - s_val)


Scary_val = Union[np.array, float, np.float64, np.float32]


class Derivativable_function:
    @staticmethod
    def compute(value: Scary_val) -> Scary_val:
        raise Exception

    @staticmethod
    def derivative(value: Scary_val, function_value: Scary_val) -> Scary_val:
        raise Exception


"""
Here are some common Activation functions with their derivatives:
1) Sigmoid
2) Hyperbolic tangent
3) ReLU (with neuron death protection)
4) ...
"""

class Activation_function(Derivativable_function):
    pass

class Sigmoid(Activation_function):
    @staticmethod
    def compute(value : Scary_val) -> Scary_val:
        return 1 / (1 + np.exp(-value))

    @staticmethod
    def derivative(value : Scary_val, function_value : Scary_val) -> Scary_val:
        return function_value * (1 - function_value)


class HyperTan(Activation_function):
    @staticmethod
    def compute(value : Scary_val) -> Scary_val:
        exp2x = np.exp(2 * value)
        return (exp2x - 1) / (exp2x + 1)

    @staticmethod
    def derivative(value : Scary_val, function_value : Scary_val) -> Union[np.array, float, np.float64, np.float32]:
        return 1 - function_value ** 2


class ReLU(Activation_function):
    @staticmethod
    def compute_impl(value : Scary_val, alpha : float = 0.075) -> Union[None, float]:
        """
        Not Really weird function :)
        Though, it isn`t used ^)), so it doesn`t need to work
        """
        if isinstance(value, np.ndarray):
            if len(value.shape) == 1:
                for i in range(len(value)):
                    value[i] = ReLU.compute_impl(value[i])
                else:
                    for i in range(len(value)):
                        ReLU.compute_impl(value[i])
        else:
            return max(-alpha * value, value)

    @staticmethod
    def derivative_impl(value: Scary_val, indexes : Union[tuple, int] = None, alpha: float = 0.075) -> Union[None, float]:
        """
        Really weird function :)
        Also not used, but it works
        """
        if isinstance(value, np.ndarray) and len(value.shape) > len(indexes):
            if len(value.shape) == 1:
                for i in range(value.shape[len(indexes)]):
                    ReLU.derivative_impl(value, indexes = tuple(list(indexes) + [i]))
            else:
                for i in range(value.shape[len(indexes)]):
                    ReLU.derivative_impl(value, indexes=tuple(list(indexes) + [i]))
        else:
            index_sequence = ""
            for j in range(len(indexes)):
                index_sequence += f"[{indexes[j]}]"
            exec(f"value{index_sequence} = (-alpha if value{index_sequence} < 0 else 1)")

    @staticmethod
    def compute(value : Scary_val, alpha : float = 0.075) -> Scary_val:
        return np.maximum(value * -alpha, value)

    @staticmethod
    def derivative(value : Scary_val,
                function_value : Scary_val, alpha : float = 0.075) -> Scary_val:
        value[value==0] = 1
        return function_value / value
        # new_val = value.copy()
        # ReLU.derivative_impl(new_val, (), alpha)
        # return new_val

"""
Here are some loss functions with their derivatives defined

1) MSE
2) SoftMax
3) Cross Entropy

"""

class Loss_function(Derivativable_function):
    @staticmethod
    def compute(output: Scary_val, answer : Scary_val) -> Scary_val:
        raise Exception

    @staticmethod
    def derivative(output: Scary_val, answer : Scary_val, function_value : Scary_val) -> Scary_val:
        raise Exception

class MSE(Loss_function):
    @staticmethod
    def compute(output: Scary_val, answer : Scary_val) -> Scary_val:
        return np.mean((output - answer)** 2)

    @staticmethod
    def derivative(output: Scary_val, answer : Scary_val, function_value : Scary_val) -> Scary_val:
        return 2 * (output - answer) / len(output)

class SoftMax(Loss_function):
    @staticmethod
    def compute(output: Scary_val, answer : Scary_val) -> Scary_val:
        exps = np.exp(output)
        return exps / exps.sum()

    @staticmethod
    def derivative(output: Scary_val, answer : Scary_val, function_value : Scary_val) -> Scary_val:
        return 2 * (output - answer) / len(output)




if __name__ == '__main__':
    example_output = np.array([1., 2, 3, 4, 0.1, 10])
    example_answer = np.array([1., 3, 5, 7, 100])

    print()

    function_out = SoftMax.compute(example_output, example_answer)
    print(list(map(lambda x: round(x, 3), function_out, )))
    print(MSE.derivative(example_output, example_answer, function_out))

    # ms = np.array([[-2, 3, 2., 0, 100],
    #                [-2200, 34, 2., -1, 101]])
    # val = ReLU.compute(ms.copy())
    # der = (ReLU.compute(ms + 0.00000001) - val) / 0.00000001
    # print(val)
    # print(ReLU.derivative(ms.copy(), val))
