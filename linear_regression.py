import random
from typing import Any
from matplotlib import pyplot as plt
import numpy as np


class linear_regression:
    b : float
    k : float
    def __init__(self, init_k : float, init_b : float):
        self.k = init_k
        self.b = init_b

    def cost_function(self, xs : list, ys : list):
        if len(xs) != len(ys):
            raise ValueError
        res = 0
        for i in range(len(xs)):
            p = self.predict(xs[i])
            res += (p - ys[i]) ** 2
        return res

    def learning_step(self, xs, ys, learning_rate : float):
        cost = self.cost_function(xs, ys)
        little_delta = 0.00000001
        self.k += little_delta
        diff_k = (self.cost_function(xs, ys) - cost) / little_delta
        self.k -= little_delta
        self.b += little_delta
        diff_b = (self.cost_function(xs, ys) - cost) / little_delta
        self.b -= little_delta

        self.b -= diff_b * learning_rate
        self.k -= diff_k * learning_rate


    def predict(self, x : float):
        return self.k * x + self.b

    def learn(self, xs, ys, epochs, learning_rate, debug = False):
        for epoch in range(epochs):
            self.learning_step(xs, ys, learning_rate)
            if debug and random.random() < 0.01:
                print("Epoch:", epoch, "k =", self.k, "b =", self.b, "Error function =", self.cost_function(xs, ys))


def generate_random_line(k : float, b : float, start : float, stop : float, distortion : float, sample_number : int) -> tuple:
    xs = []
    for _ in range(sample_number):
        rand_x = start + random.random() * (stop - start)
        xs.append(rand_x)

    samples = sorted(xs)
    ys = []
    for sample in samples:
        ys.append(k * sample + b + random.random() * distortion)
    return samples, ys

def print_warning(text : Any):
    print("\033[95m\033[93m" + str(text) + "\033[0m\033[0m")


if __name__ == '__main__':
    """
    test_xs = [1, 2, 3, 4, 7.1]
    test_ys = [-1, 20, 30, 45, 50]
    """
    test_xs, test_ys = generate_random_line(10, -5, -1, 10, 30, 10)
    print("Xs:", test_xs)
    print("Ys:", test_ys)

    r = linear_regression(0, 0)
    print_warning("Train data:")
    print("___________________________________________________________________________")
    r.learn(test_xs, test_ys, 1000, 0.001, True)
    print("___________________________________________________________________________")
    print_warning("Result:")
    print("K:", r.k, "B:", r.b)

    to_plot = []
    to_plot_xs =  np.linspace(test_xs[0], test_xs[-1], 1000)
    for i in to_plot_xs:
        to_plot.append(r.predict(i))

    plt.scatter(test_xs, test_ys)
    plt.plot(to_plot_xs, to_plot)
    plt.show()