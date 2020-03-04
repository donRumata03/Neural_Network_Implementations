import math

from linear_regression import *
from gradient import *

class poly_regression:
    coefficients : list
    max_power : int
    def __init__(self, coefficients : tuple):
        self.coefficients = list(coefficients)
        self.max_power = len(coefficients)

    def predict(self, x : float, debug = False, coeffs = None):
        if debug:
            print([x ** i * self.coefficients[i] for i in range(len(self.coefficients))])
        _coeffs = coeffs if coeffs is not None else self.coefficients
        return sum([x ** i * _coeffs[i] for i in range(len(_coeffs))])

    """
    @staticmethod
    def get_one_one_derivative(err_result : float, x : float, n : int, sgn : float):
        return 2 * math.sqrt(err_result) * x ** n

    @staticmethod
    def get_one_pow_derivative(err_result : float, xs : list, n : int, sgn : float) -> float:
        return sum([poly_regression.get_one_one_derivative(err_result, x, n) for x in range(len(xs))])

    def get_gradient(self, err_result : float, xs : list) -> list:
        # return sum([self.get_derivative(err_result, xs[i], i) for i in range(len(xs))])
        return [self.get_one_pow_derivative(err_result, xs, __i, sgn) for __i in range(self.max_power)]
    """

    def error_function(self, xs, ys):
        res = 0
        for index, x in enumerate(xs):
            predict = self.predict(x)
            target = ys[index]
            res += (predict - target) ** 2
        return res

    def get_err_and_grad_and_grad2(self, xs : list, ys : list) -> tuple:
        grad = [0 for _ in range(self.max_power)]
        grad2 = grad[:]
        err = 0

        for index, (x, y) in enumerate(list(zip(xs, ys))):
            target = y
            predict = self.predict(x)
            err += (target - predict) ** 2

            for w_index in range(self.max_power):
                grad[w_index] += 2 * (predict - target) * x ** w_index
                grad2[w_index] += 2 * x ** (2 * w_index)

        return err, grad, grad2

    def learning_step(self, xs, ys, learning_rate):
        err, gradient, ______________ = self.get_err_and_grad_and_grad2(xs, ys)

        for k_index in range(self.max_power):
            self.coefficients[k_index] -= learning_rate * gradient[k_index]

    def newton_step(self, xs, ys, learning_rate = 1):
        err, grad, grad2 = self.get_err_and_grad_and_grad2(xs, ys)
        # print(err, grad, grad2)
        for w_index in range(self.max_power):
            self.coefficients[w_index] -= learning_rate * grad[w_index] / grad2[w_index]


    def train(self, xs, ys, learning_rate, iterations, method = None, debug = False):
        this_learning_data = []
        if method is None:
            method = self.learning_step

        for iteration in range(iterations):
            if debug:
                this_learning_data.append(self.error_function(xs, ys))
                if random.random() < 0.001 or iteration == 0:
                    print(f"Iteration: {iteration}/{iterations}", 'Coefficients:', self.coefficients, "Error function:", self.error_function(xs, ys))
            method(xs, ys, learning_rate)

        print(f"Beginning speed: {this_learning_data[0] - this_learning_data[1]}")
        print(f"End speed: {this_learning_data[-2] - this_learning_data[-1]}")
        if debug:
            return this_learning_data


def emulate_polynom(coeffs : list, start : float, stop : float, distortion : float, sample_number : int) -> tuple:
    xs = []
    for _ in range(sample_number):
        rand_x = start + random.random() * (stop - start)
        xs.append(rand_x)

    samples = sorted(xs)
    ys = []
    this_model = poly_regression(tuple(coeffs))
    for sample in samples:
        ys.append(this_model.predict(sample) + random.random() * distortion)
    return samples, ys

if __name__ == '__main__':
    p = poly_regression((0, 0, 0, 0, 0))


    generated_xs, generated_ys = emulate_polynom([-1, 1, 1, 100, -20], -3, 5, 400, 100)
    learning_data = p.train(generated_xs, generated_ys, 0.0000001, 1000, debug=True)
    learning_data.extend(p.train(generated_xs, generated_ys, 0.01, 1000, p.newton_step, debug = True))

    plt.plot(learning_data, list(range(len(learning_data))))
    plt.show()

    xs_to_plot = np.linspace(-3, 5, 1000)
    ys_to_plot = []
    for this_x in xs_to_plot:
        this_y = p.predict(this_x)
        ys_to_plot.append(this_y)

    plt.plot(xs_to_plot, ys_to_plot)
    plt.scatter(generated_xs, generated_ys)
    plt.show()

