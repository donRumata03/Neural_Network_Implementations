import random
from matplotlib import pyplot as plt
import numpy as np
from typing import *


# For generation:
def rand(x0, distribution):
    return x0 - distribution / 2 + random.random() * (distribution - x0)

def generate_points_circle(center_x : float, center_y : float, radius : float, sample_number : int, inside : bool = True, pole_w = None, pole_h = None):
    res = []
    for i in range(sample_number):
        this_x = center_x
        this_y = center_y
        while (this_x == center_x and this_y == center_y) or (((this_x - center_x) ** 2 + (this_y - center_y) ** 2 > radius ** 2) == inside):
            this_x = center_x - radius + 2 * radius * random.random() if pole_w is None else rand(center_x, pole_w)
            this_y = center_y - radius + 2 * radius * random.random() if pole_h is None else rand(center_y, pole_h)
            if not inside and random.random() < 0.0001:
                pass
                # print(len(res))
                pass # print(this_x, this_y)
        res.append((this_x, this_y))

    return res


def generate_test_points(center_x : float, center_y : float, radius1 : float, radius2 : float, sample_number : int, frame_w = None, frame_h = None) -> tuple:
    blues = generate_points_circle(center_x, center_y, radius1, sample_number, True)
    reds = generate_points_circle(center_x, center_y, radius2, sample_number, False, frame_w, frame_h)
    return blues, reds

def make_training_data(number : int = 1000):
    blues, reds = generate_test_points(0, 0, 0.6, 0.7, number, 2, 2)
    arr = [(1, i[0], i[1]) for i in blues] + [(0, i[0], i[1]) for i in reds]
    random.shuffle(arr)
    return arr

# For NN:

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def known_sigmoid_derivative(s_val):
    return s_val * (1 - s_val)


def print_activation(activation):
    print("\n______________________________")
    print(f"Activation (number of steps : {len(activation)}):")
    for i in range(len(activation)):
        print(str(i) + "th step:")
        print(activation[i][0])
    print("_______________________________")


def print_weights_gradient(g : list):
    print("\n______________________________\nWeights gradient:")
    for index, this_g in enumerate(g):
        print(f"Layer {index} gradient:")
        print(this_g)


class perceptron:
    w : Union[list, np.array] = []
    input_size : int
    geometry: tuple
    def __init__(self, input_size : int, geometry : tuple, w : list = None, debug = False):
        self.input_size = input_size
        self.geometry = geometry # Example: layers: (5, 2, 4) -> 3 layers;
        if w is not None:
            self.w = w
        # Else: Random Init:
        else:
            self.add_w((input_size, self.geometry[0]))
            for layer_index in range(len(self.geometry) - 1):
                self.add_w((self.geometry[layer_index], self.geometry[layer_index + 1]))
            self.add_w((self.geometry[-1], 1))

        self.w = np.array(self.w)

        if debug:
            print("___________________________")
            print("Random weights:")
            print(f"Input weights ({self.input_size}, {self.geometry[0]}):")
            print(self.w[0])
            print(f"Hidden weights ({len(self.geometry) - 1} pcs):")
            for this_layer in self.w[1:-1]:
                print(this_layer)
            print("Output weights:")
            print(self.w[-1])
            print("____________________________")


    def add_w(self, size : tuple):
        self.w.append(np.random.uniform(low=-1, high=1, size=size))

    def predict(self, x : np.array, debug = False, return_process_data = False, return_raw_answer : bool = False) -> Union[tuple, bool]:
        assert x.shape[0] == self.input_size
        # Perform forward propagation:
        zs = []
        ais = []
        current_val : np.array = x.reshape((1, x.shape[0]))
        if debug:
            print("_____________________________________________\nPerforming forward propagation:")
            print(f"Input:{current_val}")
        for layer_index in range(len(self.w)):
            this_w = self.w[layer_index]
            this_z = np.dot(current_val, this_w)
            current_val = sigmoid( this_z )
            if return_process_data:
                ais.append(current_val)
                zs.append(this_z)
            if debug:
                print(f"After iteration {layer_index}: result:{current_val}")
        result = current_val[0][0]

        if debug:
            print("Result:", result)

        return (result > 0.5 if not return_raw_answer else result) if not return_process_data else ((result > 0.5 if not return_raw_answer else result), zs, ais)

    def count_experimental_gradient(self, inp : np.array, required : float) -> Tuple[float, list]:
        predict = self.predict(inp, return_raw_answer=True)
        J0 = (predict - required) ** 2
        # print("Model predict:", predict, "Correct predict:", required, "Error:", J0)
        w_delta = 0.0000000001

        res = []

        for layer_index in range(len(self.w)):
            res.append(np.zeros(self.w[layer_index].shape))
            for left_neuron_index in range(len(self.w[layer_index])):
                for right_neuron_index in range(len(self.w[layer_index][left_neuron_index])):
                    self.w[layer_index][left_neuron_index][right_neuron_index] += w_delta
                    new_J = (self.predict(inp, return_raw_answer=True) - required) ** 2
                    self.w[layer_index][left_neuron_index][right_neuron_index] -= w_delta
                    this_gradient = (new_J - J0) / w_delta
                    res[layer_index][left_neuron_index][right_neuron_index] = this_gradient

        return J0, res

    def count_back_propagational_gradient(self, x : np.array, y : float) -> Tuple[float, list]:
        # Getting model prediction
        predict, zs, ais = self.predict(x, return_process_data=True, return_raw_answer=True)
        res = []
        # Performing Back propagation:

        # Last layer:
        err = (y - predict) ** 2

        dJ_da = 2 * (predict - y)
        dJ_dz = dJ_da * known_sigmoid_derivative(predict)

        last_gradient = np.array([ais[-2][0][i] * dJ_dz for i in range(len(ais[-2][0]))])
        res.append(np.array([last_gradient]).T)

        dJ_das = np.array([self.w[-1][i][0] * dJ_dz for i in range(self.geometry[-1])])

        # Hidden layers:
        dJ_d_prev_as = dJ_das
        for temp_layer_index, layer_size in enumerate(self.geometry[::-1]):
            if temp_layer_index == 0:
                continue
            layer_index = len(self.geometry) - temp_layer_index - 1
            this_weights = self.w[layer_index + 1]
            this_layer_activation = ais[layer_index][0]
            this_w_gradient = np.zeros(self.w[layer_index + 1].shape, dtype=np.float64)
            new_layer_dJ_das = np.zeros(len(this_layer_activation))

            for right_n_index, right_n_grad in enumerate(dJ_d_prev_as):
                right_neuron_activation = ais[layer_index + 1][0][right_n_index]
                acting_val = known_sigmoid_derivative(right_neuron_activation) * dJ_d_prev_as[right_n_index]
                for left_n_index, left_n_activation in enumerate(this_layer_activation):
                    # Computing Weights delta for one layer:
                    this_weight = this_weights[left_n_index][right_n_index]
                    this_neuron_activation = ais[layer_index][0][left_n_index]  # Left neuron activation
                    this_w_gradient[left_n_index][right_n_index] = this_neuron_activation * acting_val # known_sigmoid_derivative(right_neuron_activation) * dJ_d_prev_as[right_n_index]
                    new_layer_dJ_das[left_n_index] += this_weight * acting_val

            res.append(this_w_gradient)
            # print("Analytical Layer", layer_index, "Gradient\n", this_w_gradient)
            dJ_d_prev_as = new_layer_dJ_das


        input_gradient = np.zeros((self.input_size, self.geometry[0]), dtype=np.float64) # TODO  : Count it!
        for input_index in range(self.input_size):
            input_acting_val = known_sigmoid_derivative(ais[0][0][input_index]) * x[input_index]
            for layer1_index in range(self.geometry[0]):
                 input_gradient[input_index][layer1_index] = dJ_d_prev_as[layer1_index] * input_acting_val

        res.append(input_gradient)

        return err, res[::-1]

    def count_gradient(self, x : np.array, y : float) -> Tuple[float, list]:
        return self.count_back_propagational_gradient(x, y)
        # return self.count_experimental_gradient(x, y)

    def SGD_fit(self, xs, ys, learning_rate : float, batch_size : int, epochs : int):
        losses = []
        for epoch in range(epochs):
            print(f"\n__________________________\nEpoch: {epoch} ; Loss: {losses[-1] if losses else 'Unknown'}...")
            this_loss_sum = 0
            batch_count = 0
            batch_gradient : np.array = np.array([np.zeros(i.shape, dtype = np.float64) for i in self.w])
            for sample_index, sample in enumerate(xs):
                batch_count += 1
                this_err, this_gradient = self.count_gradient(sample, ys[sample_index])
                batch_gradient += np.array(this_gradient)
                if batch_count == batch_size:
                    self.w -= batch_gradient * learning_rate
                    batch_count = 0
                    batch_gradient.fill(0)
                this_loss_sum += this_err
            losses.append(this_loss_sum / len(xs))
        return losses

    def GD_fit(self, xs, ys, epochs : int, learning_rate : float):
        losses = []
        for epoch in range(epochs):
            print(f"\n__________________________\nEpoch: {epoch} ; Loss: {losses[-1] if losses else 'Unknown'}...")
            this_loss_sum = 0
            for sample_index, sample in enumerate(xs):
                this_err, this_gradient = self.count_gradient(sample, ys[sample_index])

                self.w -= np.array(this_gradient) * learning_rate

                # for layer_index in range(len(self.w)):
                #    self.w[layer_index] -= this_gradient[layer_index] * learning_rate

                this_loss_sum += this_err
            losses.append(this_loss_sum / len(xs))
        return losses

    def train(self, xs, ys, epochs : int, learning_rate : float, store_loss = False, store_grad = False):
        losses = []
        for epoch in range(epochs):
            print(f"\n__________________________\nEpoch: {epoch}\nPerforming Back propagation...")
            this_loss_sum = 0
            for sample_index, sample in enumerate(xs):
                res, zs, ais = self.predict(sample, return_process_data=True)
                # print_activation(ais)
                this_y = ys[sample_index]
                # Performing Back propagation:

                # 1st layer:
                last_activation = ais[-1][0][0]
                err = (this_y - last_activation) ** 2
                this_loss_sum += err
                dJ_da = 2 * (last_activation - this_y)
                dJ_dz = dJ_da * known_sigmoid_derivative(last_activation)

                last_gradient = np.array([ais[-2][0][i] * dJ_dz for i in range(len(ais[-2][0]))])
                dJ_das = np.array([self.w[-1][i][0] for i in range(self.geometry[-1])])

                for w_index in range(len(last_gradient)):
                    self.w[-1][w_index] -= learning_rate * last_gradient[w_index]

                # Hidden layers:
                dJ_d_prev_as = dJ_das
                for temp_layer_index, layer_size in enumerate(self.geometry[::-1]):
                    if temp_layer_index == 0:
                        continue
                    layer_index = len(self.geometry) - temp_layer_index - 1
                    this_weights = self.w[layer_index + 1]
                    # print("Layer_index:", layer_index)
                    this_layer_activation = ais[layer_index][0]

                    this_w_gradient = np.zeros(self.w[layer_index + 1].shape, dtype=np.float64)

                    # print("dJ_d_prev_as", dJ_d_prev_as)

                    new_layer_dJ_das = np.zeros(len(this_layer_activation))

                    for right_n_index, right_n_grad in enumerate(dJ_d_prev_as):
                        for left_n_index, left_n_activation in enumerate(this_layer_activation):
                            # Computing Weights delta for one layer:
                            this_weight = this_weights[left_n_index][right_n_index]
                            this_neuron_activation = ais[layer_index][0][left_n_index] # Left neuron activation
                            right_z = zs[layer_index + 1][0][right_n_index]
                            right_neuron_activation = ais[layer_index + 1][0][right_n_index]
                            # print("This activation:", this_neuron_activation)
                            # print("This weight:", this_weight)

                            this_w_gradient[left_n_index][right_n_index] = this_neuron_activation * known_sigmoid_derivative(right_neuron_activation) * dJ_d_prev_as[right_n_index]
                            # TODO : Store: known_sigmoid_derivative(right_neuron_activation) * dJ_d_prev_as[right_n_index] !!!!!
                            new_layer_dJ_das[left_n_index] += this_weight * known_sigmoid_derivative(right_neuron_activation) * dJ_d_prev_as[right_n_index]


                    self.w[layer_index + 1] -= this_w_gradient * learning_rate
                    dJ_d_prev_as = new_layer_dJ_das


                # print("___________")

            if store_loss:
                losses.append(this_loss_sum / len(xs))
                print("This Loss:", losses[-1])


        if store_loss:
            return losses



# Different tests:

def test_matrix():
    test_arr = np.array([np.array([1, 3]), np.array([2, 3]), np.array([1]), np.array([4, 5, 6])])
    print(test_arr + test_arr * 0.01)


def test_gen():
    _blues, _reds = generate_test_points(0, 0, 0.6, 0.7, 10000, 2, 2)
    plt.scatter(*zip(*_blues))
    plt.scatter(*zip(*_reds))
    plt.show()

    r = [rand(0, 1) for _ in range(100)]
    print(len([i for i in r if i > 0]), len([i for i in r if i < 0]))


def test_grad_counting():
    data = make_training_data()
    this_perceptron = perceptron(2, (3, 2, 4), debug=False)

    random_sample = data[0]
    print("Chosen sample:", random_sample)
    err, experimental_grad = this_perceptron.count_experimental_gradient(np.array(random_sample[1:]), random_sample[0])
    print_weights_gradient(experimental_grad)
    err, anal_grad = this_perceptron.count_back_propagational_gradient(np.array(random_sample[1:]), random_sample[0])
    print_weights_gradient(anal_grad)


def show_model_prediction_map(model : perceptron, number : int):
    blues, reds = generate_test_points(0, 0, 0.645, 0.655, number, 2, 2)
    points = list(blues) + list(reds)
    model_blues = []
    model_reds = []
    for p in points:
        res = model.predict(np.array(p))
        if res:
            model_blues.append(p)
        else:
            model_reds.append(p)


    blue_xs = np.array(model_blues).T[0] if model_blues else []
    blue_ys = np.array(model_blues).T[1] if model_blues else []

    red_xs = np.array(model_reds).T[0] if model_reds else []
    red_ys = np.array(model_reds).T[1] if model_reds else []

    plt.scatter(blue_xs, blue_ys, color="blue")
    plt.scatter(red_xs, red_ys, color="red")

    plt.show()


def test_nn():
    p = perceptron(2, (80, 40, 20), debug=True)
    losses = []
    for i in range(40):
        print("\n\n____________________________\nReal epoch:", i)
        training_data = make_training_data(1000)
        losses.append(p.SGD_fit([np.array((i[1], i[2])) for i in training_data], [i[0] for i in training_data], 0.01, 10, 1)[0])
        print("Real loss:", losses[-1])

    show_model_prediction_map(model=p, number=10000)
    print("Losses:", np.array(losses))

    plt.plot(list(range(len(losses))), losses)
    plt.show()


    """
    print(np.array(p.predict(np.array([0.1, -0.5]), True, True)))

    print(
        np.array(p.train([np.array((i[1], i[2])) for i in training_data], [i[0] for i in training_data], 10, 1, True)))
    """


if __name__ == '__main__':
    test_nn()

