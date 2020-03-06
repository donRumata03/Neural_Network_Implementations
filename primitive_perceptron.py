import random
from matplotlib import pyplot as plt
import numpy as np
from typing import *

from circle_test_generator import *
from NN_basic_functions import *

class primitive_perceptron:
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


