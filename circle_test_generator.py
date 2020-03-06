import random
import numpy as np


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



