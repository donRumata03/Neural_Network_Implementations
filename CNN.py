from perceptron import *
from scipy.ndimage import convolve

def edge_detect_convolve_test():
    mt1 = np.array([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1]
    ])

    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    res = convolve(mt1, kernel)

    print("Matrix:")
    print(mt1)

    print("Kernel:")
    print(kernel)

    print("After convolving:")
    print(res)

class CNN:
    input_size : Tuple[int, int]
    conv_geometry : tuple
    perceptron_geometry: tuple
    category_number : int
    def __init__(self, input_image_size : Tuple[int, int], conv_geometry : tuple, perceptron_geometry : tuple):
        self.input_size = input_image_size


if __name__ == '__main__':
    edge_detect_convolve_test()

