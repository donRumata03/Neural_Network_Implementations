from primitive_perceptron import *
from scipy.ndimage import convolve

def edge_detect_convolve_test():
    mt1 = np.array([
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 1],
    ])

    kernel1 = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ])

    kernel2 = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # kernel = np.array([kernel1, kernel2])


    print("Kernel shape:")
    print(kernel1.shape)

    res = convolve(mt1, kernel1)

    print("Matrix:")
    print(mt1)

    print("Kernel:")
    print(kernel1)
    print(kernel1)


    print("After convolving:")
    print(res)

class CNN:
    input_size : Tuple[int, int, int] # Last number : number of channels  e.g. 1 or 3
    conv_geometry : Tuple
    perceptron_geometry: tuple
    category_number : int

    filters : np.ndarray # For each layer there are several filters, suppose, they are all 3 x 3 X number_of_channels

    weights : np.ndarray # For perceptron
    conv_bias : np.ndarray
    perceptron_bias : np.ndarray

    def __init__(self, input_image_size : Tuple[int, int, int], conv_geometry : tuple, perceptron_geometry : tuple, category_number : int):
        """
        :param input_image_size:
        :param conv_geometry: All filters are 3x3, conv_geometry only defines how many filters are there in each layer
        Example : Value (16, 32, 64) would mean, that filters are: 
        :param perceptron_geometry:
        :param category_number:
        """
        self.input_size = input_image_size
        self.category_number = category_number

        self.perceptron_geometry = perceptron_geometry
        self.conv_geometry = conv_geometry

        filters = np.random.uniform(low=-1, high=1, size=conv_geometry)
        print(filters)

    def predict(self, input_image : np.array):
        # TODO:!
        pass

def make_basic_nn():
    c = CNN((28, 28, 1), (10, 3, 3), (1024, 512, 64), 10)


if __name__ == '__main__':
    make_basic_nn()
    # edge_detect_convolve_test()

