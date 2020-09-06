import random
from matplotlib import pyplot as plt
import numpy as np
from typing import *
from circle_test_generator import *
from NN_basic_functions import *
from scipy.ndimage import convolve



def convolve_test():
    mt1 = np.array([
        [
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [0, 0, 0, 2, 1, 0, 1],
            [0, 0, 0, 2, 1, 0, 1],
            [0, 0, 0, 2, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
        ]
    ])

    kernel1 = np.array([
        [
            [3, 0, -3],
            [10, 0, -10],
            [3, 0, -3]
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]
    ])

    kernel2 = np.array([
        [
            [3, 0, -3],
            [10, 0, -10],
            [3, 0, -3]
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]
    ])

    kernel3 = np.array([
        [
            [3, 0, -3],
            [10, 0, -10],
            [3, 0, -3]
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]
    ])

    all_res = np.zeros((6, 7, 3))

    temp_res1 = convolve(mt1, kernel1)
    res1 = np.zeros((6, 7))
    for i in range(len(temp_res1)):
        res1 += temp_res1[i]

    temp_res2 = convolve(mt1, kernel2)
    res2 = np.zeros((6, 7))
    for i in range(len(temp_res2)):
        res2 += temp_res2[i]

    temp_res3 = convolve(mt1, kernel3)
    res3 = np.zeros((6, 7))
    for i in range(len(temp_res3)):
        res3 += temp_res3[i]

    all_res[:,:, 0] = res1
    all_res[:,:, 1] = res2
    all_res[:,:, 2] = res3

    print(all_res)

    #
    # print(res1)
    #
    # print(all_res[:,:,0].shape)
    # print(res1.shape)
    #
    # print(all_res)


def convolve_grad_test():
    """
    Counts gradient of filter in one layer

    (Only for testing)
    """
    pass


if __name__ == '__main__':
    convolve_test()
