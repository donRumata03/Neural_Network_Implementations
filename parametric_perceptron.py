import random
from matplotlib import pyplot as plt
import numpy as np
from typing import *

from NN_basic_functions import *


class parametric_perceptron:
    """
    Optional number of layers, for each you should specify activation function,
    otherwise it`s automatically perceived as hyperbolic tangent
    """

    def __init__(self, input_size : int, output_size : int):


