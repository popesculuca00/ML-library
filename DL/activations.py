import numpy as np

class sigmoid:
    def __init__(self, name = None):
        if name:
            self.__name__ = "Sigmoid_" + str(name)

class relu:
    def __init__(self, name = None):
        if name:
            self.__name__ = "ReLU_" + str(name)

    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, grad_outputs):
        relu_grads = input > 0
        return grad_outputs * relu_grads

class tanh:
    def __init__(self, name = None):
        if name :
            self.__name__ = "Tanh_" + str(name)
