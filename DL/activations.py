import numpy as np

class sigmoid:
    def __init__(self, name = None):
        if name:
            self.__name__ = "Sigmoid_" + str(name)

class ReLU:
    def __init__(self, name = None):
        self.name = name
        self.class_type = "ReLU"

    def init_class(self, output_units, name = None):
        self.output_units = output_units
        if self.name:
            self.__name__ = "ReLU_" + str(name)
        elif name :
            self.__name__ = "ReLU_" + str(name)


    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, grad_outputs):
        relu_grads = input > 0
        return grad_outputs * relu_grads

    def get_hypers(self):
        return {"name": self.__name__}


class tanh:
    def __init__(self, name = None):
        if name :
            self.__name__ = "Tanh_" + str(name)
