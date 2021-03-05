import numpy as np


class Dense:
    def __init__(self, output_units, lr = 0.1, input_units = None,  name = None):
        self.output_units = output_units
        self.lr = lr
        self.input_units = input_units
        self.name = name
        self.class_type = "Dense"

    def init_class(self, input_units = None, name = None):

        if input_units:
            self.input_units = input_units

        if self.name:
            self.__name__ = "Dense_" + str(self.name)
        elif name:
            self.__name__ = "Dense_" + str(name)
        else:
            self.__name__ = "Dense_0"

        self.weights = np.random.normal(loc = 0.0,
                                        scale = np.sqrt(2/(self.input_units+self.output_units)),
                                        size = (self.input_units, self.output_units))
        self.biases = np.zeros(self.output_units)



    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases  = grad_output.mean(axis = 0) * input.shape[0]

        self.weights -= grad_weights* self.lr
        self.biases  -= grad_biases * self.lr
        return grad_input

    def get_hypers(self):
        return {"name": self.__name__, "Input units": self.input_units, "Output units": self.output_units}

