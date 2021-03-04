import numpy as np


class Dense:
    def __init__(self, input_units, output_units, lr = 0.1, name = None):
        self.lr = lr
        self.weights = np.random.normal(loc = 0.0,
                                        scale = np.sqrt(2/(input_units+output_units)),
                                        size = (input_units, output_units))
        self.biases = np.zeros(output_units)
        self.name = name


    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases  = grad_output.mean(axis = 0) * input.shape[0]

        self.weights -= grad_weights* self.lr
        self.biases  -= grad_biases * self.lr
        return grad_input


a = Dense()