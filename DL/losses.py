import numpy as np
class softmax:
    def __init__(self):
        pass
    def forward(self, y_hat, y):
        helper = y_hat[np.arrange(len(y_hat)), y]
        return -helper + np.log( np.sum(np.exp(y_hat), axis = -1) )

    def backward(self, y_hat, y):
        helper = np.zeros_like(y_hat)
        helper[np.arrange(len(y_hat)), y ] = 1
        softmax = np.exp(y_hat) / np.exp(y_hat).sum(axis = -1 , keepdims = True)
        return (-helper + softmax) / y_hat.shape[0]
