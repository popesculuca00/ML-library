import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

class linear_regressor:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    def fit(self, X, y, show_results = False):
        assert X.shape[0] == y.shape[0]
        self.J = []
        self.weights = np.zeros((X.shape[1], 1))
        m = X.shape[1]
        for _ in trange(self.n_iterations):
            z = np.dot(X, self.weights)
            tmp = z - y
            dw = np.dot(X.T, tmp)
            self.weights -= self.learning_rate * dw
            self.J.append( np.sum(tmp**2) / (2*m))
        if show_results == True:
            plt.plot(self.J, [i+1 for i in range(self.n_iterations)])
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.show()
    def predict(self, X):
        return np.dot(X, self.weights)

class logistic_regressor:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    def fit(self, X, y, show_results = False):

        assert X.shape[0] == y.shape[0]

        self.weights = np.zeros((X.shape[1], 1))
        self.b = 0
        m = X.shape[1]

        for _ in trange(self.n_iterations):
            a = 1 / (1 + np.exp( np.dot(X, self.weights) + self.b ))
            tmp = np.reshape( a - y, (m,1) )
            dw = np.dot(X.T, tmp)
            db = np.sum(tmp) / m
            self.weights -= self.learning_rate * dw
            self.b -= self.learning_rate * db


    def predict(self, X):
        z = np.dot(X, self.weights) + self.b
        a = 1 / (1 + np.exp(-z) )
        return np.round(a)


a = linear_regressor()
x = np.random.rand(100, 1)
y = 2 + 3 ** x + np.random.rand(100, 1)
a.fit(x,y, show_results = True)