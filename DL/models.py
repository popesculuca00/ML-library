import numpy as np
from tqdm import trange
from IPython.display import clear_output

class Sequential:
    def __init__(self, layers, input_shape = None):

        self.network = [ layers[0] ]
        self.network[0].init_class(input_shape)

        for idx,layer in enumerate(layers[1:]):
            self.network.append(layer)

            if self.network[-1].class_type == "Dense" or self.network[-1].class_type == "ReLU":
                self.network[-1].init_class( self.network[-2].output_units , name = idx + 1 )
            self.network[-1].get_hypers()

        self.compiled = False

    def compile(self, loss, optimizer = None):
        self.compiled = True
        self.loss = loss
        self.optimizer = optimizer

    def forwardprop(self, X):

        activations = []
        input = X
        for l in self.network:
            activations.append(l.forward(input))
            input = activations[-1]
        return activations

    def train_batch(self, X, y):
        layer_activations = self.forwardprop(X)
        layer_inputs = [X] + layer_activations
        y_hat = layer_activations[-1]

        loss = self.loss.forward(y_hat, y)
        grad_loss = self.loss.backward(y_hat,y)


        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]

            loss_grad = layer.backward(layer_inputs[layer_index], grad_loss)

        return np.mean(loss)

    def predict(self, x):
        ans = self.forwardprop(x)
        return np.argmax(ans, axis = -1)


    def iterate_minibatch(self, inputs, targets, batchsize, shuffle = False):

        if shuffle :
            indices = np.random.permutation(len(inputs))
        for start_idx in trange(0, len(inputs)- batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx: start_idx + batchsize]
            else :
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def train(self, epochs, x, y, x_val = None, y_val = None, batchsize = 32, shuffle = False):
        train_log = []
        val_log = []

        for epoch in range(epochs):
            for x_batch, y_batch in self.iterate_minibatch(x, y, batchsize=batchsize, shuffle= shuffle):
                self.train_batch(x_batch, y_batch)
            train_log.append( np.mean(self.predict(x_batch) == y_batch  )  )
            if x_val:
                val_log.append( np.mean(self.predict(x_val) == y_val ))
            clear_output()


