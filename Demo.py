from DL.layers import Dense
from DL.activations import sigmoid , ReLU
from DL.losses import softmax
from DL.models import Sequential
import keras



def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_val, y_val, X_test, y_test
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)





model = Sequential([
    Dense(100),
    ReLU(),
    Dense(200),
    ReLU(),
    Dense(10)
], input_shape = X_train.shape[1] )

model.compile(loss = softmax)


for i in model.network:
    print(i.get_hypers())

model.train(10, X_train, y_train, X_val, y_val)
