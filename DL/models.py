
class Sequential:
    def __init__(self, layers, input_shape):
        self.network = []
        for idx,layer in enumerate(layers):
