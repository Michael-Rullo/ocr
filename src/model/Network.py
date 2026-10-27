import numpy as np

class Network:
    def __init__(self, layers):
        self.layers = layers

    def Forward(self, input):
        for layer in self.layers:
            input = layer.Forward(input)

        return input

    def Backward(self, outputGradient):
        ## backpropagation is the chain rule applied layer by layer, so it walks the network in reverse
        for layer in reversed(self.layers):
            outputGradient = layer.Backward(outputGradient)

        return outputGradient

    def Parameters(self):
        parameters = []

        for layer in self.layers:
            if hasattr(layer, "Parameters"):
                parameters += layer.Parameters()

        return parameters

    def Save(self, path):
        np.savez(path, *[parameter for parameter, _ in self.Parameters()])

    def Load(self, path):
        saved = np.load(path)

        ## np.savez names positional arrays arr_0, arr_1, ... in the order given, matching Parameters() order
        for i, (parameter, _) in enumerate(self.Parameters()):
            parameter[...] = saved[f"arr_{i}"]
