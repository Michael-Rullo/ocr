import numpy as np

## Optimizers take the (parameter, gradient) pairs from Network.Parameters() and update the parameters
## in place each time Step() is called after a backward pass.

class SGD:
    def __init__(self, parameters, learningRate = 0.05):
        self.parameters = parameters
        self.learningRate = learningRate

    def Step(self):
        for parameter, gradient in self.parameters:
            parameter -= self.learningRate * gradient

class Adam:
    def __init__(self, parameters, learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.parameters = parameters
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        ## running averages of the gradient (momentum) and of its square (per-parameter step scaling)
        self.firstMoments = [np.zeros_like(parameter) for parameter, _ in parameters]
        self.secondMoments = [np.zeros_like(parameter) for parameter, _ in parameters]
        self.stepCount = 0

    def Step(self):
        self.stepCount += 1

        for (parameter, gradient), firstMoment, secondMoment in zip(self.parameters, self.firstMoments, self.secondMoments):
            firstMoment[...] = self.beta1 * firstMoment + (1 - self.beta1) * gradient
            secondMoment[...] = self.beta2 * secondMoment + (1 - self.beta2) * gradient ** 2

            ## the averages start at zero and are biased toward it for the first few steps, this rescales them
            correctedFirst = firstMoment / (1 - self.beta1 ** self.stepCount)
            correctedSecond = secondMoment / (1 - self.beta2 ** self.stepCount)

            parameter -= self.learningRate * correctedFirst / (np.sqrt(correctedSecond) + self.epsilon)
