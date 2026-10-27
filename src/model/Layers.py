import numpy as np

## Every layer follows the same contract:
##   Forward(input) returns the output and stashes whatever Backward will need
##   Backward(outputGradient) returns the gradient with respect to the input
##   layers with weights also expose Parameters() as (parameter, gradient) pairs for the optimizer

class Dense:
    def __init__(self, inputSize, outputSize):
        ## He initialization: scaling by sqrt(2 / fanIn) keeps activation variance steady through ReLU layers,
        ## without it deeper networks either saturate or barely move at the start of training
        self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(2 / inputSize)
        self.biases = np.zeros(outputSize)

        self.weightGradient = np.zeros_like(self.weights)
        self.biasGradient = np.zeros_like(self.biases)
        self.input = None

    def Forward(self, input):
        self.input = input
        return input @ self.weights + self.biases

    def Backward(self, outputGradient):
        ## each weight's gradient sums its input times the output gradient over the whole batch
        ## written in place ([...]) so the arrays handed out by Parameters() stay the same objects between steps
        self.weightGradient[...] = self.input.T @ outputGradient
        self.biasGradient[...] = outputGradient.sum(axis=0)

        return outputGradient @ self.weights.T

    def Parameters(self):
        return [(self.weights, self.weightGradient), (self.biases, self.biasGradient)]

class Conv2D:
    def __init__(self, inputChannels, outputChannels, kernelSize, padding = 0):
        self.kernelSize = kernelSize
        self.padding = padding

        fanIn = inputChannels * kernelSize * kernelSize
        self.weights = np.random.randn(outputChannels, inputChannels, kernelSize, kernelSize) * np.sqrt(2 / fanIn)
        self.biases = np.zeros(outputChannels)

        self.weightGradient = np.zeros_like(self.weights)
        self.biasGradient = np.zeros_like(self.biases)

        self.inputShape = None
        self.columns = None

    def Forward(self, input):
        batchSize, _, height, width = input.shape
        outputHeight = height + 2 * self.padding - self.kernelSize + 1
        outputWidth = width + 2 * self.padding - self.kernelSize + 1

        ## im2col: unroll every kernel-sized patch into a row so the whole convolution becomes one matrix multiply
        padded = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.columns = ImageToColumns(padded, self.kernelSize, outputHeight, outputWidth)
        self.inputShape = input.shape

        ## each kernel is flattened to a column of the weight matrix, so (patches x patchSize) @ (patchSize x outputChannels)
        flatWeights = self.weights.reshape(len(self.weights), -1)
        output = self.columns @ flatWeights.T + self.biases

        return output.reshape(batchSize, outputHeight, outputWidth, -1).transpose(0, 3, 1, 2)

    def Backward(self, outputGradient):
        batchSize, _, height, width = self.inputShape
        outputHeight, outputWidth = outputGradient.shape[2:]

        ## line the gradient up with the rows of self.columns: one row per output pixel, one column per kernel
        flatGradient = outputGradient.transpose(0, 2, 3, 1).reshape(-1, len(self.weights))

        ## identical to the Dense backward pass since the forward pass was a matrix multiply on the unrolled patches
        self.weightGradient[...] = (flatGradient.T @ self.columns).reshape(self.weights.shape)
        self.biasGradient[...] = flatGradient.sum(axis=0)

        flatWeights = self.weights.reshape(len(self.weights), -1)
        columnsGradient = flatGradient @ flatWeights

        paddedShape = (batchSize, self.inputShape[1], height + 2 * self.padding, width + 2 * self.padding)
        paddedGradient = ColumnsToImage(columnsGradient, paddedShape, self.kernelSize, outputHeight, outputWidth)

        ## the padding pixels were constants we added, so their gradient is discarded
        return paddedGradient[:, :, self.padding:self.padding + height, self.padding:self.padding + width]

    def Parameters(self):
        return [(self.weights, self.weightGradient), (self.biases, self.biasGradient)]

def ImageToColumns(padded, kernelSize, outputHeight, outputWidth):
    batchSize, channels, _, _ = padded.shape
    patches = np.zeros((batchSize, outputHeight, outputWidth, channels, kernelSize, kernelSize), dtype=padded.dtype)

    ## loop over the kernel offsets rather than the image pixels: kernelSize^2 slices instead of height * width
    for i in range(kernelSize):
        for j in range(kernelSize):
            patches[:, :, :, :, i, j] = padded[:, :, i:i + outputHeight, j:j + outputWidth].transpose(0, 2, 3, 1)

    return patches.reshape(batchSize * outputHeight * outputWidth, channels * kernelSize * kernelSize)

def ColumnsToImage(columns, paddedShape, kernelSize, outputHeight, outputWidth):
    batchSize, channels, _, _ = paddedShape
    patches = columns.reshape(batchSize, outputHeight, outputWidth, channels, kernelSize, kernelSize)
    padded = np.zeros(paddedShape, dtype=columns.dtype)

    ## the reverse of ImageToColumns, except overlapping patches share pixels so their gradients add up
    for i in range(kernelSize):
        for j in range(kernelSize):
            padded[:, :, i:i + outputHeight, j:j + outputWidth] += patches[:, :, :, :, i, j].transpose(0, 3, 1, 2)

    return padded

class MaxPool:
    def __init__(self, size = 2):
        self.size = size
        self.mask = None

    def Forward(self, input):
        batchSize, channels, height, width = input.shape
        size = self.size

        ## splitting each axis into (blocks, size) turns every pooling window into its own pair of axes
        windows = input.reshape(batchSize, channels, height // size, size, width // size, size)
        output = windows.max(axis=(3, 5))

        ## remember which pixel won each window, Backward routes the gradient only there
        ## ties (common once ReLU has zeroed a region) split the gradient evenly instead of duplicating it
        winners = windows == output[:, :, :, None, :, None]
        self.mask = winners / winners.sum(axis=(3, 5), keepdims=True)

        return output

    def Backward(self, outputGradient):
        batchSize, channels, outputHeight, outputWidth = outputGradient.shape

        inputGradient = outputGradient[:, :, :, None, :, None] * self.mask

        return inputGradient.reshape(batchSize, channels, outputHeight * self.size, outputWidth * self.size)

class Flatten:
    def __init__(self):
        self.inputShape = None

    def Forward(self, input):
        self.inputShape = input.shape
        return input.reshape(len(input), -1)

    def Backward(self, outputGradient):
        return outputGradient.reshape(self.inputShape)

class ReLU:
    def __init__(self):
        self.mask = None

    def Forward(self, input):
        self.mask = input > 0
        return input * self.mask

    def Backward(self, outputGradient):
        ## gradient only passes where the input was positive, everywhere else the output was a flat zero
        return outputGradient * self.mask

class SoftmaxCrossEntropy:
    def __init__(self):
        self.probabilities = None
        self.labels = None

    def Forward(self, logits, labels):
        ## subtracting each row's max before exponentiating avoids overflow and leaves the softmax unchanged
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)

        self.probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        self.labels = labels

        correctClassProbabilities = self.probabilities[np.arange(len(labels)), labels]

        return -np.log(correctClassProbabilities + 1e-12).mean()

    def Backward(self):
        ## the gradient of softmax followed by cross entropy collapses to (probabilities - oneHot)
        gradient = self.probabilities.copy()
        gradient[np.arange(len(self.labels)), self.labels] -= 1

        return gradient / len(self.labels)
