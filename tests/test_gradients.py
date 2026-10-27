import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import Layers
from model.Network import Network

EPSILON = 1e-5

## Numerical gradient check: nudge each value by +/- epsilon, measure how the loss moves, and compare that
## slope against what Backward computed. A wrong Backward does not crash, it just trains badly, so every
## layer earns its place here before it is allowed anywhere near the training loop.

def NumericalGradient(computeLoss, array):
    gradient = np.zeros_like(array)
    iterator = np.nditer(array, flags=["multi_index"])

    for _ in iterator:
        index = iterator.multi_index
        original = array[index]

        array[index] = original + EPSILON
        lossPlus = computeLoss()
        array[index] = original - EPSILON
        lossMinus = computeLoss()
        array[index] = original

        gradient[index] = (lossPlus - lossMinus) / (2 * EPSILON)

    return gradient

def AssertGradientsMatch(analytic, numerical):
    assert np.allclose(analytic, numerical, rtol=1e-5, atol=1e-7), f"max difference {np.abs(analytic - numerical).max()}"

def test_SoftmaxCrossEntropy():
    np.random.seed(0)
    logits = np.random.randn(4, 5)
    labels = np.array([0, 3, 1, 4])
    loss = Layers.SoftmaxCrossEntropy()

    loss.Forward(logits, labels)
    analytic = loss.Backward()
    numerical = NumericalGradient(lambda: loss.Forward(logits, labels), logits)

    AssertGradientsMatch(analytic, numerical)

def test_Dense():
    np.random.seed(0)
    input = np.random.randn(4, 6)
    labels = np.array([0, 2, 1, 2])
    dense = Layers.Dense(6, 3)
    loss = Layers.SoftmaxCrossEntropy()

    def ComputeLoss():
        return loss.Forward(dense.Forward(input), labels)

    ComputeLoss()
    inputGradient = dense.Backward(loss.Backward())

    AssertGradientsMatch(inputGradient, NumericalGradient(ComputeLoss, input))
    AssertGradientsMatch(dense.weightGradient, NumericalGradient(ComputeLoss, dense.weights))
    AssertGradientsMatch(dense.biasGradient, NumericalGradient(ComputeLoss, dense.biases))

def test_ReLU():
    np.random.seed(0)
    input = np.random.randn(4, 6)
    ## nudge values away from zero, the kink in ReLU makes the numerical slope meaningless right at it
    input[np.abs(input) < 0.1] = 0.5
    labels = np.array([0, 2, 1, 5])
    relu = Layers.ReLU()
    loss = Layers.SoftmaxCrossEntropy()

    def ComputeLoss():
        return loss.Forward(relu.Forward(input), labels)

    ComputeLoss()
    inputGradient = relu.Backward(loss.Backward())

    AssertGradientsMatch(inputGradient, NumericalGradient(ComputeLoss, input))

def test_Network():
    np.random.seed(0)
    input = np.random.randn(4, 8)
    labels = np.array([0, 2, 1, 3])
    network = Network([Layers.Dense(8, 5), Layers.ReLU(), Layers.Dense(5, 4)])
    loss = Layers.SoftmaxCrossEntropy()

    def ComputeLoss():
        return loss.Forward(network.Forward(input), labels)

    ComputeLoss()
    inputGradient = network.Backward(loss.Backward())

    AssertGradientsMatch(inputGradient, NumericalGradient(ComputeLoss, input))

    for parameter, gradient in network.Parameters():
        AssertGradientsMatch(gradient, NumericalGradient(ComputeLoss, parameter))

def NaiveConvolution(input, weights, biases, padding):
    ## the obviously-correct version: loop over every output pixel. Only used to validate the im2col forward pass.
    batchSize, _, height, width = input.shape
    outputChannels, _, kernelSize, _ = weights.shape
    padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    outputHeight = height + 2 * padding - kernelSize + 1
    outputWidth = width + 2 * padding - kernelSize + 1
    output = np.zeros((batchSize, outputChannels, outputHeight, outputWidth))

    for n in range(batchSize):
        for o in range(outputChannels):
            for y in range(outputHeight):
                for x in range(outputWidth):
                    patch = padded[n, :, y:y + kernelSize, x:x + kernelSize]
                    output[n, o, y, x] = (patch * weights[o]).sum() + biases[o]

    return output

def test_Conv2DForwardMatchesNaive():
    np.random.seed(0)
    input = np.random.randn(2, 3, 6, 6)
    conv = Layers.Conv2D(3, 4, 3, padding=1)

    assert np.allclose(conv.Forward(input), NaiveConvolution(input, conv.weights, conv.biases, 1))

def test_Conv2D():
    np.random.seed(0)
    input = np.random.randn(2, 2, 5, 5)
    labels = np.array([1, 2])
    conv = Layers.Conv2D(2, 3, 3, padding=1)
    flatten = Layers.Flatten()
    loss = Layers.SoftmaxCrossEntropy()

    def ComputeLoss():
        return loss.Forward(flatten.Forward(conv.Forward(input)), labels)

    ComputeLoss()
    inputGradient = conv.Backward(flatten.Backward(loss.Backward()))

    AssertGradientsMatch(inputGradient, NumericalGradient(ComputeLoss, input))
    AssertGradientsMatch(conv.weightGradient, NumericalGradient(ComputeLoss, conv.weights))
    AssertGradientsMatch(conv.biasGradient, NumericalGradient(ComputeLoss, conv.biases))

def test_MaxPool():
    np.random.seed(0)
    input = np.random.randn(2, 2, 4, 4)
    labels = np.array([1, 5])
    pool = Layers.MaxPool(2)
    flatten = Layers.Flatten()
    loss = Layers.SoftmaxCrossEntropy()

    def ComputeLoss():
        return loss.Forward(flatten.Forward(pool.Forward(input)), labels)

    ComputeLoss()
    inputGradient = pool.Backward(flatten.Backward(loss.Backward()))

    AssertGradientsMatch(inputGradient, NumericalGradient(ComputeLoss, input))

def test_ConvolutionalNetwork():
    np.random.seed(0)
    input = np.random.randn(2, 1, 8, 8)
    labels = np.array([0, 3])
    network = Network([
        Layers.Conv2D(1, 2, 3, padding=1), Layers.ReLU(), Layers.MaxPool(),
        Layers.Conv2D(2, 3, 3, padding=1), Layers.ReLU(), Layers.MaxPool(),
        Layers.Flatten(), Layers.Dense(3 * 2 * 2, 4)
    ])
    loss = Layers.SoftmaxCrossEntropy()

    def ComputeLoss():
        return loss.Forward(network.Forward(input), labels)

    ComputeLoss()
    inputGradient = network.Backward(loss.Backward())

    AssertGradientsMatch(inputGradient, NumericalGradient(ComputeLoss, input))

    for parameter, gradient in network.Parameters():
        AssertGradientsMatch(gradient, NumericalGradient(ComputeLoss, parameter))

def test_SaveAndLoad(tmp_path):
    np.random.seed(0)
    network = Network([Layers.Dense(8, 5), Layers.ReLU(), Layers.Dense(5, 4)])
    input = np.random.randn(3, 8)
    expected = network.Forward(input)

    path = tmp_path / "weights.npz"
    network.Save(path)

    reloaded = Network([Layers.Dense(8, 5), Layers.ReLU(), Layers.Dense(5, 4)])
    reloaded.Load(path)

    assert np.allclose(reloaded.Forward(input), expected)
