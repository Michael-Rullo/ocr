import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.CharacterCNN import BuildCharacterCNN, CLASS_COUNT
from training import Data
from training.Train import DEFAULT_MODEL_PATH

def SetParserArgs(parser):
    parser.add_argument("-model", "-m", type=str, default=DEFAULT_MODEL_PATH, help="Path of the saved weights")
    parser.add_argument("-plot", type=str, default=None, help="Save a confusion matrix image to this path")

def PredictAll(network, images, batchSize = 256):
    predictions = []

    for batchImages, _ in Data.IterateBatches(images, np.zeros(len(images)), batchSize, shuffle=False):
        predictions.append(network.Forward(batchImages).argmax(axis=1))

    return np.concatenate(predictions)

def ConfusionMatrix(labels, predictions):
    ## rows are the true class, columns are what the network said, so the diagonal is the correct answers
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix

def main():
    argumentParser = argparse.ArgumentParser(description="Evaluate the character CNN on the EMNIST test set")
    SetParserArgs(argumentParser)
    args = argumentParser.parse_args()

    images, labels = Data.LoadEmnist("test")
    mapping = Data.LoadLabelMapping()

    network = BuildCharacterCNN()
    network.Load(args.model)

    predictions = PredictAll(network, images)
    matrix = ConfusionMatrix(labels, predictions)

    print(f"test accuracy: {(predictions == labels).mean():.4f}  ({len(images)} images)")

    perClassAccuracy = matrix.diagonal() / matrix.sum(axis=1)
    print("\nweakest characters:")
    for classIndex in np.argsort(perClassAccuracy)[:8]:
        print(f"    {mapping[classIndex]}  {perClassAccuracy[classIndex]:.3f}")

    ## zero the diagonal so the largest remaining entries are the most frequent mistakes
    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)
    print("\nmost common confusions (true -> predicted):")
    for flatIndex in np.argsort(mistakes, axis=None)[::-1][:10]:
        trueClass, predictedClass = np.unravel_index(flatIndex, mistakes.shape)
        print(f"    {mapping[trueClass]} -> {mapping[predictedClass]}  {mistakes[trueClass, predictedClass]} times")

    if args.plot:
        import matplotlib.pyplot as plt

        characters = [mapping[i] for i in range(CLASS_COUNT)]
        figure, axis = plt.subplots(figsize=(12, 12))
        axis.imshow(np.log1p(matrix), cmap="Blues")
        axis.set_xticks(range(CLASS_COUNT), characters)
        axis.set_yticks(range(CLASS_COUNT), characters)
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        axis.set_title("confusion matrix (log scale)")
        figure.savefig(args.plot, bbox_inches="tight")
        print(f"\nsaved confusion matrix to {args.plot}")

if __name__ == "__main__":
    main()
