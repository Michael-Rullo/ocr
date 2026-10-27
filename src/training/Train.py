import os
import sys
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import Layers
from model.CharacterCNN import BuildCharacterCNN
from training import Data
from training.Optimizers import Adam

MODELS_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIRECTORY, "characterCnn.npz")

def SetParserArgs(parser):
    parser.add_argument("-epochs", "-e", type=int, default=10, help="Number of passes over the training set")
    parser.add_argument("-batch", "-b", type=int, default=64, help="Images per gradient step")
    parser.add_argument("-rate", "-r", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("-subset", "-s", type=int, default=None, help="Train on only the first N images (for smoke tests)")
    parser.add_argument("-output", "-o", type=str, default=DEFAULT_MODEL_PATH, help="Where to save the best weights")
    parser.add_argument("-resume", type=str, default=None, help="Path of saved weights to continue training from")
    parser.add_argument("-no-augment", action="store_true", help="Disable random shift augmentation")

def Accuracy(network, images, labels, batchSize = 256):
    correct = 0

    ## evaluated in batches: unrolling the whole validation set through im2col at once would need gigabytes
    for batchImages, batchLabels in Data.IterateBatches(images, labels, batchSize, shuffle=False):
        predictions = network.Forward(batchImages).argmax(axis=1)
        correct += (predictions == batchLabels).sum()

    return correct / len(images)

def TrainEpoch(network, loss, optimizer, images, labels, batchSize, augment):
    totalLoss = 0
    batchCount = 0
    startTime = time.time()

    for batchImages, batchLabels in Data.IterateBatches(images, labels, batchSize):
        if augment:
            batchImages = Data.AugmentBatch(batchImages)

        logits = network.Forward(batchImages)
        totalLoss += loss.Forward(logits, batchLabels)
        network.Backward(loss.Backward())
        optimizer.Step()

        batchCount += 1
        if batchCount % 100 == 0:
            elapsed = time.time() - startTime
            print(f"    batch {batchCount}  loss {totalLoss / batchCount:.4f}  ({elapsed:.0f}s)", end="\r")

    print()
    return totalLoss / batchCount

def main():
    argumentParser = argparse.ArgumentParser(description="Train the character CNN on EMNIST")
    SetParserArgs(argumentParser)
    args = argumentParser.parse_args()

    print("loading EMNIST...")
    images, labels = Data.LoadEmnist("train")
    if args.subset:
        images, labels = images[:args.subset], labels[:args.subset]

    (trainImages, trainLabels), (validationImages, validationLabels) = Data.SplitTrainValidation(images, labels)
    print(f"{len(trainImages)} training images, {len(validationImages)} validation images")

    network = BuildCharacterCNN()
    if args.resume:
        network.Load(args.resume)
        print(f"resumed from {args.resume}")

    loss = Layers.SoftmaxCrossEntropy()
    optimizer = Adam(network.Parameters(), learningRate=args.rate)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    bestAccuracy = 0

    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}")
        averageLoss = TrainEpoch(network, loss, optimizer, trainImages, trainLabels, args.batch, not args.no_augment)

        trainAccuracy = Accuracy(network, trainImages[:5000], trainLabels[:5000])
        validationAccuracy = Accuracy(network, validationImages, validationLabels)
        print(f"    loss {averageLoss:.4f}  train accuracy {trainAccuracy:.4f}  validation accuracy {validationAccuracy:.4f}")

        ## only keep the weights from the best validation epoch so a late overfitting epoch can't overwrite a good model
        if validationAccuracy > bestAccuracy:
            bestAccuracy = validationAccuracy
            network.Save(args.output)
            print(f"    saved to {args.output}")

if __name__ == "__main__":
    main()
