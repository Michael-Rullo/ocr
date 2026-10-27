import os
import gzip
import struct

import numpy as np

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
IMAGE_SIZE = 28

def ReadIdxFile(path):
    with gzip.open(path, "rb") as file:
        ## header is 4 magic bytes (two zeros, data type, dimension count) then one big-endian int32 per dimension
        _, _, _, dimensionCount = struct.unpack(">BBBB", file.read(4))
        shape = struct.unpack(">" + "I" * dimensionCount, file.read(4 * dimensionCount))
        data = np.frombuffer(file.read(), dtype=np.uint8)

    return data.reshape(shape)

def LoadEmnist(subset, split = "balanced"):
    images = ReadIdxFile(os.path.join(DATA_DIRECTORY, f"emnist-{split}-{subset}-images-idx3-ubyte.gz"))
    labels = ReadIdxFile(os.path.join(DATA_DIRECTORY, f"emnist-{split}-{subset}-labels-idx1-ubyte.gz"))

    ## EMNIST stores every image transposed relative to how it reads, so swap the row and column axes
    images = images.transpose(0, 2, 1)

    ## (N, 1, 28, 28): the single channel axis is there so conv layers see the same layout as their own outputs
    images = images.reshape(len(images), 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32) / 255.0

    return images, labels.astype(np.int64)

def LoadLabelMapping(split = "balanced"):
    mapping = {}

    with open(os.path.join(DATA_DIRECTORY, f"emnist-{split}-mapping.txt")) as file:
        for line in file:
            classIndex, asciiCode = line.split()
            mapping[int(classIndex)] = chr(int(asciiCode))

    return mapping

def SplitTrainValidation(images, labels, validationFraction = 0.1, seed = 0):
    indices = np.random.default_rng(seed).permutation(len(images))
    validationCount = int(len(images) * validationFraction)

    validationIndices = indices[:validationCount]
    trainIndices = indices[validationCount:]

    return (images[trainIndices], labels[trainIndices]), (images[validationIndices], labels[validationIndices])

def AugmentBatch(images, maxShift = 2):
    ## shifts each image a random few pixels so the network can't rely on glyphs being perfectly centered,
    ## which they won't be once they come from our own segmentation instead of EMNIST's preprocessing
    augmented = np.empty_like(images)
    shifts = np.random.randint(-maxShift, maxShift + 1, size=(len(images), 2))

    for i, (shiftY, shiftX) in enumerate(shifts):
        augmented[i] = np.roll(images[i], (shiftY, shiftX), axis=(1, 2))

    return augmented

def IterateBatches(images, labels, batchSize, shuffle = True):
    indices = np.arange(len(images))

    ## reshuffling every epoch keeps the gradient steps from repeating the same batch sequence
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(images), batchSize):
        batchIndices = indices[start:start + batchSize]
        yield images[batchIndices], labels[batchIndices]
