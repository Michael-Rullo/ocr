from model import Layers
from model.Network import Network

IMAGE_SIZE = 28
CLASS_COUNT = 47

## Both Train.py and Predict.py build the network through this function so saved weights always line up.
##
##   (N, 1, 28, 28)  image
##   conv 3x3 x16    -> (N, 16, 28, 28)   where are the edges and stroke fragments?
##   pool            -> (N, 16, 14, 14)
##   conv 3x3 x32    -> (N, 32, 14, 14)   where do strokes meet, curve, or end?
##   pool            -> (N, 32, 7, 7)
##   flatten         -> (N, 1568)
##   dense           -> (N, 128)          which combination of parts is this?
##   dense           -> (N, 47)           one score per character class

def BuildCharacterCNN(classCount = CLASS_COUNT):
    pooledSize = IMAGE_SIZE // 4

    return Network([
        Layers.Conv2D(1, 16, 3, padding=1),
        Layers.ReLU(),
        Layers.MaxPool(2),

        Layers.Conv2D(16, 32, 3, padding=1),
        Layers.ReLU(),
        Layers.MaxPool(2),

        Layers.Flatten(),
        Layers.Dense(32 * pooledSize * pooledSize, 128),
        Layers.ReLU(),
        Layers.Dense(128, classCount)
    ])
