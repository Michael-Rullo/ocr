import numpy as np

from model.CharacterCNN import BuildCharacterCNN
from training import Data

class CharacterPredictor:
    def __init__(self, weightsPath):
        ## the network is rebuilt from the same definition used in training, so the saved arrays line up by position
        self.network = BuildCharacterCNN()
        self.network.Load(weightsPath)
        self.mapping = Data.LoadLabelMapping()

    def Predict(self, glyphs):
        ## glyphs: (N, 28, 28) arrays from Normalize.NormalizeGlyph. Returns one (character, confidence) per glyph.
        batch = np.asarray(glyphs)[:, None, :, :]
        logits = self.network.Forward(batch)

        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        classes = probabilities.argmax(axis=1)

        return [(self.mapping[classIndex], probabilities[i, classIndex]) for i, classIndex in enumerate(classes)]
