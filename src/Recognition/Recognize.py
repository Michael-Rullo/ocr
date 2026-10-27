import numpy as np

from Recognition import Normalize

## a gap wider than this fraction of the typical character width on the line is read as a space
SPACE_GAP_RATIO = 0.6

def GroupIntoLines(boundingCoordinates):
    ## RunProfiles segments columns separately per text line, so every box on a line shares the same vertical span
    lines = {}

    for box in boundingCoordinates:
        top, bottom = box[0][1], box[1][1]
        lines.setdefault((top, bottom), []).append(box)

    ## lines top to bottom, characters left to right
    return [sorted(boxes, key=lambda box: box[0][0]) for _, boxes in sorted(lines.items())]

def RecognizeText(binarizedImage, boundingCoordinates, predictor):
    textLines = []

    for lineBoxes in GroupIntoLines(boundingCoordinates):
        glyphs = []
        for (left, top), (right, bottom) in lineBoxes:
            glyphs.append(Normalize.NormalizeGlyph(binarizedImage[top:bottom + 1, left:right + 1]))

        predictions = predictor.Predict(glyphs)

        widths = [right - left + 1 for (left, _), (right, _) in lineBoxes]
        spaceThreshold = SPACE_GAP_RATIO * np.median(widths)

        line = ""
        for i, (character, _) in enumerate(predictions):
            if i > 0:
                gap = lineBoxes[i][0][0] - lineBoxes[i - 1][1][0]
                if gap > spaceThreshold:
                    line += " "
            line += character

        textLines.append(line)

    return "\n".join(textLines)
