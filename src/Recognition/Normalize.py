import math

import numpy as np

IMAGE_SIZE = 28
GLYPH_SIZE = 20

## Turns a character crop from our binarized page into the exact format the network was trained on.
## EMNIST glyphs are white ink on black, scaled so the longer side fits a 20x20 box, then placed in a
## 28x28 field with their center of mass at the center. Anything that differs here is something the
## network has never seen, so this function matters more than the architecture does.

def NormalizeGlyph(crop):
    ## our binarized images are black ink (0) on white (255), the network expects ink as 1.0 on 0.0
    ink = (np.asarray(crop) == 0).astype(np.float64)

    inkRows = np.where(ink.any(axis=1))[0]
    inkColumns = np.where(ink.any(axis=0))[0]
    if len(inkRows) == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    ink = ink[inkRows[0]:inkRows[-1] + 1, inkColumns[0]:inkColumns[-1] + 1]

    ## scale by the longer side so the aspect ratio survives (an 'l' stays thin, an 'm' stays wide)
    height, width = ink.shape
    scale = GLYPH_SIZE / max(height, width)
    newHeight = max(1, round(height * scale))
    newWidth = max(1, round(width * scale))
    resized = Blur(Resize(ink, newHeight, newWidth))

    ## position the glyph so its center of mass, not its bounding box, sits at the center of the field
    centerY, centerX = CenterOfMass(resized)
    top = round((IMAGE_SIZE - 1) / 2 - centerY)
    left = round((IMAGE_SIZE - 1) / 2 - centerX)
    top = min(max(top, 0), IMAGE_SIZE - newHeight)
    left = min(max(left, 0), IMAGE_SIZE - newWidth)

    normalized = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    normalized[top:top + newHeight, left:left + newWidth] = resized

    return normalized

def Resize(image, newHeight, newWidth):
    ## area averaging: each output pixel is the mean of the input region it covers. Point sampling a binary
    ## image would drop thin strokes entirely whenever the sample grid lands between them.
    height, width = image.shape
    resized = np.zeros((newHeight, newWidth))

    for y in range(newHeight):
        startY = int(y * height / newHeight)
        endY = max(startY + 1, math.ceil((y + 1) * height / newHeight))

        for x in range(newWidth):
            startX = int(x * width / newWidth)
            endX = max(startX + 1, math.ceil((x + 1) * width / newWidth))

            resized[y, x] = image[startY:endY, startX:endX].mean()

    return resized

def Dilate(image):
    ## every pixel takes the max of its 3x3 neighborhood, which thickens strokes by about one pixel per side.
    ## Not in the default path: EMNIST strokes are thicker than printed text, but on 56px Arial a full dilation
    ## closed the counters of O/Q/6 and hurt more than it helped. Try wrapping Resize() with it for thin fonts.
    padded = np.pad(image, 1)
    dilated = np.zeros_like(image)

    for i in range(3):
        for j in range(3):
            dilated = np.maximum(dilated, padded[i:i + image.shape[0], j:j + image.shape[1]])

    return dilated

def Blur(image):
    ## EMNIST strokes have soft anti-aliased edges, a light 3x3 blur gives our hard binary edges the same look
    kernel = np.outer([1, 2, 1], [1, 2, 1]) / 16
    padded = np.pad(image, 1)
    blurred = np.zeros_like(image)

    for i in range(3):
        for j in range(3):
            blurred += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]

    ## rescale so stroke centers reach full intensity again, as they do in the training data
    return blurred / blurred.max() if blurred.max() > 0 else blurred

def CenterOfMass(image):
    total = image.sum()
    rows = np.arange(image.shape[0])
    columns = np.arange(image.shape[1])

    centerY = (image.sum(axis=1) * rows).sum() / total
    centerX = (image.sum(axis=0) * columns).sum() / total

    return centerY, centerX
