import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from Recognition import Normalize
from Recognition import Recognize

def MakeCrop(height, width, inkTop, inkBottom, inkLeft, inkRight):
    ## white page with a black rectangle of ink, in the 0 = ink / 255 = background convention of our binarized images
    crop = np.full((height, width), 255, dtype=np.uint8)
    crop[inkTop:inkBottom, inkLeft:inkRight] = 0
    return crop

def test_NormalizeGlyphShapeAndRange():
    normalized = Normalize.NormalizeGlyph(MakeCrop(60, 40, 5, 55, 10, 30))

    assert normalized.shape == (28, 28)
    assert normalized.min() >= 0 and np.isclose(normalized.max(), 1)

def test_NormalizeGlyphIsCentered():
    ## ink placed far into one corner of the crop should still land in the middle of the field
    normalized = Normalize.NormalizeGlyph(MakeCrop(60, 60, 0, 30, 0, 15))
    centerY, centerX = Normalize.CenterOfMass(normalized)

    assert abs(centerY - 13.5) < 1
    assert abs(centerX - 13.5) < 1

def test_NormalizeGlyphKeepsAspectRatio():
    ## a tall thin stroke should stay tall and thin, not get stretched into a square
    normalized = Normalize.NormalizeGlyph(MakeCrop(60, 60, 5, 55, 28, 32))
    inkRows = np.where(normalized.max(axis=1) > 0.5)[0]
    inkColumns = np.where(normalized.max(axis=0) > 0.5)[0]

    assert len(inkRows) >= 18
    assert len(inkColumns) <= 6

def test_NormalizeGlyphEmptyCrop():
    assert np.all(Normalize.NormalizeGlyph(np.full((10, 10), 255, dtype=np.uint8)) == 0)

def test_GroupIntoLinesOrdersTopToBottomLeftToRight():
    boxes = [
        ((40, 50), (50, 70)),   ## second line, second character
        ((10, 50), (20, 70)),   ## second line, first character
        ((30, 0), (40, 20)),    ## first line, second character
        ((0, 0), (10, 20)),     ## first line, first character
    ]

    lines = Recognize.GroupIntoLines(boxes)

    assert lines == [
        [((0, 0), (10, 20)), ((30, 0), (40, 20))],
        [((10, 50), (20, 70)), ((40, 50), (50, 70))],
    ]
