import numpy as np
import math

from PIL import Image

SIGMA_MULTIPLIER = 0.25
YTHRESHOLD = 20

def RunProfiles(image):
    yAxisHistogram = GetHorizontalProfiles(image)
    yAxisSegments = SegmentProjectionProfile(yAxisHistogram, 0)
    yAxisSpacing = GetProfileSpacingAverage(yAxisSegments)
    yAxisHistogramBlurred = GaussianBlur(yAxisHistogram, yAxisSpacing * SIGMA_MULTIPLIER)
    yAxisSegmentsBlurred = SegmentProjectionProfile(yAxisHistogramBlurred, YTHRESHOLD)

    Image.fromarray(OLDDrawSegments(image, yAxisSegmentsBlurred)).show()

    boundingCoordinates = []

    ## vertical profiles are taken per line, since word gaps on different lines don't align into blank page-wide columns
    for lineSegment in yAxisSegmentsBlurred:
        lineImage = image[lineSegment[0]:lineSegment[1] + 1]

        ## no blur here: character gaps are only a few columns wide, so any blur at character scale erases them
        xAxisHistogram = GetVerticalProfiles(lineImage)
        xAxisSegments = SegmentProjectionProfile(xAxisHistogram, 0)

        boundingCoordinates += GetBoundingCoordinates([lineSegment], xAxisSegments)

    Image.fromarray(DrawBoundingBoxes(image, boundingCoordinates)).show()

    return boundingCoordinates

def GetHorizontalProfiles(image):
    yAxisHistogram = [0] * len(image)

    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] == 0:
                yAxisHistogram[i] += 1

    return yAxisHistogram

def GetVerticalProfiles(image):
    xAxisHistogram = [0] * len(image[0])

    for j in range(len(image[0])):
        for i in range(len(image)):
            if image[i][j] == 0:
                xAxisHistogram[j] += 1

    return xAxisHistogram

def SegmentProjectionProfile(projectionProfile, threshold):
    segmentsAboveThreshold = []
    openSegment = False
    segmentStart = 0

    for i, rowOrColumnValue in enumerate(projectionProfile):
        if rowOrColumnValue > threshold:
            if not openSegment:
                segmentStart = i
                openSegment = True
        else:
            if openSegment:
                segmentsAboveThreshold.append((segmentStart, i - 1))
                openSegment = False
    if openSegment:
        segmentsAboveThreshold.append((segmentStart, len(projectionProfile) - 1))

    return segmentsAboveThreshold

def CreateGaussianKernel(sigma, radius = None):
    if radius == None:
        radius = int(math.ceil(3 * sigma))

    size = 2 * radius - 1
    kernel = [math.exp(-x**2 / (2 * sigma**2)) for x in range(-radius, radius + 1)]
    total = sum(kernel)

    return [k / total for k in kernel]

def GaussianBlur(projectionProfile, sigma):
    if sigma <= 0:
        return projectionProfile

    radius = int(math.ceil(3 * sigma))
    kernel = CreateGaussianKernel(sigma, radius)

    n = len(projectionProfile)

    blurred = [0.0] * n

    for i in range(n):
        value = 0.0
        weightedSum = 0.0

        for j in range(-radius, radius + 1):
            index = i + j

            if index < 0:
                index = -index
            elif index >= n:
                index = 2 * n - 1 - index

            weight = kernel[j + radius]
            value += projectionProfile[index] * weight
            weightedSum += weight

        blurred[i] = value / weightedSum

    return blurred

def GetProfileSpacingAverage(projectionSegments):
    if len(projectionSegments) < 2:
        return 0

    gapSizeTotal = 0

    ## measures the blank gap between segments, not their pitch, so the blur sigma stays smaller than the gaps it must preserve
    for i in range(1, len(projectionSegments)):
        gapSizeTotal += projectionSegments[i][0] - projectionSegments[i - 1][1]

    return gapSizeTotal / (len(projectionSegments) - 1)

def GetBoundingCoordinates(yAxisProfileSegments, xAxisProfileSegments):
    upperLeftAndLowerRightCoordinates = []

    for yAxisSegment in yAxisProfileSegments:
        for xAxisSegment in xAxisProfileSegments:
            upperLeftAndLowerRightCoordinates.append(((xAxisSegment[0], yAxisSegment[0]), (xAxisSegment[1], yAxisSegment[1])))

    return upperLeftAndLowerRightCoordinates

## temporary drawing function for testing
def DrawSegments(image, segments, verticalProfile = False):
    y = len(image)
    x = len(image[0])

    for w in range(x if verticalProfile else y):
        if segments[w] > 0.25 * max(segments):
            for p in range(y if verticalProfile else x):
                if verticalProfile:
                    image[p][w] = 128   # paint column w
                else:
                    image[w][p] = 128   # paint row w
    
    return image

def DrawBoundingCoordinatePoints(image, listOfCoords):
    for entry in listOfCoords:
        topLeft = entry[0]
        bottomRight = entry[1]

        image[topLeft[1]][topLeft[0]] = 128
        image[bottomRight[1]][bottomRight[0]] = 128

    return image

def DrawBoundingBoxes(image, listOfCoords):
    for entry in listOfCoords:
        topLeft = entry[0]
        bottomRight = entry[1]
        for x in range(topLeft[0], bottomRight[0] + 1):
            image[topLeft[1]][x] = 128      ## top edge
            image[bottomRight[1]][x] = 128  ## bottom edge
        for y in range(topLeft[1], bottomRight[1] + 1):
            image[y][topLeft[0]] = 128      ## left edge
            image[y][bottomRight[0]] = 128  ## right edge
    return image

def OLDDrawSegments(img, segments):
    image = img.copy()
    for segment in segments:
        for i in range(segment[0], segment[1] + 1):
            for j in range(len(image[0])):
                image[i][j] = 128

    return image