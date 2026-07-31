import numpy as np
import math
from PIL import Image

def RunProfiles(image):
    yAxisHistogram = GetHorizontalProfiles(image)
    xAxisHistogram = GetVerticalProfiles(image)
    img = DrawSegments(image, yAxisHistogram, False)
    img = DrawSegments(img, xAxisHistogram, True)
    return img

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

def Segment(projection_profile, threshold):
    segmentsAboveThreshold = []
    openSegment = False
    segmentStart = 0

    for i, row in enumerate(projection_profile):
        if row > threshold:
            if not open:
                segmentStart = i
                openSegment = True
        else:
            if open:
                segmentsAboveThreshold.append((segmentStart, i - 1))
                open = False
    if open:
        segmentsAboveThreshold.append((segmentStart, len(projection_profile) - 1))

    return segmentsAboveThreshold

## temporary drawing function for testing
def DrawSegments(image, segments, verticalProfile = False):
    # for segment in segments:
    #     for i in range(segment[0], segment[1] + 1):
    #         for j in range(len(image[0])):
    #             image[i][j] = 128

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