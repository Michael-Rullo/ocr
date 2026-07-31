import numpy as np

def OtsuThreshold(image, valueRange = 256):
    histogram = [0] * valueRange
    varianceList = [0] * valueRange
    pixelCount = 0
    for pixel in image.flatten():
        histogram[pixel] += 1
        pixelCount += 1
    
    for i in range(valueRange):
        mew_f = 0
        mew_b = 0
        wf = sum(histogram[:i]) / pixelCount
        wb = sum(histogram[i:]) / pixelCount
        for j in range(i):
            mew_f += j * histogram[j]
        for j in range(i, valueRange):
            mew_b += j * histogram[j]
        mew_f /= sum(histogram[:i]) if sum(histogram[:i]) > 0 else 1
        mew_b /= sum(histogram[i:]) if sum(histogram[i:]) > 0 else 1
        varianceList[i] = wb * wf * ((mew_b - mew_f) ** 2)
    
    optimalThreshold = varianceList.index(max(varianceList))
    return optimalThreshold

def OtsuBinarize(image, threshold):
    binarized = image.copy()
    binarized[image >= threshold] = 255
    binarized[image < threshold] = 0
    return np.array(binarized, dtype=np.uint8)