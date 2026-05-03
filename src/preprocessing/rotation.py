import numpy as np
import math

def radians(degrees):
    return degrees * math.pi / 180

def rotate_image(image, angle):
    # init the necessary variables for rotation
    row_length = len(image)
    col_length = len(image[0])
    rad = radians(angle)
    rotation_matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
    rotated_image = np.zeros((col_length, row_length), dtype=image.dtype)

    # rotate each pixel in the image using the rotation matrix
    for i in range(row_length):
        for j in range(col_length):
            pixel = [[j], [i]]
            rotated_pixel = mult(rotation_matrix, pixel)
            x = int(rotated_pixel[0][0])
            y = int(rotated_pixel[1][0])
            if 0 <= x < col_length and 0 <= y < row_length:
                rotated_image[x][y] = image[i][j]
    return rotated_image

def mult(matrix, vector):
    result = [[0], [0]]
    for i in matrix:
        sum = 0
        for j in range(len(i)):
            sum += i[j] * vector[j]
        result[matrix.index(i)][0] = sum
    return result
    
