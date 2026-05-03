import numpy as np
import math

def radians(degrees):
    return degrees * math.pi / 180

def rotate_image(image, angle):
    # init the necessary variables for rotation
    row_length = len(image)
    col_length = len(image[0])
    rad = radians(angle)
    rotation_matrix = [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]]
    rotated_image = [[128 for _ in range(row_length)] for _ in range(col_length)]
    x_center = (col_length - 1) / 2
    y_center = (row_length - 1) / 2

    # rotate each pixel in the image using the rotation matrix
    for i in range(row_length):
        for j in range(col_length):
            pixel = [[j - x_center], [i - y_center]]
            rotated_pixel = mult(rotation_matrix, pixel)
            x = int(rotated_pixel[0][0]) + int(x_center)
            y = int(rotated_pixel[1][0]) + int(y_center)
            if 0 <= x < col_length and 0 <= y < row_length:
                rotated_image[x][y] = image[i][j]
    return np.array(rotated_image, dtype=np.uint8)

def mult(matrix, vector):
    result = [[0], [0]]
    for i in matrix:
        sum = 0
        for j in i:
            sum += j * vector[matrix.index(i)][0]
        result[matrix.index(i)][0] = sum
    return result
    
