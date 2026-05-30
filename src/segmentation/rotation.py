import numpy as np
import math

def radians(degrees):
    return degrees * math.pi / 180
    
def rotate_image(image, angle):
    rad = radians(angle)
    row_length = len(image)
    col_length = len(image[0])
    x_center = (col_length - 1) / 2
    y_center = (row_length - 1) / 2
    bigger = max(row_length, col_length)

    inverse_map_matrix = [[math.cos(rad), math.sin(rad)], [-math.sin(rad), math.cos(rad)]]
    rotated_image = [[255 for _ in range(bigger)] for _ in range(bigger)]

    for i in range(row_length):
        for j in range(col_length):
            x_prime = j - x_center
            y_prime = i - y_center
            pixel = [[x_prime], [y_prime]]
            rotated_pixel = mult(inverse_map_matrix, pixel)
            x = int(rotated_pixel[0][0]) + int(x_center)
            y = int(rotated_pixel[1][0]) + int(y_center)
            if 0 <= x < col_length and 0 <= y < row_length:
                rotated_image[i][j] = image[y][x]

    return np.array(rotated_image, dtype=np.uint8)
            
def mult(matrix, vector):
    result = [[0], [0]]
    for i in range(len(matrix)):
        sum = 0
        for j in range(len(matrix[0])):
            sum += matrix[i][j] * vector[j][0]
        result[i][0] = sum
    return result

def get_angle(image):

    return 0