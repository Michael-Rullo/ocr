import numpy as np
import math

def radians(degrees):
    return degrees * math.pi / 180

# def rotate_image(image, angle):
#     # init the necessary variables for rotation
#     row_length = len(image)
#     col_length = len(image[0])
#     rad = radians(angle)
#     rotation_matrix = [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]]
#     rotated_image = [[128 for _ in range(row_length)] for _ in range(col_length)]
#     x_center = (col_length - 1) / 2
#     y_center = (row_length - 1) / 2

#     # rotate each pixel in the image using the rotation matrix
#     for i in range(row_length):
#         for j in range(col_length):
#             pixel = [[j - x_center], [i - y_center]]
#             rotated_pixel = mult(rotation_matrix, pixel)
#             x = int(rotated_pixel[0][0]) + int(x_center)
#             y = int(rotated_pixel[1][0]) + int(y_center)
#             if 0 <= x < col_length and 0 <= y < row_length:
#                 rotated_image[x][y] = image[i][j]
#     return np.array(rotated_image, dtype=np.uint8)

# def mult(matrix, vector):
#     result = [[0], [0]]
#     for i in matrix:
#         sum = 0
#         for j in i:
#             sum += j * vector[matrix.index(i)][0]
#         result[matrix.index(i)][0] = sum
#     return result
    
def rotate_image(image, angle):
    row_length = len(image)
    col_length = len(image[0])
    bigger = max(row_length, col_length)
    rad = radians(angle)
    inverse_map_matrix = [[math.cos(rad), math.sin(rad)], [-math.sin(rad), math.cos(rad)]]
    rotated_image = [[255 for _ in range(bigger)] for _ in range(bigger)]
    x_center = (col_length - 1) / 2
    y_center = (row_length - 1) / 2
    
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

def deskew_image(image, candidate_angles):
    best_angle = 0
    best_score = float('inf')
    for angle in candidate_angles:
        rotated_image = rotate_image(image, angle)
        score = variance(rotated_image)
        if score < best_score:
            best_score = score
            best_angle = angle
    return np.array(rotate_image(image, best_angle), dtype=np.uint8)

def variance(image):
    rows = [0] * len(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            # 1 for white, 0 for black so no overflow issues
            rows[i] += int(image[i][j] if image[i][j] < 128 else 1)
    mean_row_val = 0
    for i in rows:
        mean_row_val += i / len(image)
    variance = 0
    for i in rows:
        variance += (i - mean_row_val) ** 2
    return variance / len(image)