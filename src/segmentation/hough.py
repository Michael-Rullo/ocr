import numpy as np
import math

# hardcode the window size to make changes easier
window = 51

def lines(image):
    num_rows = len(image)
    num_cols = len(image[0])
    mmin = -1
    mmax = 2
    bmin = -(num_rows + mmax * num_cols)
    bmax = num_rows - mmin * num_cols
    accum = [[0 for _ in range(int((mmax - mmin) / 0.1))] for _ in range(int((bmax - bmin) / 1))]
    lines = [[255 for _ in range(num_cols)] for _ in range(num_rows)]
    for y in range(num_rows):
        for x in range(num_cols):
            if (image[y][x] == 0):
                k = -1
                for ki in range(-10, 20, 1):
                    k = ki / 10
                    b = int(y - k * x)
                    i = int((k - mmin) / 0.1)
                    j = int((b - bmin) / 1)
                    accum[j][i] += 1
    max_votes = max(max(row) for row in accum)
    threshold = max(1, int(0.4 * max_votes))
    accum_rows = len(accum)
    accum_cols = len(accum[0])
    output = [[0] * accum_cols for _ in range(accum_rows)]
    window_size = math.ceil(window / 2)
    for i in range(accum_rows):
        for j in range(accum_cols):
            if accum[i][j] >= threshold:
                largest_x = j
                largest_y = i
                l = j
                r = j
                t = i
                b = i
                for p in range(window_size):
                    l = l - 1 if l > 0 else 0
                    r = r + 1 if r < accum_cols - 1 else accum_cols - 1
                    t = t - 1 if t > 0 else 0
                    b = b + 1 if b < accum_rows - 1 else accum_rows - 1
                for num in range(b - t + 1):
                    for num1 in range(r - l + 1):
                        if accum[t + num][l + num1] > accum[largest_y][largest_x]:
                            largest_y = t + num
                            largest_x = l + num1
                output[largest_y][largest_x] = accum[largest_y][largest_x]
    for i in range(accum_rows):
        for j in range(accum_cols):
            if output[i][j] > 0:
                print(f"m={mmin + j * 0.1:.2f}, b={bmin + i}, votes={output[i][j]}")
                b = bmin + i * 1
                m = mmin + j * 0.1
                for x in range(num_cols):
                    y = int(m * x + b)
                    if (0 <= y < num_rows):
                        lines[y][x] = 0
    all_vals = sorted([v for row in accum for v in row], reverse=True)
    print(all_vals[:10])
    return np.array(lines, dtype=np.uint8)
                
            








def edge_detect(image):
    num_rows = len(image)
    num_col = len(image[0])
    new = [[255 for _ in range(num_col)] for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(num_col):
            weight = image[i][j]
            a = i - 1
            b = i + 1
            l = j - 1
            r = j + 1
            if (a >= 0):
                if (weight != image[a][j]):
                    new[i][j] = 0
                    continue
                if (r < len(image[0])):
                    if (weight != image[i][r]):
                        new[i][j] = 0
                        continue
                    if (weight != image[a][r]):
                        new[i][j] = 0

                if (l >= 0):
                    if (weight != image[i][l]):
                        new[i][j] = 0
                        continue
                    if (weight != image[a][l]):
                        new[i][j] = 0
                        continue
            if (b < len(image)):
                if (weight != image[a][j]):
                    new[i][j] = 0
                    continue
                if (r < len(image[0])):
                    if (weight != image[i][r]):
                        new[i][j] = 0
                        continue
                    if (weight != image[a][r]):
                        new[i][j] = 0

                if (l >= 0):
                    if (weight != image[i][l]):
                        new[i][j] = 0
                        continue
                    if (weight != image[a][l]):
                        new[i][j] = 0
                        continue
    return np.array(new, dtype=np.uint8)
            
            