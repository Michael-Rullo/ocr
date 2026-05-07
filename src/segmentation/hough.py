import numpy as np
import math

def lines(image):
    max_votes = 0
    num_rows = len(image)
    num_cols = len(image[0])
    mmin = -1
    mmax = 2
    bmin = -(num_rows + mmax * num_cols)
    bmax = num_rows - mmin * num_cols
    # i = (m - mmin) / 0.1
    # j = (c - cmin) / 1
    accum = [[0 for _ in range(int((mmax - mmin) / 0.1))] for _ in range(int((bmax - bmin) / 1))]
    lines = [[255 for _ in range(num_cols)] for _ in range(num_rows)]

    for y in range(num_rows):
        for x in range(num_cols):
            if (image[y][x] == 0):
                k = -1
                while k < 2:
                    b = int(y - k * x)
                    i = int((k - mmin) / 0.1)
                    j = int((b - bmin) / 1)
                    accum[j][i] += 1
                    k += 0.1
    for i in image:
        for j in i:
            if j == 0:
                max_votes += 1
    peak_votes = max(max(row) for row in accum) if accum else 0
    threshold = max(1, int(0.2 * peak_votes))
    for i in range(len(accum)):
        for j in range(len(accum[0])):
            if accum[i][j] >= threshold:
                weight = accum[i][j]
                a = i - 1
                b = i + 1
                l = j - 1
                r = j + 1
                # in the future i should probably use a more robust method to find peaks in the accumulator
                # for now, we will just check the 8 neighboring cells in the accumulator to see if our current cell is a peak
                # future me, use a window of size n x n to eliminate blobs
                if (a >= 0):
                    if (weight > accum[a][j]):
                        accum[a][j] = 0
                    if (r < len(accum[0])):
                        if (weight > accum[i][r]):
                            accum[i][r] = 0
                        if (weight > accum[a][r]):
                            accum[a][r] = 0
                    if (l >= 0):
                        if (weight > accum[i][l]):
                            accum[i][l] = 0
                        if (weight > accum[a][l]):
                            accum[a][l] = 0
                if (b < len(accum)):
                    if (weight > accum[b][j]):
                        accum[b][j] = 0
                    if (r < len(accum[0])):
                        if (weight > accum[i][r]):
                            accum[i][r] = 0
                        if (weight > accum[b][r]):
                            accum[b][r] = 0
                    if (l >= 0):
                        if (weight > accum[i][l]):
                            accum[i][l] = 0
                        if (weight > accum[b][l]):
                            accum[b][l] = 0
    for i in range(len(accum)):
        for j in range(len(accum[0])):
            if accum[i][j] >= threshold:
                b = bmin + i * 1
                m = mmin + j * 0.1
                for x in range(num_cols):
                    y = int(m * x + b)
                    if (0 <= y < num_rows):
                        lines[y][x] = 0
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
            
            