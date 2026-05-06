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
    accum = [[0 for _ in range((mmax - mmin) / 0.1)] for _ in range((bmax - bmin) / 1)]

    for y in range(num_rows):
        for x in range(num_cols):
            if (image[y][x] == 0):
                for k in range(mmin, mmax, 0.1):
                    int(b = y - k * x)
                    i = (k - mmin) / 0.1
                    j = (b - bmin) / 1
                    accum[i][j] += 1
    for i in image:
        for j in i:
            if j == 0:
                max_votes += 1
    threshold = int(0.2 * max_votes)
    for i in range(len(accum)):
        for j in range(len(accum[0])):
            if accum[i][j] >= threshold:
                weight = accum[i][j]
                a = i - 1
                b = i + 1
                l = j - 1
                r = j + 1
                if (a >= 0):
                    if (weight < accum[a][j]):
                        new[i][j] = 0
                        continue
                    if (r < len(image[0])):
                        if (weight < accum[i][r]):
                            new[i][j] = 0
                            continue
                        if (weight < accum[a][r]):
                            new[i][j] = 0

                    if (l >= 0):
                        if (weight < accum[i][l]):
                            new[i][j] = 0
                            continue
                        if (weight < accum[a][l]):
                            new[i][j] = 0
                            continue
                if (b < len(accum)):
                    if (weight < accum[a][j]):
                        new[i][j] = 0
                        continue
                    if (r < len(accum[0])):
                        if (weight < accum[i][r]):
                            new[i][j] = 0
                            continue
                        if (weight < accum[a][r]):
                            new[i][j] = 0

                    if (l >= 0):
                        if (weight < accum[i][l]):
                            new[i][j] = 0
                            continue
                        if (weight < accum[a][l]):
                            new[i][j] = 0
                            continue
                # our pixel is peak out of neighbor

                
            








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
            
            