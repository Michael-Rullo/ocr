import numpy as np
import math

def run_hough(image):
    rows = len(image)
    cols = len(image[0])
    mmin = -1
    mmax = 1
    m_step = 0.1
    bmin = -(rows + mmax * cols)
    bmax = rows - mmin * cols
    b_step = 1
    window_size = math.ceil(26 / 2)
    lamb = 4

    edge_image = edge_detect(image, rows, cols)
    accumulator = accumulate(edge_image, mmin, mmax, bmin, bmax, m_step, b_step, rows, cols)
    lines_array = lines(accumulator, mmin, mmax, bmin, bmax, rows, cols, window_size, lamb)
    return lines_array


def accumulate(image, mmin, mmax, bmin, bmax, m_step, b_step, rows, cols):
    votes_array = [[0 for _ in range(round((mmax - mmin) / m_step))] for _ in range(round((bmax - bmin) / b_step))]

    for y in range(rows):
        for x in range(cols):
            if image[y][x] == 0:
                for m in range(round(mmin / m_step), round(mmax / m_step), round(m_step / m_step)):
                    m = m * m_step
                    b = round(y - m * x)
                    i = round((m - mmin) / m_step)
                    j = round((b - bmin) / b_step)
                    votes_array[j][i] += 1 
    return votes_array

def lines(accum, mmin, mmax, bmin, bmax, num_rows, num_cols, window_size, lamb = 4):
    total = 0
    accum_rows = len(accum)
    accum_cols = len(accum[0])

    for row in accum:
        for v in row:
            total += v
    
    average = total / (len(accum) * len(accum[0]))
    threshold = average + lamb * np.std(np.array(accum).flatten())

    output = [[0] * accum_cols for _ in range(accum_rows)]
    lines = [[255 for _ in range(num_cols)] for _ in range(num_rows)]

    # Greedy NMS: strongest peaks first, suppress overlapping bins (fixes duplicate / plateau peaks).
    suppressed = [[False] * accum_cols for _ in range(accum_rows)]
    candidates = [
        (accum[i][j], i, j)
        for i in range(accum_rows)
        for j in range(accum_cols)
        if accum[i][j] >= threshold
    ]
    candidates.sort(reverse=True)
    for votes, i, j in candidates:
        if suppressed[i][j]:
            continue
        output[i][j] = votes
        i_lo = max(0, i - window_size)
        i_hi = min(accum_rows - 1, i + window_size)
        j_lo = max(0, j - window_size)
        j_hi = min(accum_cols - 1, j + window_size)
        for ii in range(i_lo, i_hi + 1):
            for jj in range(j_lo, j_hi + 1):
                suppressed[ii][jj] = True
    ## End Greedy NMS

    for i in range(accum_rows):
        for j in range(accum_cols):
            if output[i][j] > threshold:
                b = bmin + i * 1
                m = mmin + j * 0.1
                for x in range(num_cols):
                    y = round(m * x + b)
                    if (0 <= y < num_rows):
                        lines[y][x] = 0
    return np.array(lines, dtype=np.uint8)


def edge_detect(image, num_rows, num_col):
    new = [[255 for _ in range(num_col)] for _ in range(num_rows)]

    ## suboptimal when i have a better solution to sliding window in this exact file but whatevs
    ## it is bothering me whenever i look at it i will fix it (probably soon)
    for i in range(num_rows):
        for j in range(num_col):
            weight = image[i][j]
            t = i
            b = i
            l = j
            r = j
            window_size = 3
            for p in range(int(window_size / 2)):
                t = t - 1 if t - 1 >= 0 else t
                b = b + 1 if b + 1 < num_rows else b
                l = l - 1 if l - 1 >= 0 else l
                r = r + 1 if r + 1 < num_col else r
            for num in range(b - t + 1):
                for num1 in range(r - l + 1):
                    if (weight != image[t + num][l + num1]):
                        new[i][j] = 0
                        break

    return np.array(new, dtype=np.uint8)