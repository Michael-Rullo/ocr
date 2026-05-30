import numpy as np
import math
import statistics

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
    estimated_max_lines = round(rows / 26) * 1
    num_lines = estimated_max_lines + 1

    edge_image = edge_detect(image, rows, cols)
    accumulator = accumulate(edge_image, mmin, mmax, bmin, bmax, m_step, b_step, rows, cols)
    while num_lines > estimated_max_lines:
        lines_result = lines(accumulator, mmin, mmax, m_step, bmin, bmax, b_step, rows, cols, window_size, lamb)
        lines_array = lines_result[0]
        num_lines = lines_result[1]
        slopes = lines_result[2]
        lamb += 0.2
    
    angle = get_angle(slopes)
    return (angle, lines_array)

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

def lines(accum, mmin, mmax,m_step, bmin, bmax, b_step, num_rows, num_cols, window_size, lamb = 4):
    total = 0
    num_lines = 0
    accum_rows = len(accum)
    accum_cols = len(accum[0])
    slopes = []

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
                num_lines += 1
                b = bmin + i * b_step
                m = mmin + j * m_step
                slopes.append(m)
                for x in range(num_cols):
                    y = round(m * x + b)
                    if (0 <= y < num_rows):
                        lines[y][x] = 0
    return (np.array(lines, dtype=np.uint8), num_lines, slopes)


def edge_detect(image, num_rows, num_col):
    new = [[255 for _ in range(num_col)] for _ in range(num_rows)]

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

## very basic, should change in the fiture to use the number of votes to determine rotation angle
## for now its just based on the lowest (closest to horizontal) slope of the detected lines
def get_angle(slopes):
    if not slopes:
        return 0.0

    line_angles_deg = np.array([math.degrees(math.atan(m)) for m in slopes], dtype=float)

    # Keep only plausible text-skew angles; this rejects many unrelated lines.
    plausible = line_angles_deg[np.abs(line_angles_deg) <= 30.0]
    if plausible.size == 0:
        return 0.0

    # Find dominant orientation cluster (1-degree bins) with a mild bias toward
    # smaller absolute tilt, then refine with a local median.
    bin_width = 1.0
    edges = np.arange(-30.0, 30.0 + bin_width, bin_width)
    hist, edges = np.histogram(plausible, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    scores = hist / (1.0 + np.abs(centers) / 10.0)
    best_idx = int(np.argmax(scores))

    if hist[best_idx] == 0:
        return 0.0

    local_window = 1.5
    cluster = plausible[np.abs(plausible - centers[best_idx]) <= local_window]
    if cluster.size == 0:
        return 0.0

    baseline_angle_deg = float(np.median(cluster))

    return -baseline_angle_deg