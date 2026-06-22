import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

POSE_LEN = 33 * 4   
HAND_LEN = 21 * 3   
LEFT_HAND_START = POSE_LEN
RIGHT_HAND_START = POSE_LEN + HAND_LEN

MIN_SCALE = 1e-4 


def _get_pose_xy(frame, landmark_idx):
    base = landmark_idx * 4
    return frame[base], frame[base + 1]


def _compute_center_and_scale(frame):
    lx, ly = _get_pose_xy(frame, LEFT_SHOULDER)
    rx, ry = _get_pose_xy(frame, RIGHT_SHOULDER)

    if (lx == 0 and ly == 0) or (rx == 0 and ry == 0):
        return None

    center_x = (lx + rx) / 2.0
    center_y = (ly + ry) / 2.0
    scale = np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

    if scale < MIN_SCALE:
        return None

    return center_x, center_y, scale


def normalize_frame(frame):
    result = frame.copy()

    cs = _compute_center_and_scale(frame)
    if cs is None:
        return result  

    center_x, center_y, scale = cs

    for i in range(33):
        base = i * 4
        x, y, z, vis = result[base], result[base + 1], result[base + 2], result[base + 3]
        if x == 0 and y == 0 and z == 0:
            continue 
        result[base]     = (x - center_x) / scale
        result[base + 1] = (y - center_y) / scale
        result[base + 2] = z / scale

    for i in range(21):
        base = LEFT_HAND_START + i * 3
        x, y, z = result[base], result[base + 1], result[base + 2]
        if x == 0 and y == 0 and z == 0:
            continue
        result[base]     = (x - center_x) / scale
        result[base + 1] = (y - center_y) / scale
        result[base + 2] = z / scale

    for i in range(21):
        base = RIGHT_HAND_START + i * 3
        x, y, z = result[base], result[base + 1], result[base + 2]
        if x == 0 and y == 0 and z == 0:
            continue
        result[base]     = (x - center_x) / scale
        result[base + 1] = (y - center_y) / scale
        result[base + 2] = z / scale

    return result


def normalize_sequence(sequence):
    return np.array([normalize_frame(frame) for frame in sequence])