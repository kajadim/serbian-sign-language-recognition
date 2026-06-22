import numpy as np


def smooth_curl_sequence(curl_sequence, window=5):
    if window % 2 == 0:
        window += 1 

    n_frames = curl_sequence.shape[0]
    half = window // 2
    smoothed = np.zeros_like(curl_sequence)

    for i in range(n_frames):
        start = max(0, i - half)
        end = min(n_frames, i + half + 1)
        smoothed[i] = curl_sequence[start:end].mean(axis=0)

    return smoothed