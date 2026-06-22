import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from load_and_explore_data import load_dataset, DATA_PATH
from body_normalization import normalize_sequence
from interpolation import interpolate_sequence_gaps
from curl_smoothing import smooth_curl_sequence
import joblib

WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

POSE_LEN = 33 * 4
HAND_LEN = 21 * 3
LEFT_HAND_START = POSE_LEN
RIGHT_HAND_START = POSE_LEN + HAND_LEN


def get_hand_landmarks(frame, hand_start):
    hand_flat = frame[hand_start: hand_start + HAND_LEN]
    return hand_flat.reshape(21, 3)


def finger_curl(landmarks, tip_idx, mcp_idx, wrist_idx=WRIST):
    wrist = landmarks[wrist_idx]
    tip = landmarks[tip_idx]
    mcp = landmarks[mcp_idx]

    if np.all(landmarks == 0):
        return 0.0

    dist_tip_wrist = np.linalg.norm(tip - wrist)
    dist_mcp_wrist = np.linalg.norm(mcp - wrist) + 1e-6

    return dist_tip_wrist / dist_mcp_wrist


def compute_curl_features(frame):
    left_hand = get_hand_landmarks(frame, LEFT_HAND_START)
    right_hand = get_hand_landmarks(frame, RIGHT_HAND_START)

    curls = []
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
        curls.append(finger_curl(left_hand, tip, mcp))
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
        curls.append(finger_curl(right_hand, tip, mcp))

    return np.array(curls, dtype=np.float64)  # (10,)


def add_curl_features_to_sequence(sequence, smooth=True):
    curl_features = np.array([compute_curl_features(frame) for frame in sequence])
    if smooth:
        curl_features = smooth_curl_sequence(curl_features, window=3)
    return np.concatenate([sequence, curl_features], axis=1)


def process_sequence_v3(sequence):
    interpolated, _, _ = interpolate_sequence_gaps(sequence)
    normalized = normalize_sequence(interpolated)
    with_curl = add_curl_features_to_sequence(normalized)
    return with_curl


if __name__ == "__main__":
    print("Loading dataset...")
    X_raw, y_raw = load_dataset(DATA_PATH)
    print(f"Total recordings: {len(X_raw)}")

    print("\nProcessing (body normalization + curl features)...")

    X_processed = []
    for i, seq in enumerate(X_raw):
        seq_v3 = process_sequence_v3(seq)
        X_processed.append(seq_v3)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(X_raw)}...")

    X = np.array(X_processed, dtype=np.float32)
    print(f"\nShape of X: {X.shape}")
    print(f"  258 normalized coordinates + 10 curl features = {X.shape[2]} features per frame")

    n_samples, n_frames, n_features = X.shape
    X_2d = X.reshape(n_samples, n_frames * n_features)

    scaler = MinMaxScaler()
    X_norm_2d = scaler.fit_transform(X_2d)
    X_norm = X_norm_2d.reshape(n_samples, n_frames, n_features)

    print(f"Min-Max scaling completed. Min: {X_norm.min():.2f}, Max: {X_norm.max():.2f}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    for i, label in enumerate(le.classes_):
        print(f"{label:4s} -> {i}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_norm, y_encoded,
        test_size=0.15,
        random_state=42,
        stratify=y_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.176,
        random_state=42,
        stratify=y_temp
    )

    print(f"\nDataset split:")
    print(f"  Training   : {X_train.shape[0]} recordings")
    print(f"  Validation : {X_val.shape[0]} recordings")
    print(f"  Test       : {X_test.shape[0]} recordings")

    os.makedirs("models", exist_ok=True)
    np.save("models/X_train.npy", X_train)
    np.save("models/X_val.npy", X_val)
    np.save("models/X_test.npy", X_test)
    np.save("models/y_train.npy", y_train)
    np.save("models/y_val.npy", y_val)
    np.save("models/y_test.npy", y_test)
    np.save("models/classes.npy", le.classes_)
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nAll V3 files have been saved to the 'models/' directory!")
    print(f"Number of features per frame: {n_features}")