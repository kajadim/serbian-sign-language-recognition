"""
Priprema dataseta V3:
  1. Normalizacija svih koordinata relativno na ramena (resava problem
     udaljenosti od kamere / pozicije u kadru)
  2. Curl feature-i za svaki prst (resava M/N tip problema)

Input modela ce biti (40, 268) - isto kao V2, ali sada su sirove
koordinate (258 od toga) normalizovane na telo pre racunanja curl-a.

Pokrenuti UMESTO prepare_dataset_v2.py:
    python prepare_dataset_v3.py

Zatim trenirati:
    python train_model_v3.py
"""
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

# ============================================================
# CURL FEATURE-I (isto kao u prepare_dataset_v2.py)
# ============================================================
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
    """sequence: (n_frames, 258) -> (n_frames, 268)"""
    curl_features = np.array([compute_curl_features(frame) for frame in sequence])
    if smooth:
        curl_features = smooth_curl_sequence(curl_features, window=3)
    return np.concatenate([sequence, curl_features], axis=1)


def process_sequence_v3(sequence):
    """
    Puni V3 pipeline za jednu sekvencu:
      1. Interpoliraj rupe (kratka gubljenja detekcije usred pokreta)
      2. Normalizuj na telo (ramena)
      3. Dodaj curl feature-e (sa temporalnim zagladjivanjem - smanjuje sum
         koji se uvecava na velikim udaljenostima od kamere)
    sequence: (n_frames, 258) -> (n_frames, 268)
    """
    interpolated, _, _ = interpolate_sequence_gaps(sequence)
    normalized = normalize_sequence(interpolated)
    with_curl = add_curl_features_to_sequence(normalized)
    return with_curl


if __name__ == "__main__":
    print("Ucitavanje dataseta..")
    X_raw, y_raw = load_dataset(DATA_PATH)
    print(f"Ukupno snimaka: {len(X_raw)}")

    print("\nObrada V3 (normalizacija na telo + curl feature-i)...")
    print("(Ovo moze potrajati nekoliko minuta)")

    X_processed = []
    for i, seq in enumerate(X_raw):
        seq_v3 = process_sequence_v3(seq)
        X_processed.append(seq_v3)
        if (i + 1) % 1000 == 0:
            print(f"  Obradjeno {i + 1}/{len(X_raw)}...")

    X = np.array(X_processed, dtype=np.float32)
    print(f"\nOblik X (V3): {X.shape}")
    print(f"  258 normalizovanih koordinata + 10 curl feature-a = {X.shape[2]} po frejmu")

    n_samples, n_frames, n_features = X.shape
    X_2d = X.reshape(n_samples, n_frames * n_features)

    scaler = MinMaxScaler()
    X_norm_2d = scaler.fit_transform(X_2d)
    X_norm = X_norm_2d.reshape(n_samples, n_frames, n_features)

    print(f"Normalizacija (MinMax) gotova. Min: {X_norm.min():.2f}, Max: {X_norm.max():.2f}")

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

    print(f"\nPodela dataseta:")
    print(f"  Trening: {X_train.shape[0]} snimaka")
    print(f"  Validacija: {X_val.shape[0]} snimaka")
    print(f"  Test   : {X_test.shape[0]} snimaka")

    os.makedirs("models", exist_ok=True)
    np.save("models/X_train_v3.npy", X_train)
    np.save("models/X_val_v3.npy", X_val)
    np.save("models/X_test_v3.npy", X_test)
    np.save("models/y_train_v3.npy", y_train)
    np.save("models/y_val_v3.npy", y_val)
    np.save("models/y_test_v3.npy", y_test)
    np.save("models/classes_v3.npy", le.classes_)
    joblib.dump(scaler, "models/scaler_v3.pkl")

    print(f"\nSvi fajlovi (V3) sacuvani u 'models/' folder!")
    print(f"Feature broj po frejmu: {n_features}")