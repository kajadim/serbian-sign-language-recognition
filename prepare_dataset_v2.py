"""
Feature engineering za M/N problem.

Cilj: dodati eksplicitne, izracunate karakteristike "savijenosti" (curl) prstiju
kao dodatne kolone u podatke, kako bi model lakse naucio finu razliku
(npr. da li je srednji prst leve ruke savijen ili ne).

Ovo NE menja postojece podatke nego dodaje NOVE feature pored postojecih 258
po frejmu, pa input modela postaje (40, 258 + N_novih_feature).

Pokrenuti OVO UMESTO obicnog prepare_dataset.py:
    python prepare_dataset_v2.py

Zatim trenirati NOVI model (jer se menja input shape):
    python train_model_v2.py
"""

import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from load_and_explore_data import load_dataset, DATA_PATH
import joblib

# ============================================================
# MEDIAPIPE HAND LANDMARK INDEKSI (unutar 21 tacke jedne ruke)
# ============================================================
WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Layout jednog frejma (258 vrednosti):
#   [0:132]   pose  (33 tacke x 4: x,y,z,visibility)
#   [132:195] leva ruka  (21 tacke x 3: x,y,z)
#   [195:258] desna ruka (21 tacke x 3: x,y,z)
POSE_LEN = 33 * 4
HAND_LEN = 21 * 3
LEFT_HAND_START = POSE_LEN
RIGHT_HAND_START = POSE_LEN + HAND_LEN


def get_hand_landmarks(frame, hand_start):
    """Izvuci (21, 3) niz koordinata za jednu ruku iz 258-vrednosnog frejma."""
    hand_flat = frame[hand_start: hand_start + HAND_LEN]
    return hand_flat.reshape(21, 3)


def finger_curl(landmarks, tip_idx, mcp_idx, wrist_idx=WRIST):
    """
    Izracunaj 'curl' (savijenost) prsta kao odnos:
        rastojanje(tip, wrist) / rastojanje(mcp, wrist)
    Ako je prst ispravljen, tip je daleko od wrist-a -> odnos je veci (>1.5 obicno).
    Ako je prst savijen (pesnica), tip je blizu wrist-a -> odnos je manji (~0.5-1).

    Vraca 0.0 ako su koordinate sve nule (ruka nije detektovana).
    """
    wrist = landmarks[wrist_idx]
    tip = landmarks[tip_idx]
    mcp = landmarks[mcp_idx]

    if np.all(landmarks == 0):
        return 0.0

    dist_tip_wrist = np.linalg.norm(tip - wrist)
    dist_mcp_wrist = np.linalg.norm(mcp - wrist) + 1e-6  # izbegni deljenje nulom

    return dist_tip_wrist / dist_mcp_wrist


def compute_curl_features(frame):
    """
    Za jedan frejm (258,) izracunaj curl feature za svih 5 prstiju
    na obe ruke -> 10 novih vrednosti.
    Redosled: [L_thumb, L_index, L_middle, L_ring, L_pinky,
               R_thumb, R_index, R_middle, R_ring, R_pinky]
    """
    left_hand = get_hand_landmarks(frame, LEFT_HAND_START)
    right_hand = get_hand_landmarks(frame, RIGHT_HAND_START)

    curls = []
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
        curls.append(finger_curl(left_hand, tip, mcp))
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
        curls.append(finger_curl(right_hand, tip, mcp))

    return np.array(curls, dtype=np.float64)  # (10,)


def add_curl_features_to_sequence(sequence):
    """
    sequence: (n_frames, 258)
    Vraca: (n_frames, 258 + 10) sa dodatim curl feature-ima po frejmu.
    """
    n_frames = sequence.shape[0]
    curl_features = np.array([compute_curl_features(frame) for frame in sequence])  # (n_frames, 10)
    return np.concatenate([sequence, curl_features], axis=1)  # (n_frames, 268)


if __name__ == "__main__":
    print("Ucitavanje dataseta..")
    X_raw, y_raw = load_dataset(DATA_PATH)
    print(f"Ukupno snimaka: {len(X_raw)}")

    print("\nRacunanje curl (savijenost prstiju) feature-a za svaki snimak...")
    print("(Ovo moze potrajati minut-dva)")

    X_with_curl = []
    for i, seq in enumerate(X_raw):
        seq_with_curl = add_curl_features_to_sequence(seq)
        X_with_curl.append(seq_with_curl)
        if (i + 1) % 1000 == 0:
            print(f"  Obradjeno {i + 1}/{len(X_raw)}...")

    X = np.array(X_with_curl, dtype=np.float32)
    print(f"\nOblik X (sa curl feature-ima): {X.shape}")
    print(f"  Originalno: 258 vrednosti po frejmu")
    print(f"  Sada: {X.shape[2]} vrednosti po frejmu (258 + 10 curl feature-a)")

    n_samples, n_frames, n_features = X.shape
    X_2d = X.reshape(n_samples, n_frames * n_features)

    scaler = MinMaxScaler()
    X_norm_2d = scaler.fit_transform(X_2d)
    X_norm = X_norm_2d.reshape(n_samples, n_frames, n_features)

    print(f"Normalizacija gotova. Min: {X_norm.min():.2f}, Max: {X_norm.max():.2f}")

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
    np.save("models/X_train_v2.npy", X_train)
    np.save("models/X_val_v2.npy", X_val)
    np.save("models/X_test_v2.npy", X_test)
    np.save("models/y_train_v2.npy", y_train)
    np.save("models/y_val_v2.npy", y_val)
    np.save("models/y_test_v2.npy", y_test)
    np.save("models/classes_v2.npy", le.classes_)
    joblib.dump(scaler, "models/scaler_v2.pkl")

    print(f"\nSvi fajlovi (V2 - sa curl feature-ima) sacuvani u 'models/' folder!")
    print(f"Novi feature broj po frejmu: {n_features} (umesto 258)")