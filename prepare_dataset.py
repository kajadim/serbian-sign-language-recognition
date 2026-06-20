import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from load_and_explore_data import load_dataset, FRAME_SIZE, DATA_PATH
import joblib

if __name__ == "__main__":
    print("Ucitavanje dataseta..")
    X_raw, y_raw = load_dataset(DATA_PATH)
    print(f"Ukupno snimaka: {len(X_raw)}")
    X = np.array(X_raw, dtype=np.float32)
    print(f"Oblik X: {X.shape}")

    n_samples , n_frames, n_features = X.shape
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
    print(f"  Trening: {X_val.shape[0]} snimaka")
    print(f"  Test   : {X_test.shape[0]} snimaka")

    os.makedirs("models", exist_ok=True)
    np.save("models/X_train.npy", X_train)
    np.save("models/X_val.npy", X_val)
    np.save("models/X_test.npy",  X_test)
    np.save("models/y_train.npy", y_train)
    np.save("models/y_val.npy", y_val)
    np.save("models/y_test.npy",  y_test)
    np.save("models/classes.npy", le.classes_)
    joblib.dump(scaler, "models/scaler.pkl")
    
    print(f"\nSvi fajlovi sacuvani u 'models/' folder!")