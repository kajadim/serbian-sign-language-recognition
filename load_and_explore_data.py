import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

POSE_POINTS = 33 * 4
LEFT_HAND_PTS = 21 * 3
RIGHT_HAND_PTS = 21 * 3
FRAME_SIZE = POSE_POINTS + LEFT_HAND_PTS + RIGHT_HAND_PTS

DATA_PATH = "data"

LABELS = sorted([
    f for f in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, f)) and not f.startswith('.')
    ])
print(f"Pronadjene klase: {LABELS}")
print(f"Broj klasa: {len(LABELS)}")


def load_npy_file(file_path):
    data = np.load(file_path, allow_pickle= True)


    data = data.flatten().astype(np.float64)

    num_frames = len(data) // FRAME_SIZE
    frames = []

    for i in range(num_frames):
        frame = data[i * FRAME_SIZE : ( i + 1 ) * FRAME_SIZE]
        if len(frame) == FRAME_SIZE:
            frames.append(frame)
    
    return np.array(frames)


def load_dataset(data_path, max_files_per_class = None):
    X = []
    y = []

    labels = sorted([
        f for f in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, f)) and not f.startswith('.')
        ])

    for label in labels :
        folder_path = os.path.join(data_path, label)

        if not os.path.isdir(folder_path):
            continue

        files = [ f for f in os.listdir(folder_path) if f.endswith('.npy')]

        if max_files_per_class:
            files = files[:max_files_per_class]

        print(f"Ucitaval {label} : {len((files))} fajlova...")

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            try:
                frames = load_npy_file(file_path)

                if len(frames) > 0 :
                    X.append(frames)
                    y.append(label)

            except Exception as e :
                print (f"Error at file {file_name}: {e}")

    print(f"Ukupno ucitano : {len(X)} snimaka")
    return X, y
    
def analyze_dataset(X, y):
    class_counts = Counter(y)

    for label, count in sorted(class_counts.items()):
        print(f"{label:4s}:{count} snimaka")

    frame_counts = [len(frames) for frames in X]
    print(f"Broj frejmova po snimku")
    print(f"Min: {min(frame_counts)}")
    print(f"Max: {max(frame_counts)}")
    print(f"Prosecno: {np.mean(frame_counts):.1f}")

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    labels_sorted = sorted(class_counts.keys())
    counts_sorted = [class_counts[l] for l in labels_sorted]
    plt.bar(labels_sorted, counts_sorted, color='steelblue')
    plt.title('Broj snimaka po slovu')
    plt.xlabel('Slovo')
    plt.ylabel('Broj snimaka')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.hist(frame_counts, bins=20, color='coral', edgecolor='black')
    plt.title('Distribucija dužine snimaka (broj frejmova)')
    plt.xlabel('Broj frejmova')
    plt.ylabel('Frekvencija')
    
    plt.tight_layout()
    plt.savefig('dataset_analiza.png')
    plt.show()

if __name__ == "__main__" :
    print("Ucitavanje dataseta")
    print("Za prvo testiranje ucitavamo 10 fajlova po klasi")

    X, y = load_dataset(DATA_PATH, max_files_per_class=None)

    analyze_dataset(X, y)
    print(f"\nPrimer prvog snimka:")
    print(f"  Slovo  : {y[0]}")
    print(f"  Frejmovi: {X[0].shape}  (broj_frejmova x 258 vrednosti)")
    print(f"  Prve 3 vrednosti prvog frejma: {X[0][0][:3]}")