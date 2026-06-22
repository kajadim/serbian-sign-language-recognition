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
print(f"Detected classes: {LABELS}")
print(f"Number of classes: {len(LABELS)}")


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

        print(f"Loading {label}: {len(files)} files...")

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            try:
                frames = load_npy_file(file_path)

                if len(frames) > 0 :
                    X.append(frames)
                    y.append(label)

            except Exception as e :
                print (f"Error at file {file_name}: {e}")

    print(f"Total loaded recordings: {len(X)}")
    return X, y
    
def analyze_dataset(X, y):
    class_counts = Counter(y)

    for label, count in sorted(class_counts.items()):
        print(f"{label:4s}:{count} recordings")

    frame_counts = [len(frames) for frames in X]
    print("Number of frames per recording")
    print(f"Min: {min(frame_counts)}")
    print(f"Max: {max(frame_counts)}")
    print(f"Average: {np.mean(frame_counts):.1f}")

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    labels_sorted = sorted(class_counts.keys())
    counts_sorted = [class_counts[l] for l in labels_sorted]
    plt.bar(labels_sorted, counts_sorted, color='steelblue')
    plt.title('Number of recordings per letter')
    plt.xlabel('Letter')
    plt.ylabel('Number of recordings')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.hist(frame_counts, bins=20, color='coral', edgecolor='black')
    plt.title('Recording length distribution (number of frames)')
    plt.xlabel('Number of frames')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    plt.show()

if __name__ == "__main__" :
    print("Loading dataset...")

    X, y = load_dataset(DATA_PATH, max_files_per_class=None)

    analyze_dataset(X, y)
    print(f"\nExample of the first recording:")
    print(f"  Letter: {y[0]}")
    print(f"  Frames: {X[0].shape} (number_of_frames x 258 values)")
    print(f"  First 3 values of the first frame: {X[0][0][:3]}")