import numpy as np

file_path = "data/A/1702281490252934-XVID.npy"

data = np.load(file_path, allow_pickle=True)

print(f"Tip podataka: {type(data)}")
print(f"Oblik (shape): {data.shape}")
print(f"dtype: {data.dtype}")
print(f"Prva vrednost: {data[0]}")
print(f"Tip prve vrednosti: {type(data[0])}")
print(f"Oblik prve vrednosti: {np.array(data[0]).shape}")