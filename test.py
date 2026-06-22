import numpy as np

file_path = "data/A/1702281490252934-XVID.npy"

data = np.load(file_path, allow_pickle=True)

print(f"Data type: {type(data)}")
print(f"Shape: {data.shape}")
print(f"dtype: {data.dtype}")
print(f"First value: {data[0]}")
print(f"Type of the first value: {type(data[0])}")
print(f"Shape of the first value: {np.array(data[0]).shape}")