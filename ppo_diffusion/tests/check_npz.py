import numpy as np

# Check what's in the npz file
file_path = "/Users/bryce2hua/Desktop/RL/ppo_diffusion/reference_features/reference_crater_features.npz"
npz_data = np.load(file_path)
print("Keys in the npz file:", list(npz_data.keys()))

for key in npz_data.keys():
    print(f"Key '{key}': shape {npz_data[key].shape}, dtype {npz_data[key].dtype}")

npz_data.close()