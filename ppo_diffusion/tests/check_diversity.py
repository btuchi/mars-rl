import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


ref_features = np.load("ppo_diffusion/reference_crater_features.npz")
ref_array = np.stack([ref_features[key] for key in
ref_features.keys()])

# Calculate pairwise similarities
similarities = cosine_similarity(ref_array)
print(f"Reference feature diversity: {similarities.std():.4f}")
print(f"Min similarity: {similarities.min():.4f}")
print(f"Max similarity: {similarities.max():.4f}")