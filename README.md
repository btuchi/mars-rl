# Martian Terrain Diffusion with Diversity-Oriented Reinforcement Learning

This project explores how to fine-tune diffusion models (e.g., Stable Diffusion) to generate synthetic Martian terrain images — such as craters — while encouraging **diversity** across features like size, lighting, morphology, and context. The goal is to use **reinforcement learning (RL)** to improve sample quality by maximizing a diversity reward function (e.g., Maximum Mean Discrepancy, MMD).

## 🚀 Project Goals

- Fine-tune a diffusion model to generate diverse and realistic Martian crater images.
- Define and compute **feature-based diversity rewards** (e.g., MMD, GP-MI).
- Train using **policy gradient RL**, treating each denoising step in the diffusion trajectory as an action.
- Evaluate how well generated images span the reference space of real crater images.


## 🧪 Project Plan Overview

This project follows a step-by-step pipeline to fine-tune a diffusion model for Martian terrain image generation with a focus on visual diversity.

### 🔹 Step 1: Reference Dataset Preparation

We curate ~68 real Martian crater images per category, making sure they cover characteristics such as:
- Size (small to massive)
- Hue, Contrast, Saturation (hsv)


### 🔹 Step 2: Feature Extraction Pipeline

We use a CLIP visual encoder to extract 512-dimensional feature vectors from each reference image. These features are stored as `.npy` files for reuse.

Once synthetic images are generated, we extract features from them too and compare them to the reference set — enabling reward calculation later.

### 🔹 Step 3: Diversity Reward Function

We implement MMD (Maximum Mean Discrepancy) as our reward function:
- **MMD (Maximum Mean Discrepancy)**: measures how closely the generated feature distribution matches the reference distribution.

We validate the reward by testing known datasets with increasing diversity levels (e.g., all identical → highly varied craters), and check if the reward increases accordingly.

### 🔹 Step 4: Individual Reward Estimation

We compute marginal utility for each generated image:
```math
\text{Reward}(x_i) = \text{DiversityReward}(X) - \text{DiversityReward}(X \setminus \{x_i\})

```

This tells us which individual images contribute most to the overall diversity. We can rank images and identify duplicates or particularly novel outputs.

### 🔹 Step 5: RL Fine-tuning Framework

We modify the diffusion sampler to record **trajectories** — the full sequence of denoising steps taken to generate each image.

Then we apply **Proximal Policy Optimization (PPO) RL**:
- Good (diverse) images get higher reward → increase likelihood of those denoising steps
- Low-reward images → discourage similar actions

To stabilize training, we plan to use:
- **Trust region constraints** to limit drastic model updates
- **LoRA (Low-Rank Adaptation)** for efficient fine-tuning of large diffusion models

### 🔹 Step 6: Training Loop Implementation

We build a full training loop that:
- Generates batches of images
- Computes rewards
- Logs reward values over time
- Saves model checkpoints

### 🔹 Step 7: Evaluation System (TBD)

To assess improvement, we compute:
- **Quantitative metrics**: FID, Precision, Recall, KL divergence, MMD
- **Visual metrics**: t-SNE plots, feature space clustering
- **Baseline comparison**: Check if the fine-tuned model generates more modes and more distinct images than the original

This evaluation confirms if the model is learning to generate **diverse, high-quality crater images** aligned with scientific variability observed in real Martian terrain.

## ✅ Research Progress & Implementation Status

### 🎯 **Core Infrastructure** 
- ✅ **Trajectory Recording**: Complete denoising step logging and action tracking
- ✅ **GPU Pipeline**: Verified Stable Diffusion sampling on Bridges2 (V100/H100)
- ✅ **Feature Extraction**: CLIP-based 512D feature vectors from generated images
- ✅ **Reward System**: Multiple diversity metrics (MMD, MI, FID, combined)

### 🧠 **Reinforcement Learning Algorithms**

#### **PPO Implementation** (`ppo_diffusion/`)
- ✅ **Complete PPO Agent**: Full Proximal Policy Optimization with clipping
- ✅ **Multi-Modal Training**: Three training modes implemented:
  - `SCHEDULER_POLICY`: Controls β schedules and guidance scales
  - `DIVERSITY_POLICY`: Controls prompt diversity and generation parameters  
  - `LORA_UNET`: Fine-tunes U-Net layers using LoRA adaptation
- ✅ **Advanced Features**:
  - Gaussian/Laplacian prior distributions
  - GAE (Generalized Advantage Estimation)
  - Gradient clipping and flow preservation
  - Memory replay buffer with trajectory storage
- ✅ **Reward Metrics**: MMD, MI, FID, and weighted combinations
- 🐛 **Status**: Encountered zero gradient issues in scheduler policy training

#### **TRPO Implementation** (`trpo_diffusion/`)
- ✅ **Complete TRPO Agent**: Trust Region Policy Optimization with natural gradients
- ✅ **Mathematical Components**:
  - Fisher Information Matrix computation via KL divergence Hessian
  - Conjugate gradient solver for natural policy gradients
  - Line search with backtracking for optimal step sizes
  - Trust region constraints (KL divergence ≤ 0.01)
- ✅ **Policy Updates**: Proper old/new policy comparison for trust regions
- ✅ **Same Multi-Modal Support**: Scheduler, Diversity, and LoRA modes
- 📝 **Status**: Fully implemented but not tested (final day limitation)

### 📊 **Reward & Evaluation Systems**
- ✅ **Multiple Metrics**: MMD, Mutual Information, FID scores
- ✅ **Individual Rewards**: Per-image diversity contribution calculation
- ✅ **Weighted Combinations**: Configurable MMD+MI hybrid rewards
- ✅ **Feature Analysis**: ResNet-18 feature extraction pipeline
- ✅ **Logging & Visualization**: Comprehensive training metrics and plots

### 🔧 **Development & Deployment**
- ✅ **Sync Scripts**: Automated Bridges2 ↔ Local synchronization for both PPO/TRPO
- ✅ **Job Management**: SLURM batch scripts with GPU resource allocation
- ✅ **Memory Management**: GPU cache clearing and efficient tensor handling
- ✅ **Error Handling**: Anomaly detection and gradient flow debugging

## 🔬 **Key Research Findings**

### **Algorithm Comparison**
- **PPO Challenges**: Zero gradient issues in scheduler policy mode despite multiple debugging attempts
- **TRPO Promise**: More stable gradient flow through natural gradients and trust regions
- **Multi-Modal Success**: Diversity and LoRA modes worked reliably across both algorithms

### **Reward Metric Insights** 
- **MMD**: Effective for feature distribution matching
- **Mutual Information**: Better for image-level diversity assessment
- **FID**: Standard quality metric but computationally expensive
- **Hybrid Approaches**: MMD+MI combinations showed promise

### **Technical Discoveries**
- **Gradient Flow**: Critical to preserve gradients through trajectory generation
- **Policy Priors**: Gaussian distributions more stable than Laplacian
- **Memory Management**: Essential for large-scale diffusion model training
- **Trust Regions**: TRPO's mathematical constraints may solve PPO instability

## 🚀 **Future Research Directions**

1. **Test TRPO Implementation**: Validate natural gradient approach vs PPO
2. **Advanced Reward Functions**: Implement LPIPS and perceptual metrics  
3. **Multi-GPU Scaling**: Distribute training across multiple GPUs
4. **Evaluation Framework**: Comprehensive FID/IS/Precision/Recall metrics
5. **Real Dataset Validation**: Test on larger Martian terrain datasets
6. **Hyperparameter Optimization**: Systematic tuning of trust region constraints

## 📦 Dependencies

- `diffusers`
- `torch`, `transformers`
- `numpy`, `PIL`, `clip-by-openai`
- GPU (V100 or H100 preferred)
- Optional: SLURM + Conda for HPC (Bridges2)

Output includes:

* GPU availability
* Trajectory length
* CLIP feature vector shape
* Diversity reward value

## 📬 Contact

- Maintained by **Bryce Tu Chi** ([brycetuchi@gmail.com](mailto:brycetuchi@gmail.com))
- Instructor: Dr. Adyasha Mohanty


