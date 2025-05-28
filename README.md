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

We implement two reward functions:
- **MMD (Maximum Mean Discrepancy)**: measures how closely the generated feature distribution matches the reference distribution.
- **GP-MI (Gaussian Process Mutual Information)**: captures mutual coverage between generated and reference features.

We validate the reward by testing known datasets with increasing diversity levels (e.g., all identical → highly varied craters), and check if the reward increases accordingly.

### 🔹 Step 4: Individual Reward Estimation

We compute marginal utility for each generated image:
```math
\text{Reward}(x_i) = \text{DiversityReward}(X) - \text{DiversityReward}(X \setminus \{x_i\})

```

This tells us which individual images contribute most to the overall diversity. We can rank images and identify duplicates or particularly novel outputs.

### 🔹 Step 5: RL Fine-tuning Framework

We modify the diffusion sampler to record **trajectories** — the full sequence of denoising steps taken to generate each image.

Then we apply **policy gradient RL**:
- Good (diverse) images get higher reward → increase likelihood of those denoising steps
- Low-reward images → discourage similar actions
- Loss: `-log(probability of trajectory) × reward`

To stabilize training, we plan to use:
- **Trust region constraints** to limit drastic model updates
- **LoRA (Low-Rank Adaptation)** for efficient fine-tuning of large diffusion models

### 🔹 Step 6: Training Loop Implementation

We build a full training loop that:
- Generates batches of images
- Computes rewards
- Logs reward values over time
- Saves model checkpoints
- Uses evaluation metrics like FID, Precision, and Recall to monitor diversity

### 🔹 Step 7: Evaluation System

To assess improvement, we compute:
- **Quantitative metrics**: FID, Precision, Recall, KL divergence, MMD
- **Visual metrics**: t-SNE plots, feature space clustering
- **Baseline comparison**: Check if the fine-tuned model generates more modes and more distinct images than the original

This evaluation confirms if the model is learning to generate **diverse, high-quality crater images** aligned with scientific variability observed in real Martian terrain.


## 🧱 Directory Structure

```

RL/
│
├── by\_size/, by\_lighting/, generated\_craters/, real\_craters/, reference\_images/
│   └── Image subfolders grouped by crater characteristics
│
├── tests/
│   ├── simple\_test.py              # Quick end-to-end test of model + reward pipeline
│   └── test\_trajectory\_recording.py
│
├── trajectory\_recording.py        # Sampling logic with step-by-step logging for RL
├── diversity\_reward.py            # MMD-based reward calculation
├── build\_reference\_images.ipynb   # Notebook for collecting and grouping reference data
├── diversity\_reward\_MMD.ipynb     # Notebook for testing MMD and other reward functions
├── feature\_extraction.ipynb       # Notebook for extracting CLIP features
├── reference\_clip\_features.npz    # Saved CLIP features of reference images
│
├── simple\_test.sh                 # SLURM script to run simple\_test.py on Bridges2
├── clean\_up\_bridges2.sh           # Sync and overwrite Bridges2 files
├── sync\_to\_bridges2.sh            # rsync project to PSC

````

## ✅ Current Progress

- ✅ Set up trajectory recorder to log all denoising steps and actions.
- ✅ Verified GPU-enabled Stable Diffusion sampling on Bridges2.
- ✅ Extracted CLIP-based features from generated images.
- ✅ Implemented and tested MMD reward function using synthetic and real feature vectors.
- ✅ Conducted full pipeline smoke test (`simple_test.py`) — **passed on GPU node**.
- 🔜 Next: Implement training loop with policy gradient updates using recorded trajectories.

## 📦 Dependencies

- `diffusers`
- `torch`, `transformers`
- `numpy`, `PIL`, `clip-by-openai`
- GPU (V100 or H100 preferred)
- Optional: SLURM + Conda for HPC (Bridges2)

## 🧪 Running Tests

To run a quick end-to-end test:
```bash
sbatch simple_test.sh
````

Output includes:

* GPU availability
* Trajectory length
* CLIP feature vector shape
* Diversity reward value

## 📬 Contact

Maintained by **Bryce Tu Chi**
Email: [btuchi@g.hmc.edu](mailto:btuchi@g.hmc.edu)

