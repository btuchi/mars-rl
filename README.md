# Martian Terrain Diffusion with Diversity-Oriented Reinforcement Learning

This project explores how to fine-tune diffusion models (e.g., Stable Diffusion) to generate synthetic Martian terrain images — such as craters — while encouraging **diversity** across features like size, lighting, morphology, and context. The goal is to use **reinforcement learning (RL)** to improve sample quality by maximizing a diversity reward function (e.g., Maximum Mean Discrepancy, MMD).

## 🚀 Project Goals

- Fine-tune a diffusion model to generate diverse and realistic Martian crater images.
- Define and compute **feature-based diversity rewards** (e.g., MMD, GP-MI).
- Train using **policy gradient RL**, treating each denoising step in the diffusion trajectory as an action.
- Evaluate how well generated images span the reference space of real crater images.

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

