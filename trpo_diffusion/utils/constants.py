# File: utils/constants.py
"""Global constants and configuration for TRPO Diffusion"""

# Training configuration
DEFAULT_CATEGORY = "crater"
DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"

# TRPO hyperparameters (different from PPO)
# Note: TRPO doesn't use actor learning rate (uses natural gradients)
DEFAULT_LR_CRITIC = 1e-3
DEFAULT_GAMMA = 0.9
DEFAULT_LAMBDA = 0.95

# TRPO-specific parameters
DEFAULT_KL_TARGET = 0.01        # KL divergence constraint
DEFAULT_DAMPING = 0.1           # Damping parameter for Fisher matrix
DEFAULT_CG_ITERS = 10           # Conjugate gradient iterations
DEFAULT_BACKTRACK_ITERS = 10    # Line search iterations
DEFAULT_BACKTRACK_COEFF = 0.8   # Line search coefficient
DEFAULT_ACCEPT_RATIO = 0.1      # Minimum improvement ratio

# Training parameters
DEFAULT_NUM_EPISODES = 10000
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPISODES_PER_UPDATE = 1
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_IMAGES_PER_PROMPT = 4

# Image settings
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_GUIDANCE_SCALE = 8.0

# Memory optimization
USE_MEMORY_EFFICIENT_ATTENTION = True
USE_FP16 = False

# Logging
LOG_SAVE_FREQUENCY = 5

# Reward configuration
DEFAULT_REWARD_METRIC = "MMD"  # MMD, MI, MMD_MI, FID

# FID reward scaling
FID_REWARD_SCALE = 0.1

# MMD_MI combination weights
MMD_WEIGHT = 0.7
MI_WEIGHT = 0.3

# Multi-component reward weights
USE_MULTI_COMPONENT_REWARD = True
SEQUENTIAL_DIVERSITY_WEIGHT = 0.3
SPATIAL_DIVERSITY_WEIGHT = 0.2
ENTROPY_REWARD_WEIGHT = 0.1
SEQUENTIAL_THRESHOLD = 0.7

# TRAINING MODE CONFIGURATION
# Available options: "DIVERSITY_POLICY", "LORA_UNET", "SCHEDULER_POLICY"
DEFAULT_TRAINING_MODE = "SCHEDULER_POLICY"