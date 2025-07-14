# File: utils/constants.py
"""Global constants and configuration"""

# Training configuration
DEFAULT_CATEGORY = "crater"
DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"

# PPO hyperparameters
DEFAULT_LR_ACTOR = 1e-4
DEFAULT_LR_CRITIC = 1e-4
DEFAULT_GAMMA = 0.9
DEFAULT_LAMBDA = 0.95
DEFAULT_EPSILON_CLIP = 0.1

# Training parameters
DEFAULT_NUM_EPISODES = 10000
DEFAULT_BATCH_SIZE = 4  # Reduced from 4 to save memory
DEFAULT_EPISODES_PER_UPDATE = 1
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_IMAGES_PER_PROMPT = 4  # Reduced from 4 to save memory

# Image settings
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_GUIDANCE_SCALE = 8.0

# Memory optimization
USE_MEMORY_EFFICIENT_ATTENTION = True
USE_FP16 = False  # Can enable for more memory savings

# Logging
LOG_SAVE_FREQUENCY = 5

# Available options: "MMD", "MI", "MMD_MI", "FID" (Not yet implemented), "LPIPS" (Not yet implemented)
DEFAULT_REWARD_METRIC = "MI"  # Switch between MMD, MI, or MMD_MI easily

# MMD_MI combination weights (only used when DEFAULT_REWARD_METRIC = "MMD_MI")
MMD_WEIGHT = 0.7  # Weight for diversity (MMD component)
MI_WEIGHT = 0.3   # Weight for prompt relevance (MI component)

# TRAINING MODE CONFIGURATION
# Available options: "DIVERSITY_POLICY", "LORA_UNET"
DEFAULT_TRAINING_MODE = "DIVERSITY_POLICY"  # Switch between training approaches