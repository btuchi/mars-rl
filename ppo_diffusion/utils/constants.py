# File: utils/constants.py
"""Global constants and configuration"""

# Training configuration
DEFAULT_CATEGORY = "crater"
DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"

# PPO hyperparameters
DEFAULT_LR_ACTOR = 1e-3  # Increased for LoRA (was 1e-4)
DEFAULT_LR_CRITIC = 1e-3  # Increased to match actor (was 1e-4)
DEFAULT_GAMMA = 0.9
DEFAULT_LAMBDA = 0.95
DEFAULT_EPSILON_CLIP = 0.1
DEFAULT_ENTROPY_COEFF = 0.0  # PPO entropy bonus disabled temporarily

# Training parameters
DEFAULT_NUM_EPISODES = 10000
DEFAULT_BATCH_SIZE = 1  # Reduced from 4 to save memory
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

# Available options: "MMD", "MI", "MMD_MI", "FID", "LPIPS" (Not yet implemented)
DEFAULT_REWARD_METRIC = "MMD"  # Switch between MMD, MI, MMD_MI, or FID easily

# FID reward scaling (only used when DEFAULT_REWARD_METRIC = "FID")
FID_REWARD_SCALE = 0.1  # Scale factor for FID reward (negative FID)

# MMD_MI combination weights (only used when DEFAULT_REWARD_METRIC = "MMD_MI")
MMD_WEIGHT = 0.7  # Weight for diversity (MMD component)
MI_WEIGHT = 0.3   # Weight for prompt relevance (MI component)

# Multi-component reward weights (adds diversity bonuses to base metric)
USE_MULTI_COMPONENT_REWARD = True  # Enable/disable multi-component rewards
SEQUENTIAL_DIVERSITY_WEIGHT = 0.3   # Weight for temporal diversity (anti-repetition)
SPATIAL_DIVERSITY_WEIGHT = 0.2      # Weight for intra-image spatial diversity  
ENTROPY_REWARD_WEIGHT = 0.1         # Weight for feature entropy bonus
SEQUENTIAL_THRESHOLD = 0.7          # Cosine similarity threshold for sequential diversity

# TRAINING MODE CONFIGURATION
# Available options: "DIVERSITY_POLICY", "LORA_UNET", "SCHEDULER_POLICY"
DEFAULT_TRAINING_MODE = "SCHEDULER_POLICY"  # Switch between training approaches
