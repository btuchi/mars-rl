# PPO-Diffusion Codebase: Complete Technical Walkthrough

This document provides a comprehensive analysis of the `ppo_diffusion/` codebase, explaining the architecture, components, and training pipeline for fine-tuning Stable Diffusion models using Proximal Policy Optimization (PPO) to maximize visual diversity in generated Mars crater images.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Training Pipeline](#training-pipeline)
5. [Configuration System](#configuration-system)
6. [Data Flow](#data-flow)
7. [Key Algorithms](#key-algorithms)
8. [Extensibility](#extensibility)

---

## 🏗️ Architecture Overview

The PPO-diffusion system implements a **reinforcement learning approach to fine-tune diffusion models** for generating diverse visual content. The key innovation is treating the diffusion denoising process as a Markov Decision Process (MDP) where:

- **States**: Current noise level and prompt context
- **Actions**: Denoising parameters (scheduler settings, guidance scales, latent modifications)
- **Rewards**: Visual diversity metrics (MMD, MI, FID)
- **Policy**: Neural networks controlling diffusion parameters

### Core Philosophy

```
Reference Images → Feature Extraction → Diversity Rewards → PPO Training → Fine-tuned Model
```

The system learns to generate images that maximize diversity while maintaining prompt relevance, using multiple configurable reward metrics and training modes.

---

## 📁 Directory Structure

```
ppo_diffusion/
├── Entry Points
│   ├── train.py                    # Main training script
│   ├── requirements.txt            # Dependencies
│   ├── build_reference_features.py # Feature preprocessing
│   ├── create_reference_images_npz.py # Data preparation
│   └── plot_reference_features.py  # Visualization utilities
│
├── Core Architecture
│   ├── core/
│   │   ├── __init__.py
│   │   ├── features.py             # ResNet-18 visual feature extractor
│   │   └── trajectory.py           # Diffusion sampling & trajectory recording
│   │
│   └── models/
│       ├── __init__.py
│       ├── policy.py               # Policy networks (3 modes)
│       └── value.py                # Critic network for PPO
│
├── Training System
│   └── training/
│       ├── agent.py                # Main PPO agent
│       ├── memory.py               # Experience replay buffer
│       ├── rewards.py              # Multi-component reward system
│       └── reward_metrics/         # Pluggable reward metrics
│           ├── __init__.py         # Metric registry
│           ├── mmd.py             # Maximum Mean Discrepancy
│           ├── mi.py              # Mutual Information
│           └── fid.py             # Fréchet Inception Distance
│
├── Utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py            # Global configuration
│   │   ├── device.py              # GPU optimization
│   │   ├── logging.py             # CSV logging system
│   │   └── visualization.py       # Training plots
│   │
│   └── tests/                     # Unit tests & validation
│       ├── test_*.py              # Unit tests
│       ├── check_*.py             # Validation scripts
│       └── plot_*.py              # Visualization tests
│
└── Data & Results
    ├── prompts/                   # Training prompts
    │   ├── train/crater.txt       # 25 diverse Mars crater descriptions
    │   └── test/crater.txt        # Test prompts
    ├── reference_images/          # Mars crater dataset
    │   └── crater/                # 73 Mars crater reference images
    ├── reference_features/        # Pre-computed features
    │   ├── reference_crater_features_v*.npz
    │   └── reference_crater_images.npz
    └── outputs/                   # Generated results
        ├── images/               # Sample images
        ├── logs/                 # Training logs
        ├── models/               # Checkpoints
        └── plots/                # Visualizations
```

---

## ⚙️ Core Components

### 1. Training Entry Point (`train.py`)

**Location**: `ppo_diffusion/train.py` (Lines 1-200+)  
**Purpose**: Main training orchestrator and configuration hub

```python
# Key responsibilities:
def main():
    # 1. Load reference data (images + features)
    ref_features, ref_images = load_reference_data()
    
    # 2. Initialize diffusion pipeline
    sampler = DiffusionSampler(model_id="runwayml/stable-diffusion-v1-5")
    
    # 3. Create PPO agent with configurable training mode
    agent = DiffusionPPOAgent(
        sampler=sampler,
        ref_features=ref_features,
        training_mode=DEFAULT_TRAINING_MODE,  # SCHEDULER_POLICY, DIVERSITY_POLICY, LORA_UNET
        reward_metric=DEFAULT_REWARD_METRIC   # MMD, MI, FID, MMD_MI
    )
    
    # 4. Run training loop
    train_agent(agent, num_episodes=DEFAULT_NUM_EPISODES)
```

**Key Features**:
- Configurable training modes and reward metrics
- Automatic reference data loading from NPZ files
- Comprehensive error handling and recovery
- Integration with HPC job systems (SLURM)

### 2. Diffusion Trajectory System (`core/trajectory.py`)

**Location**: `ppo_diffusion/core/trajectory.py` (Lines 1-800+)  
**Purpose**: Records complete diffusion sampling process for RL training

```python
class DiffusionTrajectory:
    """Records every denoising step as (state, action, next_state)"""
    
    def __init__(self):
        self.states = []        # Noise levels at each step
        self.actions = []       # Denoising parameters used
        self.log_probs = []     # Policy probabilities
        self.rewards = []       # Step-wise rewards
        
class DiffusionSampler:
    """Enhanced diffusion pipeline with trajectory recording"""
    
    def sample_with_trajectory(self, prompt, policy_network):
        trajectory = DiffusionTrajectory()
        
        for t in scheduler.timesteps:
            # Record current state
            trajectory.states.append(latents.clone())
            
            # Get action from policy
            action, log_prob = policy_network.get_action(latents, t, prompt)
            trajectory.actions.append(action)
            trajectory.log_probs.append(log_prob)
            
            # Apply denoising step
            latents = self.unet(latents, t, text_embeddings).sample
            
        return final_image, trajectory
```

**Innovation**: This system captures the complete denoising process as an RL trajectory, enabling policy gradient methods to optimize diffusion parameters.

### 3. Policy Networks (`models/policy.py`)

**Location**: `ppo_diffusion/models/policy.py` (Lines 1-600+)  
**Purpose**: Three distinct policy types for different aspects of diffusion control

#### **Policy Type 1: Scheduler Policy** (Lines 50-200)
```python
class SchedulerPolicyNetwork:
    """Controls diffusion scheduler parameters per prompt"""
    
    def __init__(self):
        # Maps text embeddings → scheduler parameters
        self.beta_predictor = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(scheduler.betas))  # Per-step beta values
        )
        
        self.guidance_net = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()  # Guidance scale [0,1] → scaled to [1,20]
        )
    
    def forward(self, text_embeddings):
        # Predict custom beta schedule
        beta_params = self.beta_predictor(text_embeddings)
        
        # Predict guidance scale
        guidance_scale = self.guidance_net(text_embeddings) * 19 + 1
        
        return {
            'beta_schedule': F.softplus(beta_params),  # Ensure positive
            'guidance_scale': guidance_scale
        }
```

#### **Policy Type 2: Diversity Policy** (Lines 250-400)
```python
class DiversityPolicyNetwork:
    """Modifies latent space for diversity without changing base model"""
    
    def __init__(self):
        self.latent_modifier = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # Match latent channels
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding=1),  # Output same dimensions
            nn.Tanh()  # Bounded modifications
        )
    
    def forward(self, latents, text_embeddings):
        # Apply learnable modifications to latent space
        modifications = self.latent_modifier(latents)
        
        # Scale modifications based on text context
        text_scale = torch.sigmoid(self.text_scaler(text_embeddings))
        scaled_mods = modifications * text_scale.unsqueeze(-1).unsqueeze(-1)
        
        return latents + 0.1 * scaled_mods  # Small, learnable perturbations
```

#### **Policy Type 3: LoRA Policy** (Lines 450-600)
```python
class LoRAPolicyNetwork:
    """Fine-tunes UNet layers using Low-Rank Adaptation"""
    
    def __init__(self, unet):
        self.base_unet = unet
        self.lora_layers = {}
        
        # Add LoRA adapters to attention layers
        for name, module in unet.named_modules():
            if isinstance(module, nn.Linear) and 'attn' in name:
                self.lora_layers[name] = LoRALayer(
                    module.in_features, 
                    module.out_features, 
                    rank=16  # Low-rank bottleneck
                )
    
    def forward(self, *args, **kwargs):
        # Forward pass with LoRA modifications
        return self.modified_unet(*args, **kwargs)
```

### 4. Reward System (`training/rewards.py`)

**Location**: `ppo_diffusion/training/rewards.py` (Lines 1-500+)  
**Purpose**: Multi-component reward calculation with pluggable metrics

```python
class DiffusionRewardFunction:
    """Comprehensive reward system for visual diversity"""
    
    def __init__(self, ref_features, feature_extractor, reward_metric="MMD"):
        self.ref_features = ref_features
        self.feature_extractor = feature_extractor
        self.reward_metric = get_reward_metric(reward_metric)
        
    def calculate_batch_rewards(self, generated_images, prompts):
        """Calculate individual rewards for each generated image"""
        
        # 1. Extract visual features
        gen_features = self.feature_extractor.extract_features(generated_images)
        
        # 2. Calculate primary diversity reward
        diversity_rewards = self.reward_metric.calculate_rewards(
            gen_features, self.ref_features
        )
        
        # 3. Add multi-component bonuses (if enabled)
        if USE_MULTI_COMPONENT_REWARD:
            # Sequential diversity: reward temporal variation
            sequential_bonus = self._calculate_sequential_reward(gen_features)
            
            # Spatial diversity: reward varied compositions
            spatial_bonus = self._calculate_spatial_reward(generated_images)
            
            # Entropy bonus: reward information content
            entropy_bonus = self._calculate_entropy_reward(generated_images)
            
            total_rewards = (
                diversity_rewards + 
                0.1 * sequential_bonus + 
                0.1 * spatial_bonus + 
                0.05 * entropy_bonus
            )
        else:
            total_rewards = diversity_rewards
            
        return total_rewards, {
            'diversity': diversity_rewards,
            'sequential': sequential_bonus if USE_MULTI_COMPONENT_REWARD else 0,
            'spatial': spatial_bonus if USE_MULTI_COMPONENT_REWARD else 0,
            'entropy': entropy_bonus if USE_MULTI_COMPONENT_REWARD else 0
        }
```

### 5. PPO Agent (`training/agent.py`)

**Location**: `ppo_diffusion/training/agent.py` (Lines 1-800+)  
**Purpose**: Complete PPO implementation adapted for diffusion models

```python
class DiffusionPPOAgent:
    """PPO Agent specialized for diffusion model fine-tuning"""
    
    def __init__(self, sampler, ref_features, training_mode="SCHEDULER_POLICY"):
        # Initialize networks based on training mode
        if training_mode == "SCHEDULER_POLICY":
            self.actor = SchedulerPolicyNetwork(sampler)
        elif training_mode == "DIVERSITY_POLICY":
            self.actor = DiversityPolicyNetwork()
        elif training_mode == "LORA_UNET":
            self.actor = LoRAPolicyNetwork(sampler.unet)
            
        self.critic = DiffusionValueNetwork(feature_dim=512)
        
        # PPO hyperparameters
        self.clip_ratio = DEFAULT_CLIP_RATIO  # 0.2
        self.vf_coef = DEFAULT_VF_COEF        # 0.5
        self.entropy_coef = DEFAULT_ENTROPY_COEF  # 0.01
        
    def ppo_update(self, batch_trajectories):
        """Standard PPO update with clipping"""
        
        for epoch in range(self.ppo_epochs):
            # Calculate advantages using GAE
            advantages = self.calculate_gae(batch_trajectories)
            
            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Combined loss
            total_loss = (
                policy_loss + 
                self.vf_coef * value_loss + 
                self.entropy_coef * entropy_loss
            )
            
            # Gradient update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
            self.optimizer.step()
```

### 6. Reward Metrics (`training/reward_metrics/`)

**Location**: `ppo_diffusion/training/reward_metrics/`  
**Purpose**: Pluggable reward metric system

#### **MMD (Maximum Mean Discrepancy)** - `mmd.py` (Lines 1-200)
```python
def calculate_individual_mmd_rewards(generated_features, reference_features, gamma=None):
    """
    Calculate how much each generated sample contributes to overall diversity
    using Maximum Mean Discrepancy with RBF kernel
    """
    
    # Convert to PyTorch tensors
    gen_tensor = torch.from_numpy(generated_features).float()
    ref_tensor = torch.from_numpy(reference_features).float()
    
    # Auto-select gamma if not provided
    if gamma is None:
        gamma = 1.0 / (2 * torch.median(torch.pdist(ref_tensor)) ** 2)
    
    def rbf_kernel(X, Y, gamma):
        """Radial Basis Function kernel"""
        XX = torch.sum(X**2, dim=1, keepdim=True)
        YY = torch.sum(Y**2, dim=1, keepdim=True)
        XY = torch.mm(X, Y.t())
        distances = XX + YY.t() - 2 * XY
        return torch.exp(-gamma * distances)
    
    # Calculate individual contributions
    individual_rewards = []
    
    for i in range(len(gen_tensor)):
        # MMD with this sample included
        gen_subset = gen_tensor
        mmd_with = compute_mmd(gen_subset, ref_tensor, gamma)
        
        # MMD without this sample  
        gen_without = torch.cat([gen_tensor[:i], gen_tensor[i+1:]])
        mmd_without = compute_mmd(gen_without, ref_tensor, gamma) if len(gen_without) > 0 else 0.0
        
        # Individual contribution (marginal utility)
        contribution = mmd_without - mmd_with  # Higher when sample increases diversity
        individual_rewards.append(contribution)
    
    return np.array(individual_rewards)
```

#### **Mutual Information** - `mi.py` (Lines 1-150)
```python
def calculate_individual_mi_rewards(generated_images, reference_images, gamma=1.0):
    """
    Calculate mutual information between generated and reference image sets
    using k-nearest neighbors estimation
    """
    
    # Convert images to feature vectors using CLIP or ResNet
    gen_features = extract_image_features(generated_images)
    ref_features = extract_image_features(reference_images) 
    
    # Estimate MI using k-NN entropy estimation
    def estimate_entropy(data, k=3):
        """Estimate entropy using k-nearest neighbor distances"""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
        distances, _ = nbrs.kneighbors(data)
        # Use k-th nearest neighbor distance (excluding self)
        knn_distances = distances[:, k]
        # Entropy estimation: H ≈ log(N) + log(2π) + (d/2) + <log(r_k)>
        return np.mean(np.log(knn_distances + 1e-10))
    
    # Calculate individual MI contributions
    individual_rewards = []
    
    for i in range(len(gen_features)):
        # MI with this sample
        combined_with = np.vstack([gen_features, ref_features])
        mi_with = estimate_entropy(combined_with) - estimate_entropy(gen_features) - estimate_entropy(ref_features)
        
        # MI without this sample
        gen_without = np.delete(gen_features, i, axis=0)
        if len(gen_without) > 0:
            combined_without = np.vstack([gen_without, ref_features])
            mi_without = estimate_entropy(combined_without) - estimate_entropy(gen_without) - estimate_entropy(ref_features)
        else:
            mi_without = 0.0
            
        # Individual contribution
        mi_contribution = mi_with - mi_without
        individual_rewards.append(mi_contribution)
    
    return np.array(individual_rewards)
```

#### **FID (Fréchet Inception Distance)** - `fid.py` (Lines 1-100)
```python
def calculate_fid_batch_rewards(generated_images, reference_images, reward_scale=0.1):
    """
    Calculate FID-based rewards using Inception features
    Lower FID = higher quality = higher reward
    """
    
    # Extract Inception features
    gen_features = extract_inception_features(generated_images)
    ref_features = extract_inception_features(reference_images)
    
    # Calculate individual FID contributions
    individual_rewards = []
    fid_scores = []
    
    for i in range(len(generated_images)):
        # FID for single generated image vs reference set
        single_gen = gen_features[i:i+1]
        
        # Calculate Fréchet distance
        mu_gen = np.mean(single_gen, axis=0)
        sigma_gen = np.cov(single_gen.T)
        
        mu_ref = np.mean(ref_features, axis=0)  
        sigma_ref = np.cov(ref_features.T)
        
        # FID calculation
        diff = mu_gen - mu_ref
        covmean = sqrtm(sigma_gen @ sigma_ref)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = np.sum(diff**2) + np.trace(sigma_gen + sigma_ref - 2*covmean)
        fid_scores.append(fid)
        
        # Convert to reward (lower FID = higher reward)
        reward = reward_scale * np.exp(-fid / 100.0)  # Exponential decay
        individual_rewards.append(reward)
    
    return np.array(individual_rewards), np.mean(individual_rewards), np.array(fid_scores)
```

### 7. Feature Extraction System (`core/features.py`)

**Location**: `ppo_diffusion/core/features.py` (Lines 1-200)  
**Purpose**: Visual feature extraction using ResNet-18

```python
class FeatureExtractor:
    """ResNet-18 based visual feature extractor for diversity metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load pre-trained ResNet-18
        self.model = models.resnet18(pretrained=True)
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(device)
        self.model.eval()
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, images, preserve_gradients=False):
        """Extract 512D feature vectors from images"""
        
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        # Normalize for ResNet
        images = self.transform(images)
        
        if preserve_gradients:
            # Keep gradients for policy training
            features = self.model(images)
        else:
            # Detach for reward calculation
            with torch.no_grad():
                features = self.model(images)
        
        return features.squeeze().cpu().numpy()
```

### 8. Value Network (`models/value.py`)

**Location**: `ppo_diffusion/models/value.py` (Lines 1-50)  
**Purpose**: Critic network for PPO value function estimation

```python
class DiffusionValueNetwork(nn.Module):
    """
    Value Function for Diffusion Models:
        - Estimates how good a prompt/context is for generating diverse images
        - Takes text embeddings as input (like state in vanilla PPO)
    Input:
        - Text embedding or features
    Output:
        - Scalar value V(prompt): expected diversity reward for this prompt
    """
    def __init__(self, feature_dim: int = 512):
        super(DiffusionValueNetwork, self).__init__()
        
        # Simplified network to prevent overfitting
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate value from prompt features"""
        return self.network(features)
```

---

## 🚀 Training Pipeline

### Complete Training Flow

```python
def train_ppo_diffusion():
    """Complete training pipeline"""
    
    # === PHASE 1: Setup ===
    # Load reference data (73 Mars crater images → features)
    ref_features = np.load('reference_features/reference_crater_features_v3.npz')['features']
    ref_images = np.load('reference_features/reference_crater_images.npz')['images']
    
    # Initialize diffusion pipeline  
    sampler = DiffusionSampler("runwayml/stable-diffusion-v1-5")
    
    # Create PPO agent with configurable mode
    agent = DiffusionPPOAgent(
        sampler=sampler,
        ref_features=ref_features,
        training_mode=DEFAULT_TRAINING_MODE,  # From constants.py
        reward_metric=DEFAULT_REWARD_METRIC
    )
    
    # === PHASE 2: Training Loop ===
    for episode in range(DEFAULT_NUM_EPISODES):
        
        # 2a. Generate batch of images
        batch_trajectories = []
        batch_rewards = []
        
        for prompt in training_prompts:
            # Sample image with trajectory recording
            image, trajectory = agent.generate_with_trajectory(prompt)
            
            # Calculate diversity reward
            reward = agent.reward_function.calculate_reward(image, ref_features)
            
            batch_trajectories.append(trajectory)  
            batch_rewards.append(reward)
        
        # 2b. PPO Update
        agent.ppo_update(batch_trajectories, batch_rewards)
        
        # 2c. Logging & Checkpointing
        if episode % 100 == 0:
            agent.save_checkpoint(f"episode_{episode}")
            log_training_metrics(episode, batch_rewards, agent.losses)
            
    # === PHASE 3: Evaluation ===
    evaluate_final_model(agent, test_prompts)
```

### Key Training Components

1. **Batch Generation**: Multiple prompts processed in parallel
2. **Trajectory Recording**: Complete diffusion process captured
3. **Reward Calculation**: Individual diversity scores per image
4. **PPO Updates**: Actor-critic networks updated with clipping
5. **Memory Management**: GPU cache clearing and optimization
6. **Checkpointing**: Model states saved regularly
7. **Logging**: Comprehensive CSV logs with all metrics

---

## ⚙️ Configuration System

### Central Configuration (`utils/constants.py`)

**Location**: `ppo_diffusion/utils/constants.py` (Lines 1-100)

```python
# === Training Mode Selection ===
DEFAULT_TRAINING_MODE = "SCHEDULER_POLICY"  # SCHEDULER_POLICY, DIVERSITY_POLICY, LORA_UNET

# === Reward Metric Selection ===  
DEFAULT_REWARD_METRIC = "MMD"  # MMD, MI, FID, MMD_MI, LPIPS

# === PPO Hyperparameters ===
DEFAULT_CLIP_RATIO = 0.2          # PPO clipping parameter
DEFAULT_VF_COEF = 0.5             # Value function loss coefficient  
DEFAULT_ENTROPY_COEF = 0.01       # Entropy bonus coefficient
DEFAULT_LR_ACTOR = 3e-4           # Actor learning rate
DEFAULT_LR_CRITIC = 1e-3          # Critic learning rate

# === Training Parameters ===
DEFAULT_NUM_EPISODES = 10000      # Total training episodes
DEFAULT_BATCH_SIZE = 1            # Images per batch (memory limited)
DEFAULT_PPO_EPOCHS = 4            # PPO update epochs per batch
DEFAULT_GAE_LAMBDA = 0.95         # GAE discount parameter

# === Reward Composition (Multi-component rewards) ===
USE_MULTI_COMPONENT_REWARD = True
SEQUENTIAL_REWARD_WEIGHT = 0.1    # Temporal diversity bonus
SPATIAL_REWARD_WEIGHT = 0.1       # Spatial composition bonus  
ENTROPY_REWARD_WEIGHT = 0.05      # Information content bonus

# === MMD_MI Combined Rewards ===
MMD_WEIGHT = 0.7                  # MMD contribution in MMD_MI metric
MI_WEIGHT = 0.3                   # MI contribution in MMD_MI metric

# === Model Parameters ===
DEFAULT_NUM_INFERENCE_STEPS = 20  # Diffusion sampling steps
DEFAULT_IMAGES_PER_PROMPT = 1     # Batch size per prompt
DEFAULT_GUIDANCE_SCALE = 7.5      # Default classifier-free guidance

# === Hardware Optimization ===
ENABLE_GRADIENT_CHECKPOINTING = True
ENABLE_ATTENTION_SLICING = True
CLEAR_CACHE_EVERY_N_EPISODES = 10
```

### Runtime Configuration

The system supports dynamic configuration through:
- **Environment variables**: Override constants at runtime
- **Command line arguments**: Training mode and metric selection
- **Config files**: JSON configuration for complex setups
- **Interactive mode**: Real-time parameter adjustment

---

## 📊 Data Flow Architecture

### 1. Reference Data Pipeline

```
Mars Crater Images (73 samples)
    ↓ [build_reference_features.py]
ResNet-18 Feature Extraction (512D vectors)
    ↓ [create_reference_images_npz.py]  
NPZ Storage (efficient loading)
    ↓ [train.py]
Agent Initialization (reward calculation ready)
```

### 2. Training Data Flow

```
Training Prompts (25 diverse descriptions)
    ↓ [DiffusionSampler]
Generated Images + Trajectories
    ↓ [FeatureExtractor] 
Visual Features (512D)
    ↓ [RewardFunction]
Individual Diversity Scores
    ↓ [PPOAgent]
Policy Updates (gradient ascent)
```

### 3. Logging & Monitoring (`utils/logging.py`)

**Location**: `ppo_diffusion/utils/logging.py` (Lines 1-150)

```python
# CSV logging structure
log_columns = [
    'episode',           # Training episode number
    'prompt_idx',        # Which prompt was used
    'reward_total',      # Total reward for this image
    'reward_diversity',  # Primary diversity component
    'reward_sequential', # Temporal bonus
    'reward_spatial',    # Spatial bonus  
    'reward_entropy',    # Information bonus
    'actor_loss',        # Policy loss
    'critic_loss',       # Value function loss
    'entropy_loss',      # Entropy regularization
    'kl_divergence',     # Policy change magnitude
    'grad_norm_actor',   # Gradient norms (debugging)
    'grad_norm_critic',
    'memory_allocated',  # GPU memory usage
    'timestamp'          # Training time
]

def log_log_probability(log_prob_tensor, episode, prompt_idx, step, csv_writer):
    """Log policy probability for debugging gradient flow"""
    csv_writer.writerow([
        episode, prompt_idx, step,
        log_prob_tensor.item() if hasattr(log_prob_tensor, 'item') else log_prob_tensor,
        log_prob_tensor.requires_grad,
        time.time()
    ])
```

---

---

## 🖥️ Command Line Usage & Workflow

### Running the PPO-Diffusion Pipeline

The main training pipeline is executed as a Python module from the project root:

```bash
# Navigate to project root
cd /Users/bryce2hua/Desktop/RL

# Run PPO training pipeline
python3 -m ppo_diffusion.train
```

**Command Breakdown**:
- `python3 -m`: Runs the module as a script
- `ppo_diffusion.train`: Points to `ppo_diffusion/train.py` as the main entry point
- Uses the `__main__.py` or direct module execution

### Training Configuration

The training run will automatically:
1. **Load configuration** from `ppo_diffusion/utils/constants.py`
2. **Initialize the selected policy** (SCHEDULER_POLICY, DIVERSITY_POLICY, or LORA_UNET)
3. **Load reference data** from `reference_features/` directory
4. **Start training loop** with comprehensive logging
5. **Save checkpoints** to `outputs/models/`
6. **Log metrics** to `outputs/logs/crater_YYYYMMDDHHMMSS/`

### Real-time Monitoring

During training, you'll see output like:
```
🎯 Training 45,234 parameters in SCHEDULER_POLICY mode (PPO)
🎯 PPO settings: clip_ratio=0.2, lr_actor=3e-4, lr_critic=1e-3
🔍 Using MMD reward metric (configurable in constants.py)
Episode 1/10000: reward=0.0045, actor_loss=0.234, critic_loss=0.823
Episode 10/10000: reward=0.0041, actor_loss=0.189, critic_loss=0.765
```

### Syncing Training Results from Remote

**IMPORTANT**: Before plotting results, always sync the latest logs from Bridges2:

```bash
# Sync training logs and results from remote server
./jobs/sync_from_bridges2.sh
```

This script will:
- Download latest training logs from `outputs/logs/`
- Sync generated images from `outputs/images/`
- Update model checkpoints from `outputs/models/`
- Fetch any new plots from `outputs/plots/`

### Plotting Training Results

After syncing, visualize training progress using the plotting script:

```bash
# Plot training metrics from current logs
python3 ppo_diffusion/tests/plot_test_simple.py
```

**What the plotting script does**:
1. **Automatically detects** the most recent training log directory
2. **Reads CSV logs** from `outputs/logs/crater_YYYYMMDDHHMMSS/`
3. **Generates plots** for:
   - Reward progression over episodes
   - Actor and critic loss curves
   - Gradient norm tracking
   - Policy entropy decay
   - Memory usage patterns

**Output locations**:
- Plots saved to `outputs/plots/`
- Interactive plots displayed if running locally
- Summary statistics printed to console

### Complete Workflow Example

```bash
# 1. Sync latest results from remote training
./jobs/sync_from_bridges2.sh

# 2. Check what training sessions are available
ls outputs/logs/

# 3. Plot the latest training results
python3 ppo_diffusion/tests/plot_test_simple.py

# 4. (Optional) Start new training run locally
python3 -m ppo_diffusion.train

# 5. (Optional) Sync new results back to remote
./jobs/sync_to_bridges2.sh
```

### Training Session Management

**Log Directory Structure**:
```
outputs/logs/
├── crater_20241126_143022/    # Scheduler policy run
│   ├── training_log.csv
│   ├── gradient_log.csv
│   └── reward_components.csv
├── crater_20241126_150055/    # Diversity policy run
│   ├── training_log.csv
│   └── policy_updates.csv
└── crater_20241126_183015/    # LoRA policy run
    ├── training_log.csv
    └── memory_usage.csv
```

**Key Log Files**:
- `training_log.csv`: Main metrics (rewards, losses, gradients)
- `gradient_log.csv`: Detailed gradient flow tracking
- `reward_components.csv`: Breakdown of multi-component rewards
- `policy_updates.csv`: PPO update statistics
- `memory_usage.csv`: GPU memory tracking

### Debugging Failed Training Runs

If training fails or produces unexpected results:

```bash
# 1. Check the latest log directory
ls -la outputs/logs/ | tail -5

# 2. Examine the training log for issues
tail -20 outputs/logs/crater_YYYYMMDDHHMMSS/training_log.csv

# 3. Look for gradient flow problems
grep "0.0000" outputs/logs/crater_YYYYMMDDHHMMSS/gradient_log.csv

# 4. Check memory issues
tail -10 outputs/logs/crater_YYYYMMDDHHMMSS/memory_usage.csv

# 5. Plot partial results to diagnose issues
python3 ppo_diffusion/tests/plot_test_simple.py
```

### Configuration Overrides

You can modify training behavior by editing `ppo_diffusion/utils/constants.py`:

```python
# Change training mode
DEFAULT_TRAINING_MODE = "DIVERSITY_POLICY"  # or "LORA_UNET"

# Switch reward metric  
DEFAULT_REWARD_METRIC = "MMD_MI"  # or "FID", "MI"

# Adjust training parameters
DEFAULT_NUM_EPISODES = 5000  # Reduce for testing
DEFAULT_BATCH_SIZE = 2       # Increase if memory allows
```

### Performance Monitoring

**Real-time monitoring during training**:
- **Episode progress**: Shows current episode out of total
- **Reward trends**: Immediate feedback on policy performance  
- **Loss values**: Actor and critic training progress
- **Gradient norms**: Early warning for gradient issues
- **Memory usage**: GPU utilization tracking
- **Timing**: Episodes per second performance

**Post-training analysis with plots**:
- **Convergence analysis**: When did training plateau?
- **Policy comparison**: Different training modes side-by-side
- **Reward decomposition**: Which components drive learning?
- **Stability metrics**: Gradient and loss variance over time

---

## 🔬 Key Algorithms

### 1. Generalized Advantage Estimation (GAE)

**Location**: `ppo_diffusion/training/agent.py` (Lines 400-450)

```python
def calculate_gae(self, trajectories, gamma=0.99, lambda_=0.95):
    """
    Calculate advantages using GAE for variance reduction
    GAE(γ,λ) = Σ (γλ)^l δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    
    advantages = []
    
    for trajectory in trajectories:
        # Calculate TD errors
        values = self.critic(trajectory.states)
        next_values = torch.cat([values[1:], torch.zeros(1)])
        
        deltas = trajectory.rewards + gamma * next_values - values
        
        # Calculate GAE
        gae = 0
        traj_advantages = []
        
        for delta in reversed(deltas):
            gae = delta + gamma * lambda_ * gae
            traj_advantages.insert(0, gae)
            
        advantages.extend(traj_advantages)
        
    return torch.tensor(advantages)
```

### 2. PPO Clipping Mechanism

**Location**: `ppo_diffusion/training/agent.py` (Lines 500-550)

```python
def ppo_policy_loss(self, old_log_probs, new_log_probs, advantages):
    """
    PPO clipped objective: L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    """
    
    # Probability ratios
    ratios = torch.exp(new_log_probs - old_log_probs)
    
    # Unclipped objective
    surr1 = ratios * advantages
    
    # Clipped objective  
    surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
    
    # Take minimum (conservative update)
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss
```

### 3. Multi-Component Reward Fusion

**Location**: `ppo_diffusion/training/rewards.py` (Lines 300-400)

```python
def calculate_multi_component_reward(self, images, features, trajectories):
    """
    Combine multiple reward signals for richer training signal
    """
    
    # Primary diversity reward (configurable metric)
    diversity_rewards = self.primary_metric.calculate_rewards(features, self.ref_features)
    
    # Sequential diversity: reward temporal variation in batch
    sequential_rewards = self._sequential_diversity(features)
    
    # Spatial diversity: reward varied image compositions  
    spatial_rewards = self._spatial_diversity(images)
    
    # Entropy reward: reward information content
    entropy_rewards = self._entropy_reward(images)
    
    # Quality reward: basic aesthetic/technical quality
    quality_rewards = self._quality_reward(images)
    
    # Weighted combination
    total_rewards = (
        1.0 * diversity_rewards +           # Primary objective
        0.1 * sequential_rewards +          # Temporal bonus
        0.1 * spatial_rewards +             # Spatial bonus
        0.05 * entropy_rewards +            # Information bonus
        0.05 * quality_rewards              # Quality bonus
    )
    
    return total_rewards, {
        'diversity': diversity_rewards,
        'sequential': sequential_rewards, 
        'spatial': spatial_rewards,
        'entropy': entropy_rewards,
        'quality': quality_rewards
    }
```

---

## 🔧 Extensibility & Customization

### 1. Adding New Reward Metrics

```python
# 1. Create new metric class in training/reward_metrics/
class CustomRewardMetric(RewardMetric):
    def __init__(self, **kwargs):
        super().__init__("CUSTOM")
        # Initialize custom parameters
        
    def calculate_rewards(self, generated_features, reference_features, **kwargs):
        # Implement custom reward logic
        return individual_rewards

# 2. Register in __init__.py
REWARD_METRICS["CUSTOM"] = CustomRewardMetric

# 3. Use in configuration
DEFAULT_REWARD_METRIC = "CUSTOM"
```

### 2. Adding New Training Modes

```python
# 1. Create new policy network in models/policy.py
class CustomPolicyNetwork(nn.Module):
    def __init__(self, sampler):
        # Define custom architecture
        
    def forward(self, inputs):
        # Define custom forward pass
        return actions, log_probs

# 2. Add to agent initialization
if training_mode == "CUSTOM_MODE":
    self.actor = CustomPolicyNetwork(sampler)

# 3. Update constants.py
DEFAULT_TRAINING_MODE = "CUSTOM_MODE"
```

### 3. Custom Feature Extractors

```python
# Replace or extend the ResNet-18 feature extractor
class CustomFeatureExtractor:
    def __init__(self):
        # Load custom model (CLIP, DINOv2, etc.)
        self.model = load_custom_model()
        
    def extract_features(self, images):
        # Custom feature extraction logic
        return features

# Use in agent initialization
agent = DiffusionPPOAgent(
    feature_extractor=CustomFeatureExtractor(),
    # ... other parameters
)
```

---

## 📈 Performance Monitoring

### 1. Training Metrics Dashboard

**Location**: `ppo_diffusion/utils/logging.py` (Lines 50-100)

The system logs comprehensive metrics for monitoring:

```python
# Real-time metrics (logged every episode)
training_metrics = {
    'rewards': {
        'mean': np.mean(batch_rewards),
        'std': np.std(batch_rewards), 
        'min': np.min(batch_rewards),
        'max': np.max(batch_rewards)
    },
    'losses': {
        'actor': policy_loss.item(),
        'critic': value_loss.item(),
        'entropy': entropy_loss.item()
    },
    'gradients': {
        'actor_norm': actor_grad_norm,
        'critic_norm': critic_grad_norm,
        'clipped': grad_clipped_ratio
    },
    'memory': {
        'gpu_allocated': torch.cuda.memory_allocated(),
        'gpu_reserved': torch.cuda.memory_reserved()
    }
}
```

### 2. Visualization System (`utils/visualization.py`)

**Location**: `ppo_diffusion/utils/visualization.py` (Lines 1-300)

```python
def plot_training_progress(csv_log_path):
    """Generate comprehensive training plots"""
    
    # Reward progression over time
    plot_reward_curves(rewards_over_time)
    
    # Loss curves (actor, critic, entropy)
    plot_loss_curves(losses_over_time)
    
    # Feature space evolution (t-SNE)
    plot_feature_evolution(generated_features, reference_features)
    
    # Sample quality over time
    plot_sample_grid(saved_images, episode_numbers)
    
    # Gradient norms and clipping statistics
    plot_gradient_stats(gradient_norms, clipping_ratios)
```

---

## 🚨 Policy Training Issues & Current Status

### Critical Challenges Encountered

Throughout the development and testing of the PPO-diffusion system, significant issues have emerged with each policy type. While some fundamental problems have been resolved, new challenges have surfaced.

### 1. Scheduler Policy Issues (`SCHEDULER_POLICY`)

**Previous Problem**: ~~Zero Gradient Flow~~ **RESOLVED** ✅

**Current Problem**: **Reward Flattening**

#### **Previous Symptom (SOLVED)**
```csv
# OLD gradient logs showing zero gradients:
update,episode,actor_grad_before,actor_grad_after,critic_grad,grad_clipped,timestamp
1,1,0.0000,0.0000,0.8234,False,20241126_143022  # ❌ FIXED
```

#### **Current Symptom**
```csv
# NEW gradient logs showing gradients working but rewards flattening:
update,episode,actor_grad_before,actor_grad_after,critic_grad,reward_mean,timestamp
1,1,0.2341,0.1892,0.8234,0.0045,20241126_150022  # ✅ Gradients flowing
2,2,0.1987,0.1654,0.7891,0.0043,20241126_150023  # ✅ Gradients flowing
10,10,0.1234,0.1023,0.6543,0.0041,20241126_150032 # ❌ Rewards plateauing
50,50,0.0892,0.0734,0.5234,0.0041,20241126_151022 # ❌ Rewards flat
100,100,0.0756,0.0623,0.4987,0.0041,20241126_152022 # ❌ No improvement
```

#### **Current Status**: **GRADIENT FLOW FIXED, REWARD PLATEAU** ⚠️

### 2. Diversity Policy Issues (`DIVERSITY_POLICY`)

**Previous Problem**: ~~Latent Space Modification Conflicts~~ **PARTIALLY RESOLVED** ✅

**Current Problem**: **Reward Flattening**

#### **Current Symptom**
```csv
# Diversity policy reward progression:
episode,reward_mean,reward_std,grad_norm,status
1,0.008,0.003,0.234,LEARNING
10,0.012,0.004,0.189,LEARNING
25,0.016,0.002,0.156,LEARNING
50,0.018,0.001,0.023,PLATEAUING
100,0.018,0.001,0.019,FLAT
200,0.018,0.001,0.018,STUCK
```

#### **Current Status**: **FUNCTIONAL BUT PLATEAUS QUICKLY** ⚠️

### 3. LoRA Policy Issues (`LORA_UNET`)

**Previous Problem**: ~~Memory and Computational Overhead~~ **MANAGED** ✅

**Current Problems**: **Extreme Slowness + Reward Flattening**

#### **Current Symptoms**
```python
# Training speed comparison:
SCHEDULER_POLICY: ~30 seconds/episode
DIVERSITY_POLICY: ~45 seconds/episode  
LORA_UNET: ~180 seconds/episode  # 4-6x slower!

# Reward progression:
episode,reward_mean,time_per_episode,status
1,0.012,185s,SLOW_BUT_LEARNING
10,0.019,178s,SLOW_BUT_LEARNING
25,0.024,182s,SLOW_BUT_LEARNING
50,0.025,189s,SLOW_PLATEAU
100,0.025,191s,SLOW_FLAT
```

#### **Current Status**: **WORKING BUT EXTREMELY SLOW + QUICK PLATEAU** ❌

### 4. Cross-Policy Issues - Updated Analysis

#### **Common Problems Across All Policies**

1. **Reward Plateau Phenomenon** (NEW PRIMARY ISSUE):
```python
# All policies show same pattern:
# - Initial learning (episodes 1-25)
# - Gradual improvement (episodes 25-50)  
# - Rapid plateau (episodes 50+)
# - No further improvement despite continued training

scheduler_rewards = [0.004, 0.005, 0.006, 0.004, 0.004, ...]  # Quick plateau
diversity_rewards = [0.008, 0.012, 0.016, 0.018, 0.018, ...]  # Quick plateau  
lora_rewards = [0.012, 0.019, 0.024, 0.025, 0.025, ...]       # Quick plateau
```

2. **Limited Diversity Metric Range**:
```python
# MMD, MI, and FID all show limited dynamic range
# Policies optimize what they can measure, but hit ceiling quickly
mmd_range = [0.001, 0.025]      # Small effective range
mi_range = [0.005, 0.030]       # Limited improvement possible
fid_range = [25.0, 15.0]        # Modest improvements only
```

3. **Policy Exploration Limitations**:
```python
# All policies converge to local optima within ~50 episodes
# Entropy decay is too rapid across all policy types
entropy_progression = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, ...]  # Too fast decay
```

### 5. Root Cause Hypotheses

#### **Fundamental Issues**

1. **Diversity Metric Limitations**:
   - MMD, MI, FID have limited dynamic range for generated images
   - Policies quickly exhaust measurable improvements
   - Need richer diversity metrics or different formulations

2. **Local Optima Trapping**:
   - All policies converge to suboptimal solutions too quickly
   - PPO's conservative updates may be insufficient for exploration
   - Need stronger exploration mechanisms

3. **Feature Space Constraints**:
   - ResNet-18 features may not capture full diversity spectrum
   - Policies optimize limited aspects of visual diversity
   - May need higher-dimensional or multi-modal feature representations

4. **Reward Engineering Problems**:
   - Individual image rewards may be fundamentally flawed approach
   - Batch-level or sequence-level rewards might work better
   - Current formulation encourages quick local convergence

### 6. Lessons Learned

#### **Key Insights from Latest Debugging**

1. **Gradient Flow Success**: Fixed trajectory detachment - this was critical
2. **Reward Formulation is Key**: Current rewards plateau too quickly across all policies
3. **Exploration vs Exploitation**: PPO may be too conservative for this problem
4. **Speed vs Quality Tradeoff**: LoRA offers quality but is prohibitively slow
5. **Metric Limitations**: Current diversity metrics hit ceiling too fast

#### **Successful Fixes**

1. ✅ **Gradient Preservation**: Proper trajectory recording maintains learning signals
2. ✅ **Architecture Stability**: Policies train without catastrophic failures
3. ✅ **Memory Management**: LoRA memory issues controlled
4. ✅ **Reward Signal Quality**: Rewards are meaningful, just limited in range

#### **Remaining Challenges**

1. ❌ **Reward Plateau**: All policies hit ceiling within 50 episodes
2. ❌ **Training Speed**: LoRA is 4-6x slower than other approaches
3. ❌ **Exploration**: Insufficient exploration leads to local optima
4. ❌ **Metric Range**: Diversity metrics have limited dynamic range

#### **Next Research Directions**

1. **TRPO Implementation**: May provide better exploration vs current PPO
2. **Batch-Level Rewards**: Reward entire batches rather than individual images
3. **Multi-Modal Metrics**: Combine visual + semantic + perceptual diversity measures
4. **Curriculum Learning**: Progressive difficulty in diversity requirements
5. **Ensemble Approaches**: Multiple policies or reward combinations

---

## 🎯 Summary

The PPO-diffusion codebase implements a sophisticated reinforcement learning system for fine-tuning diffusion models. While significant progress has been made in solving gradient flow issues, the current challenge is reward plateau across all policy types within 40-50 episodes.

### **Technical Achievements**

1. ✅ **Multi-Modal Policy Control**: Three distinct training modes working
2. ✅ **Gradient Flow Resolution**: Fixed trajectory detachment issues
3. ✅ **Pluggable Reward System**: MMD, MI, FID, and combination metrics
4. ✅ **Complete PPO Implementation**: Full actor-critic architecture
5. ✅ **Production Infrastructure**: Comprehensive logging and monitoring

### **Current Challenges**

1. ❌ **Reward Saturation**: All policies plateau quickly despite proper training
2. ❌ **LoRA Speed**: 4-6x slower than other policy types
3. ❌ **Limited Exploration**: PPO may be too conservative for this domain
4. ❌ **Metric Constraints**: Current diversity measures have limited range

### **Research Contributions**

- **RL for Diffusion**: Novel application of PPO to diffusion model fine-tuning
- **Diversity Rewards**: Systematic comparison of visual diversity metrics
- **Multi-Modal Training**: Different aspects of diffusion control
- **Debugging Insights**: Comprehensive analysis of policy training challenges

This research demonstrates the complexities of applying reinforcement learning to generative models and provides a foundation for future work with TRPO and alternative approaches.

### **File Count Summary**

The complete `ppo_diffusion/` codebase contains:
- **Core Python files**: 15+ implementation files
- **Utility modules**: 5 helper modules
- **Test files**: 10+ validation scripts
- **Data files**: 73 reference images + processed features
- **Configuration**: Centralized constants and settings
- **Documentation**: Comprehensive logging and visualization

Each component is designed to work together in a cohesive pipeline from data preparation through training to evaluation and monitoring
