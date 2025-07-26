# TRPO-Diffusion Codebase: Complete Technical Walkthrough

This document provides a comprehensive analysis of the `trpo_diffusion/` codebase, explaining the architecture, components, and training pipeline for fine-tuning Stable Diffusion models using Trust Region Policy Optimization (TRPO) to maximize visual diversity in generated Mars crater images.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [TRPO Algorithm Implementation](#trpo-algorithm-implementation)
5. [Command Line Usage](#command-line-usage)
6. [Configuration System](#configuration-system)
7. [Key Differences from PPO](#key-differences-from-ppo)
8. [Current Status & Future Work](#current-status--future-work)

---

## 🏗️ Architecture Overview

The TRPO-diffusion system implements a **Trust Region Policy Optimization approach** to fine-tune diffusion models for generating diverse visual content. The key innovation is using **natural gradients with trust region constraints** instead of clipped policy updates.

### Core Philosophy

```
Reference Images → Feature Extraction → Diversity Rewards → TRPO Training → Fine-tuned Model
```

**TRPO Key Differences from PPO**:
- **Natural Gradients**: Uses Fisher Information Matrix for second-order optimization
- **Trust Region Constraints**: Explicit KL divergence limits instead of clipping
- **Conjugate Gradients**: Solves linear systems for optimal policy updates
- **Line Search**: Adaptive step sizes with backtracking for constraint satisfaction

---

## 📁 Directory Structure

```
trpo_diffusion/
├── Entry Points
│   ├── train.py                    # Main TRPO training script
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
│       └── value.py                # Critic network for TRPO
│
├── Training System
│   └── training/
│       ├── agent.py                # Main TRPO agent implementation
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
│   │   ├── constants.py            # TRPO-specific hyperparameters
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

**Location**: `trpo_diffusion/train.py`  
**Purpose**: Main training orchestrator using TRPO instead of PPO

```python
# Key responsibilities:
def main():
    # 1. Load reference data (images + features)
    ref_features, ref_images = load_reference_data()
    
    # 2. Initialize diffusion pipeline
    sampler = DiffusionSampler(model_id="runwayml/stable-diffusion-v1-5")
    
    # 3. Create TRPO agent with configurable training mode
    agent = DiffusionTRPOAgent(  # 🔄 TRPO Agent instead of PPO
        sampler=sampler,
        ref_features=ref_features,
        training_mode=DEFAULT_TRAINING_MODE,  # SCHEDULER_POLICY, DIVERSITY_POLICY, LORA_UNET
        reward_metric=DEFAULT_REWARD_METRIC   # MMD, MI, FID, MMD_MI
    )
    
    # 4. Run training loop with trust region updates
    train_agent(agent, num_episodes=DEFAULT_NUM_EPISODES)
```

### 2. TRPO Agent (`training/agent.py`)

**Location**: `trpo_diffusion/training/agent.py` (Lines 30-800+)  
**Purpose**: Complete TRPO implementation adapted for diffusion models

```python
class DiffusionTRPOAgent:
    """TRPO Agent specialized for diffusion model fine-tuning"""
    
    def __init__(self, sampler, ref_features, training_mode="SCHEDULER_POLICY"):
        # Initialize networks based on training mode
        if training_mode == "SCHEDULER_POLICY":
            self.actor = SchedulerPolicyNetwork(sampler)
        elif training_mode == "DIVERSITY_POLICY":
            self.actor = DiversityPolicyNetwork()
        elif training_mode == "LORA_UNET":
            self.actor = LoRAPolicyNetwork(sampler.unet)
            
        # Copy of actor for computing KL divergence
        self.old_actor = copy.deepcopy(self.actor)
        self.critic = DiffusionValueNetwork(feature_dim=512)
        
        # TRPO-specific hyperparameters
        self.kl_target = DEFAULT_KL_TARGET        # 0.01
        self.damping = DEFAULT_DAMPING            # 0.1
        self.cg_iters = DEFAULT_CG_ITERS          # 10
        self.backtrack_iters = DEFAULT_BACKTRACK_ITERS  # 10
        self.backtrack_coeff = DEFAULT_BACKTRACK_COEFF  # 0.8
        self.accept_ratio = DEFAULT_ACCEPT_RATIO  # 0.1
        
    def trpo_update(self, batch_trajectories):
        """TRPO update with trust region constraints"""
        
        # 1. Compute policy gradient
        policy_gradient = self._compute_policy_gradient(
            batch_trajectories.log_probs, 
            batch_trajectories.advantages
        )
        
        # 2. Solve natural gradient using conjugate gradients
        natural_gradient = self._conjugate_gradient(
            policy_gradient, 
            batch_trajectories
        )
        
        # 3. Compute step size using line search
        step_size = self._line_search(
            natural_gradient, 
            policy_gradient, 
            batch_trajectories
        )
        
        # 4. Apply parameter update
        self._apply_update(natural_gradient, step_size)
        
        # 5. Update old policy for next iteration
        self._update_old_policy()
```

### 3. Policy Networks (`models/policy.py`)

**Location**: `trpo_diffusion/models/policy.py`  
**Purpose**: Same three policy architectures as PPO, adapted for TRPO

The policy networks are identical to PPO implementation but are used differently:

#### **Scheduler Policy Network**
```python
class SchedulerPolicyNetwork:
    """Controls diffusion scheduler parameters per prompt"""
    
    def get_trainable_parameters(self):
        """Return parameters for TRPO optimization"""
        if self.training_mode == "SCHEDULER_POLICY":
            return [p for p in self.parameters() if p.requires_grad]
        # Different parameter sets for different modes
    
    def compute_log_prob(self, text_embeddings, actions):
        """Compute log probability for TRPO gradient calculation"""
        # Gaussian distribution over scheduler parameters
        beta_mean, beta_log_std = self.forward(text_embeddings)
        dist = Normal(beta_mean, torch.exp(beta_log_std))
        return dist.log_prob(actions).sum()
```

---

## 🔬 TRPO Algorithm Implementation

### 1. Trust Region Constraint

**Location**: `trpo_diffusion/training/agent.py` (Lines 650-720)

```python
def _compute_kl_divergence(self, trajectories):
    """
    Compute KL divergence between old and new policies
    KL(π_old || π_new) for trust region constraint
    """
    
    if self.training_mode == "SCHEDULER_POLICY":
        return self._compute_scheduler_kl_divergence_between_policies(trajectories)
    elif self.training_mode == "DIVERSITY_POLICY":
        return self._compute_diversity_kl_divergence_between_policies(trajectories)
    elif self.training_mode == "LORA_UNET":
        return self._compute_lora_kl_divergence_between_policies(trajectories)

def _compute_scheduler_kl_divergence_between_policies(self, prompt):
    """
    KL divergence for scheduler policy (Gaussian distributions)
    KL(N(μ₁,Σ₁) || N(μ₂,Σ₂)) = log(Σ₂/Σ₁) + (Σ₁ + (μ₁-μ₂)²)/(2Σ₂) - 1/2
    """
    
    # Get old policy parameters
    old_beta_mean, old_beta_log_std = self.old_actor(prompt)
    old_beta_std = torch.exp(old_beta_log_std)
    
    # Get current policy parameters  
    curr_beta_mean, curr_beta_log_std = self.actor(prompt)
    curr_beta_std = torch.exp(curr_beta_log_std)
    
    # KL divergence between two multivariate Gaussians
    beta_var_ratio = (curr_beta_std ** 2) / (old_beta_std ** 2 + 1e-8)
    beta_mean_diff = curr_beta_mean - old_beta_mean.detach()
    beta_mean_term = (beta_mean_diff ** 2) / (old_beta_std ** 2 + 1e-8)
    beta_log_det_term = 2 * (old_beta_log_std.detach() - curr_beta_log_std)
    beta_kl = 0.5 * torch.sum(beta_var_ratio + beta_mean_term + beta_log_det_term - 1)
    
    return beta_kl
```

### 2. Fisher Information Matrix & Conjugate Gradients

**Location**: `trpo_diffusion/training/agent.py` (Lines 500-600)

```python
def _conjugate_gradient(self, gradient, trajectories, residual_tol=1e-10):
    """
    Solve Fisher Information Matrix equation: F * x = g
    using Conjugate Gradients where F is Fisher matrix, g is policy gradient
    """
    
    x = torch.zeros_like(gradient)
    r = gradient.clone()  # Initial residual
    p = r.clone()         # Initial search direction
    rsold = torch.dot(r, r)
    
    for i in range(self.cg_iters):
        # Compute Fisher-vector product: F * p
        Ap = self._fisher_vector_product(p, trajectories)
        
        # CG step size
        alpha = rsold / (torch.dot(p, Ap) + 1e-8)
        
        # Update solution
        x = x + alpha * p
        
        # Update residual
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        
        # Check convergence
        if rsnew < residual_tol:
            break
            
        # Update search direction (β = rsnew/rsold)
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x

def _fisher_vector_product(self, vector, trajectories):
    """
    Compute Fisher Information Matrix - vector product using automatic differentiation
    F * v = ∇_θ KL(π_old || π_θ) * v
    """
    
    # Get trainable parameters
    grad_params = [p for p in self.actor.parameters() if p.requires_grad]
    
    # Compute KL divergence
    kl_div = 0.0
    for trajectory in trajectories:
        kl_div += self._compute_kl_divergence(trajectory)
    kl_div = kl_div / len(trajectories)
    
    # First derivative of KL (gradient)
    kl_grads = torch.autograd.grad(
        kl_div, grad_params, create_graph=True, retain_graph=True
    )
    flat_kl_grad = torch.cat([grad.flatten() for grad in kl_grads])
    
    # Second derivative (Hessian-vector product)
    grad_vector_product = torch.sum(flat_kl_grad * vector)
    hvp = torch.autograd.grad(
        grad_vector_product, grad_params, retain_graph=True
    )
    flat_hvp = torch.cat([grad.flatten() for grad in hvp])
    
    # Add damping for numerical stability
    return flat_hvp + self.damping * vector
```

### 3. Line Search with Trust Region

**Location**: `trpo_diffusion/training/agent.py` (Lines 400-500)

```python
def _line_search(self, search_direction, gradient, trajectories):
    """
    Backtracking line search to find largest step size satisfying:
    1. KL divergence constraint: KL ≤ kl_target
    2. Improvement in objective function
    """
    
    # Maximum theoretical step size from trust region constraint
    # For quadratic approximation: max_step = sqrt(2*δ / (g^T * F^-1 * g))
    max_step_size = torch.sqrt(
        2 * self.kl_target / (torch.dot(gradient, search_direction) + 1e-8)
    )
    
    # Backtracking line search
    step_size = max_step_size
    
    for i in range(self.backtrack_iters):
        # Apply tentative update
        self._apply_update(search_direction, step_size, test=True)
        
        # Check KL constraint
        kl_div = 0.0
        for trajectory in trajectories:
            kl_div += self._compute_kl_divergence(trajectory)
        kl_div = kl_div / len(trajectories)
        
        # Check improvement in objective
        new_loss = self._compute_policy_loss(trajectories)
        old_loss = self._compute_policy_loss(trajectories, use_old_policy=True)
        improvement = old_loss - new_loss
        
        # Accept step if constraints satisfied
        if (kl_div <= self.kl_target and 
            improvement / (torch.dot(gradient, search_direction) * step_size) >= self.accept_ratio):
            
            # Restore parameters and apply accepted update
            self._restore_parameters()
            self._apply_update(search_direction, step_size)
            return step_size
        
        # Reduce step size and try again
        step_size *= self.backtrack_coeff
        self._restore_parameters()
    
    # If no acceptable step found, return zero (no update)
    return 0.0
```

---

## 🖥️ Command Line Usage

### Running the TRPO-Diffusion Pipeline

```bash
# Navigate to project root
cd /Users/bryce2hua/Desktop/RL

# Run TRPO training pipeline
python3 -m trpo_diffusion.train
```

### Syncing and Plotting Results

```bash
# 1. Sync latest results from remote training
./jobs/sync_from_bridges2.sh

# 2. Check what TRPO training sessions are available
ls outputs/logs/

# 3. Plot the latest TRPO training results
python3 trpo_diffusion/tests/plot_test_simple.py

# 4. Compare TRPO vs PPO results
python3 compare_ppo_trpo_results.py
```

---

## ⚙️ Configuration System

### TRPO-Specific Configuration (`utils/constants.py`)

```python
# === TRPO Hyperparameters ===
DEFAULT_KL_TARGET = 0.01              # Trust region size (KL divergence limit)
DEFAULT_DAMPING = 0.1                 # Fisher matrix regularization
DEFAULT_CG_ITERS = 10                 # Conjugate gradient iterations
DEFAULT_BACKTRACK_ITERS = 10          # Line search backtracking steps
DEFAULT_BACKTRACK_COEFF = 0.8         # Line search step reduction factor
DEFAULT_ACCEPT_RATIO = 0.1            # Minimum improvement acceptance ratio

# === Training Parameters ===
DEFAULT_TRAINING_MODE = "SCHEDULER_POLICY"  # Same options as PPO
DEFAULT_REWARD_METRIC = "MMD"              # Same reward metrics as PPO
DEFAULT_NUM_EPISODES = 10000               # Total training episodes
DEFAULT_BATCH_SIZE = 1                     # Images per batch

# === Only Critic Uses Optimizer (TRPO uses natural gradients for actor) ===
DEFAULT_LR_CRITIC = 1e-3              # Critic learning rate only
DEFAULT_GAMMA = 0.99                  # Discount factor
DEFAULT_LAMBDA = 0.95                 # GAE parameter
```

---

## 🔄 Key Differences from PPO

### 1. **Optimization Approach**

| Aspect | PPO | TRPO |
|--------|-----|------|
| **Actor Updates** | Gradient descent with clipping | Natural gradients with trust region |
| **Learning Rate** | Fixed learning rate | Adaptive step size via line search |
| **Constraint** | Clipped probability ratios | KL divergence constraint |
| **Order** | First-order optimization | Second-order optimization |

### 2. **Mathematical Formulation**

**PPO Objective**:
```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

**TRPO Objective**:
```
maximize E[π_θ(a_t|s_t)/π_θ_old(a_t|s_t) * A_t]
subject to E[KL(π_θ_old(·|s_t) || π_θ(·|s_t))] ≤ δ
```

### 3. **Implementation Complexity**

| Component | PPO | TRPO |
|-----------|-----|------|
| **Lines of Code** | ~600 | ~800+ |
| **Memory Usage** | Standard | 2-3x higher |
| **Compute Cost** | Standard | 3-5x higher |
| **Debugging** | Straightforward | Complex |

### 4. **Convergence Properties**

- **PPO**: Faster per iteration, simpler, more stable
- **TRPO**: Slower per iteration, theoretically more principled, potentially better final performance

---

## 🚧 Current Status & Future Work

### ✅ **Fully Implemented Components**

1. **Complete TRPO Algorithm**: All mathematical components implemented
2. **Trust Region Enforcement**: KL divergence constraints with line search
3. **Fisher Information Matrix**: Automatic differentiation-based computation
4. **Conjugate Gradient Solver**: Iterative linear system solver
5. **Multi-Modal Policy Support**: Same three training modes as PPO
6. **Reward System**: Complete modular reward system (MMD, MI, FID)
7. **Infrastructure**: Logging, checkpointing, visualization

### ⚠️ **Testing Status: TO BE TESTED**

**Current Status**: The TRPO implementation is **mathematically complete but untested**. 

**Planned Testing Phases**:

1. **Unit Testing** (Not yet conducted):
   ```bash
   # Test individual TRPO components
   python3 -m trpo_diffusion.tests.test_conjugate_gradients
   python3 -m trpo_diffusion.tests.test_fisher_matrix
   python3 -m trpo_diffusion.tests.test_line_search
   ```

2. **Integration Testing** (Not yet conducted):
   ```bash
   # Test full TRPO training loop
   python3 -m trpo_diffusion.train --episodes 10 --test-mode
   ```

3. **Performance Comparison** (Planned):
   ```bash
   # Compare TRPO vs PPO performance
   python3 compare_algorithms.py --ppo-logs outputs/logs/ppo_* --trpo-logs outputs/logs/trpo_*
   ```

### 🔮 **Areas for Future Development**

#### **Immediate Priorities**

1. **Testing & Validation**:
   - [ ] Unit tests for all TRPO components
   - [ ] Integration testing with small-scale problems
   - [ ] Memory profiling and optimization
   - [ ] Numerical stability validation

2. **Performance Optimization**:
   - [ ] Memory-efficient Fisher matrix computation
   - [ ] GPU acceleration for conjugate gradients
   - [ ] Batch processing optimization
   - [ ] Gradient checkpointing integration

3. **Hyperparameter Tuning**:
   - [ ] KL target sensitivity analysis
   - [ ] Conjugate gradient iteration optimization
   - [ ] Line search parameter tuning
   - [ ] Damping factor optimization

#### **Advanced Features**

1. **Adaptive Trust Region**:
   ```python
   # Dynamic KL target based on training progress
   if improvement_ratio > 0.5:
       self.kl_target *= 1.5  # Expand trust region
   elif improvement_ratio < 0.25:
       self.kl_target *= 0.5  # Shrink trust region
   ```

2. **Preconditioning**:
   ```python
   # Use diagonal Fisher approximation for preconditioning
   def _diagonal_fisher_preconditioner(self, gradient):
       diag_fisher = self._compute_diagonal_fisher()
       return gradient / (diag_fisher + self.damping)
   ```

3. **Distributed Training**:
   ```python
   # Multi-GPU TRPO with gradient aggregation
   def _distributed_conjugate_gradients(self, gradient):
       # Aggregate gradients across GPUs
       # Solve CG in parallel
       pass
   ```

### 🎯 **Expected Advantages Over PPO**

1. **Better Exploration**: Trust region constraints may enable better exploration than clipping
2. **Theoretical Guarantees**: TRPO has stronger theoretical convergence guarantees
3. **Stable Learning**: Natural gradients can provide more stable learning
4. **Complex Domains**: May perform better on complex policy spaces like diffusion models

### ⚠️ **Potential Challenges**

1. **Computational Cost**: 3-5x slower than PPO per iteration
2. **Memory Requirements**: Significantly higher memory usage
3. **Hyperparameter Sensitivity**: More sensitive to hyperparameter choices
4. **Implementation Complexity**: More complex debugging and tuning

### 🚀 **Next Steps**

1. **Begin Small-Scale Testing**: Test on simplified diffusion problems
2. **Memory Profiling**: Ensure it fits within available GPU memory
3. **Baseline Comparison**: Compare against PPO on same problems
4. **Hyperparameter Optimization**: Tune TRPO-specific parameters
5. **Production Readiness**: Add robust error handling and monitoring

---

## 🎯 Summary

The TRPO-diffusion implementation represents a **complete, mathematically sound approach** to policy optimization for diffusion models. While more complex than PPO, it offer
