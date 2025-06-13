#!/bin/bash
#SBATCH -N1                             
#SBATCH --job-name=diffusion_ppo_training
#SBATCH -p GPU-shared                           
#SBATCH -t 2:00:00                      
#SBATCH --gres=gpu:h100-80:1            # Request 1 H100 (not 4!)
#SBATCH --mem=128G                      
#SBATCH -A eng240004p                    
#SBATCH --mail-user=btuchi@g.hmc.edu     
#SBATCH --mail-type=END,FAIL             
#SBATCH --output=ppo_diffusion/outputs/diffusion_ppo_output_%j.txt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

export RL=/ocean/projects/eng240004p/btuchi/BRYCE/RL      # Set path to your RL project directory

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Activate your virtual environment
echo "Activating virtual environment..."
source $RL/rl_env/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Set CUDA environment variables for memory optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check GPU availability
echo "Checking for GPU:"
nvidia-smi || echo "nvidia-smi failed — no GPU visible"

# Check CUDA availability in Python
echo "Checking CUDA in Python:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check available memory
echo "Available memory:"
free -h

# Navigate to project directory
cd $RL

# Create directories if they don't exist
mkdir -p ppo_diffusion/outputs
mkdir -p ppo_diffusion/models  
mkdir -p ppo_diffusion/plots

# Run training
echo "Starting diffusion PPO training..."
python /ocean/projects/eng240004p/btuchi/BRYCE/RL/ppo_diffusion/diffusion_ppo_trainer.py

echo "Job completed at: $(date)"