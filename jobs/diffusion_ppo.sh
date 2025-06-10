#!/bin/bash
#SBATCH -N1                             # Request 1 node
#SBATCH --job-name=diffusion_ppo_training
#SBATCH -p GPU-shared                    # Use the GPU-shared partition
#SBATCH -t 2:00:00                      # Job time limit: 2 hours
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=32G                       # Request 32GB of memory
#SBATCH -A eng240004p                    # Charge time to your PSC allocation
#SBATCH --mail-user=btuchi@g.hmc.edu     # Email for job updates
#SBATCH --mail-type=END,FAIL             # Email on job completion or failure
#SBATCH --output=diffusion_ppo_output_%j.txt  # Include job ID in output filename

export RL=/jet/home/btuchi/BRYCE/RL      # Set path to your RL project directory

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Activate your virtual environment
echo "Activating virtual environment..."
source /jet/home/btuchi/BRYCE/RL/rl_env/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Check GPU availability
echo "Checking for GPU:"
nvidia-smi || echo "nvidia-smi failed — no GPU visible"

# Check CUDA availability in Python
echo "Checking CUDA in Python:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Check available memory
echo "Available memory:"
free -h

# Navigate to project directory
cd $RL

# Run training
echo "Starting diffusion PPO training..."
python /jet/home/btuchi/BRYCE/RL/ppo_diffusion/diffusion_ppo_trainer.py

echo "Job completed at: $(date)"