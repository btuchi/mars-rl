#!/bin/bash
#SBATCH -N1                             # Request 1 node
#SBATCH --job-name=simple_test_job
#SBATCH -p GPU-shared                    # Use the GPU-shared partition (good for shared GPU access)
#SBATCH -t 0:30:00                      # Job time limit:30 minutes
#SBATCH --gres=gpu:1 # Request1 GPU
#SBATCH -A eng240004p                    # Charge time to your PSC allocation
#SBATCH --mail-user=btuchi@g.hmc.edu     # Email for job updates
#SBATCH --mail-type=END,FAIL             # Email on job completion or failure
#SBATCH --output=simple_traj_test_output.txt

export RL=/ocean/projects/eng240004p/btuchi/BRYCE/RL      # Set path to your RL project directory

# Activate your virtual environment
source /jet/home/btuchi/BRYCE/RL/rl_env/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

echo "Checking for GPU:"
nvidia-smi || echo "nvidia-smi failed — no GPU visible"

# Run test
echo "Running test..."
python /ocean/projects/eng240004p/btuchi/BRYCE/RL/tests/simple_test_traj_recording.py
