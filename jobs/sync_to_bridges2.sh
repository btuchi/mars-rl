#!/bin/bash

# Sync script to upload files to Bridges-2

# Configuration - MODIFY THESE VARIABLES
BRIDGES2_USER="btuchi"                                    # Your Bridges-2 username
BRIDGES2_HOST="data.bridges2.psc.edu"                   # Bridges-2 data transfer hostname
REMOTE_DIR="/ocean/projects/eng240004p/btuchi/BRYCE/"    # Remote directory on Bridges-2
LOCAL_DIR="${HOME}/Desktop/RL"                           # Local directory to sync

echo "========================================"
echo "Syncing LOCAL CODE to Bridges-2"
echo "========================================"
echo "Local:  ${LOCAL_DIR}"
echo "Remote: ${BRIDGES2_USER}@${BRIDGES2_HOST}:${REMOTE_DIR}"
echo ""

# Rsync options explanation:
# -a : archive mode (preserves permissions, timestamps, etc.)
# -v : verbose (show what's being transferred)
# -z : compress during transfer
# --exclude : exclude certain files/directories

echo "📦 Syncing both PPO and TRPO diffusion codebases..."
echo "Excluded: rl_env (virtual environment)"
echo ""

rsync -avz --exclude 'rl_env' "${LOCAL_DIR}" "${BRIDGES2_USER}@${BRIDGES2_HOST}:${REMOTE_DIR}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Sync completed successfully!"
    echo ""
    echo "Synced directories:"
    echo "  - ppo_diffusion/"
    echo "  - trpo_diffusion/"
    echo "  - jobs/"
    echo "  - (other files)"
else
    echo ""
    echo "❌ Sync failed"
    echo "Please check:"
    echo "  1. Your Bridges-2 credentials"
    echo "  2. Network connection"
    echo "  3. Remote directory permissions"
fi

echo ""
echo "========================================"
