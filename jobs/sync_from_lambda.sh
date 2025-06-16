#!/bin/bash

# Sync script to download files from Lambda that don't exist locally

# Configuration - MODIFY THESE VARIABLES IF NEEDED
LAMBDA_USER="ubuntu"
LAMBDA_HOST="192.222.52.105"
PEM_KEY="/Users/bryce2hua/Desktop/RL/mars.pem"
REMOTE_DIR="~/mars-rl/RL/ppo_diffusion/"                 # Remote directory on Lambda
LOCAL_DIR="/Users/bryce2hua/Desktop/RL/ppo_diffusion"    # Local destination

# Rsync options explanation:
# -a : archive mode (preserves permissions, timestamps, etc.)
# -v : verbose
# -z : compress during transfer
# -h : human readable output
# -P : progress bar
# --update : only transfer if the file is newer on remote
# --include/--exclude : selectively sync certain folders

# Check if local directory exists, create if not
if [ ! -d "${LOCAL_DIR}" ]; then
    echo "Creating local directory: ${LOCAL_DIR}"
    mkdir -p "${LOCAL_DIR}"
fi

echo "========================================"
echo "Syncing TRAINING RESULTS from Lambda Lab to local"
echo "========================================"
echo "Remote: ${LAMBDA_USER}@${LAMBDA_HOST}:${REMOTE_DIR}"
echo "Local:  ${LOCAL_DIR}"
echo "Current working directory: $(pwd)"
echo ""

# Initialize success flag
SYNC_SUCCESS=0

# Run rsync
echo "📦 Syncing outputs/, models/, plots/ from Lambda..."
rsync -avzhP --update \
    -e "ssh -i $PEM_KEY" \
    --include="outputs/" \
    --include="outputs/**" \
    --include="models/" \
    --include="models/**" \
    --include="plots/" \
    --include="plots/**" \
    --exclude="*" \
    "${LAMBDA_USER}@${LAMBDA_HOST}:${REMOTE_DIR}" "${LOCAL_DIR}/"

if [ $? -eq 0 ]; then 
    SYNC_SUCCESS=3
    echo "✅ All folders synced successfully!"
else
    echo "❌ Sync failed"
fi

# Result summary
if [ $SYNC_SUCCESS -eq 3 ]; then
    echo ""
    echo "✅ All syncs completed successfully!"
    echo ""
    echo "Files synced to: ${LOCAL_DIR}/"
    echo "  📄 Job outputs: outputs/"
    echo "  🏗️ Model weights: models/"  
    echo "  📊 Training plots: plots/"
else
    echo ""
    echo "⚠️ Some syncs may have failed ($SYNC_SUCCESS/3 successful)"
    echo "Please check:"
    echo "  1. Your PEM key path and permissions"
    echo "  2. SSH access to Lambda"
    echo "  3. Remote directory paths"
    echo "  4. Local directory permissions"
fi

echo ""
echo "========================================"
