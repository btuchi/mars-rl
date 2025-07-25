#!/bin/bash

# Sync script to download files from Bridges-2 that don't exist locally

# Configuration - MODIFY THESE VARIABLES
BRIDGES2_USER="btuchi"                                    # Your Bridges-2 username
BRIDGES2_HOST="bridges2.psc.edu"                        # Bridges-2 hostname
REMOTE_DIR="/ocean/projects/eng240004p/btuchi/BRYCE/RL/"                 # Remote directory on Bridges-2
LOCAL_DIR="/Users/bryce2hua/Desktop/RL"                 # FIXED: Always sync to this directory

# Rsync options explanation:
# -a : archive mode (preserves permissions, timestamps, etc.)
# -v : verbose (show what's being transferred)
# -z : compress during transfer
# -h : human readable output
# -P : show progress and keep partial transfers
# --update : only transfer files that are newer on remote
# --exclude : exclude certain files/directories

# Check if local directory exists, create if not
if [ ! -d "${LOCAL_DIR}" ]; then
    echo "Creating local directory: ${LOCAL_DIR}"
    mkdir -p "${LOCAL_DIR}"
fi

echo "========================================"
echo "Syncing TRAINING RESULTS from Bridges-2 to local"
echo "========================================"
echo "Remote: ${BRIDGES2_USER}@${BRIDGES2_HOST}:${REMOTE_DIR}"
echo "Local:  ${LOCAL_DIR}"
echo "Current working directory: $(pwd)"
echo ""

# Initialize success counter
SYNC_SUCCESS=0

# Sync PPO diffusion training results
echo "📦 Syncing PPO diffusion training results..."
rsync -avzhP --update \
    --include="outputs/plots/" \
    --include="outputs/plots/**" \
    --include="outputs/images/" \
    --include="outputs/images/**" \
    --include="outputs/logs/" \
    --include="outputs/logs/**" \
    "${BRIDGES2_USER}@${BRIDGES2_HOST}:/ocean/projects/eng240004p/btuchi/BRYCE/RL/ppo_diffusion/" "${LOCAL_DIR}/ppo_diffusion/"

if [ $? -eq 0 ]; then 
    SYNC_SUCCESS=$((SYNC_SUCCESS + 1))
    echo "✅ PPO diffusion folders synced successfully!"
else
    echo "❌ PPO diffusion sync failed"
fi

# Sync TRPO diffusion training results
echo "📦 Syncing TRPO diffusion training results..."
rsync -avzhP --update \
    --include="outputs/plots/" \
    --include="outputs/plots/**" \
    --include="outputs/images/" \
    --include="outputs/images/**" \
    --include="outputs/logs/" \
    --include="outputs/logs/**" \
    "${BRIDGES2_USER}@${BRIDGES2_HOST}:/ocean/projects/eng240004p/btuchi/BRYCE/RL/trpo_diffusion/" "${LOCAL_DIR}/trpo_diffusion/"

if [ $? -eq 0 ]; then 
    SYNC_SUCCESS=$((SYNC_SUCCESS + 1))
    echo "✅ TRPO diffusion folders synced successfully!"
else
    echo "❌ TRPO diffusion sync failed"
fi

# Check if all syncs were successful
if [ $SYNC_SUCCESS -eq 2 ]; then
    echo ""
    echo "✅ All syncs completed successfully!"
    echo ""
    echo "Files synced to:"
    echo "  - ${LOCAL_DIR}/ppo_diffusion/outputs"
    echo "  - ${LOCAL_DIR}/trpo_diffusion/outputs"
else
    echo ""
    echo "⚠️ Some syncs may have failed ($SYNC_SUCCESS/2 successful)"
    echo "Please check:"
    echo "  1. Your Bridges-2 credentials"
    echo "  2. Network connection"
    echo "  3. Remote directory paths exist"
    echo "  4. Local directory permissions"
fi

echo ""
echo "========================================"