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

# Sync all three folders in one command (single password prompt)
echo "📦 Syncing all training results folders..."
rsync -avzhP --update \
    --include="plots/" \
    --include="plots/**" \
    --include="images/" \
    --include="images/**" \
    --include="logs/" \
    --include="logs/**" \
    --include="images/" \
    --include="images/**" \
    "${BRIDGES2_USER}@${BRIDGES2_HOST}:/ocean/projects/eng240004p/btuchi/BRYCE/RL/ppo_diffusion/outputs/" "${LOCAL_DIR}/ppo_diffusion/outputs/"

if [ $? -eq 0 ]; then 
    SYNC_SUCCESS=3
    echo "✅ All folders synced successfully!"
else
    echo "❌ Sync failed"
fi

# Check if all syncs were successful
if [ $SYNC_SUCCESS -eq 3 ]; then
    echo ""
    echo "✅ All syncs completed successfully!"
    echo ""
    echo "Files synced to: ${LOCAL_DIR}/ppo_diffusion/outputs"
else
    echo ""
    echo "⚠️ Some syncs may have failed ($SYNC_SUCCESS/3 successful)"
    echo "Please check:"
    echo "  1. Your Bridges-2 credentials"
    echo "  2. Network connection"
    echo "  3. Remote directory paths exist"
    echo "  4. Local directory permissions"
fi

echo ""
echo "========================================"