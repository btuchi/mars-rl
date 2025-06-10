#!/bin/bash

# Sync script to download files from Bridges-2 that don't exist locally

# Configuration - MODIFY THESE VARIABLES
BRIDGES2_USER="btuchi"                                    # Your Bridges-2 username
BRIDGES2_HOST="bridges2.psc.edu"                        # Bridges-2 hostname
REMOTE_DIR="/jet/home/btuchi/BRYCE/RL/"                 # Remote directory on Bridges-2
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
echo "Syncing files from Bridges-2 to local"
echo "========================================"
echo "Remote: ${BRIDGES2_USER}@${BRIDGES2_HOST}:${REMOTE_DIR}"
echo "Local:  ${LOCAL_DIR}"
echo "Current working directory: $(pwd)"
echo ""

# Perform the sync
rsync -avzhP --update \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='*.log' \
    --exclude='slurm-*.out' \
    --exclude='core.*' \
    --exclude='*.tmp' \
    "${BRIDGES2_USER}@${BRIDGES2_HOST}:${REMOTE_DIR}" \
    "${LOCAL_DIR}"

# Check if rsync was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Sync completed successfully!"
    echo ""
    echo "Files synced to: ${LOCAL_DIR}"
    echo "You can now run: ls -la to see the synced files"
else
    echo ""
    echo "❌ Sync failed. Please check:"
    echo "  1. Your Bridges-2 credentials"
    echo "  2. Network connection"
    echo "  3. Remote directory path"
    echo "  4. Local directory permissions"
fi

echo ""
echo "========================================"