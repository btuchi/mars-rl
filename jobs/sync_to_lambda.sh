#!/bin/bash

# Configuration
PEM_KEY="/Users/bryce2hua/Desktop/RL/mars.pem"
REMOTE_USER="ubuntu"
REMOTE_IP="192.222.52.105"
REMOTE_PATH="~/mars-rl/RL/"
LOCAL_BASE="/Users/bryce2hua/Desktop/RL"

# Sync each folder individually
rsync -avz -e "ssh -i $PEM_KEY" "$LOCAL_BASE/jobs"         $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH
rsync -avz -e "ssh -i $PEM_KEY" "$LOCAL_BASE/ppo_diffusion" $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH
rsync -avz -e "ssh -i $PEM_KEY" "$LOCAL_BASE/tests"         $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH
