#!/bin/bash

# === User Config ===
REMOTE_USER=btuchi
REMOTE_HOST=data.bridges2.psc.edu
REMOTE_PATH=/ocean/projects/eng240004p/btuchi/BRYCE/RL

# === Sync Local RL/ → Remote RL/ (clean mirror) ===
rsync -avz --delete --exclude 'rl_env/' /Users/bryce2hua/Desktop/RL/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}
