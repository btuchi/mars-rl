#!/bin/bash
rsync -avz --exclude 'rl_env' ~/Desktop/RL btuchi@data.bridges2.psc.edu:/jet/home/btuchi/BRYCE/
# how can I ignore rl_env when I sync?