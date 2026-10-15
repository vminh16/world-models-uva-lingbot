#!/bin/bash
# wait for the LingBot-VA prompt run (exact tmux name), then EXP-07 in the uva env
while tmux has-session -t =lb_prompt 2>/dev/null; do sleep 20; done
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate uva 2>/dev/null
echo "python: $(which python)"
cd ~/world-models-uva-lingbot && bash scripts/run_exp07.sh
