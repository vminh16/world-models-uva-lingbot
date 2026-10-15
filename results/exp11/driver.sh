#!/bin/bash
# EXP-11 L3: 10 chunks x seeds 0,1,2 (authors' default length), then 40 chunks (past the 36-chunk KV window), seed 0.
cd ~/world-models-uva-lingbot
source ~/venvs/lingbot/bin/activate
export PYTHONNOUSERSITE=1
for s in 0 1 2; do
  bash scripts/run_exp11.sh i2av_c10_s$s 10 $s > results/exp11/i2av_c10_s$s.driver.log 2>&1
done
bash scripts/run_exp11.sh i2av_c40_s0 40 0 > results/exp11/i2av_c40_s0.driver.log 2>&1
echo DRIVER_DONE
