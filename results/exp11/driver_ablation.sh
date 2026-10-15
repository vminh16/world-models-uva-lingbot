#!/bin/bash
# EXP-11 small ablations, all seed 0, 10 chunks unless noted; compared with i2av_c10_s0 (same initial noise).
cd ~/world-models-uva-lingbot
source ~/venvs/lingbot/bin/activate
export PYTHONNOUSERSITE=1
R=scripts/run_exp11.sh
bash $R steps_v10_a20 10 0 num_inference_steps=10 action_num_inference_steps=20 > results/exp11/steps_v10_a20.driver.log 2>&1
bash $R steps_v5_a10 10 0 num_inference_steps=5 action_num_inference_steps=10 > results/exp11/steps_v5_a10.driver.log 2>&1
bash $R cfg1 10 0 guidance_scale=1 > results/exp11/cfg1.driver.log 2>&1
bash $R win8_c40 40 0 attn_window=8 > results/exp11/win8_c40.driver.log 2>&1
bash $R prompt_left 10 0 "prompt=Grab the medium-sized white mug and place it on the left side of the table." > results/exp11/prompt_left.driver.log 2>&1
echo DRIVER_DONE
