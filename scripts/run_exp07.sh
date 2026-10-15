#!/usr/bin/env bash
# EXP-07 (W3): UVA inference cost vs action denoising steps, pusht.ckpt.
#   1. PushT, paper protocol (50 test seeds from 100000), 10 parallel envs, policy seed 42 -> score per step count
#   2. one env (batch 1), 1 test seed -> latency of one action chunk, comparable with LingBot-VA's batch-1 i2av
# Run from the repository root in the `uva` conda env:  bash scripts/run_exp07.sh
# Writes results/exp07/{pusht,b1}_steps<N>/{run.log,vram.csv,latency.json,eval_log_*.json,metrics.json}
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/uva"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {  # run <out> <eval_sim options...>
  out=$1; shift
  if [ -f "$out/latency.json" ]; then echo "skip $out (done)"; return; fi
  mkdir -p "$out"
  echo "python eval_sim.py -c checkpoints/pusht.ckpt -o $out $*" > "$out/command.txt"
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 2 > "$out/vram.csv" &
  vram_pid=$!
  python eval_sim.py -c checkpoints/pusht.ckpt -o "$out" "$@" > "$out/run.log" 2>&1
  echo $? > "$out/exit_code"
  kill $vram_pid
  python "$ROOT/scripts/summarize_eval.py" "$out/eval_log_pusht.ckpt.json" >> "$out/run.log" 2>&1 || true
}

for s in 100 50 20 10 5; do
  run "$ROOT/results/exp07/pusht_steps$s" --seed 42 --n_test 50 --n_test_vis 4 --n_train 0 --n_envs 10 \
      --act_diff_testing_steps $s
done
for s in 100 50 20 10 5; do
  run "$ROOT/results/exp07/b1_steps$s" --seed 42 --n_test 1 --n_test_vis 0 --n_train 0 --n_envs 1 \
      --act_diff_testing_steps $s
done
echo EXP07_DONE
