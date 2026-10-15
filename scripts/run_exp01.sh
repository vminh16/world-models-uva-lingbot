#!/usr/bin/env bash
# EXP-01: evaluate the released UVA checkpoints with the paper's protocol (50 test seeds from 100000, no train seeds),
# repeated over 3 policy-sampling seeds. Run from the repository root, inside the `uva` conda env:
#   bash scripts/run_exp01.sh [n_envs] [tasks...]
# Optional env vars: EXTRA (extra eval_sim.py options), TAG (suffix of the task folder), e.g.
#   TAG=_seeded_goal EXTRA="--seeded_goal True" bash scripts/run_exp01.sh 10 pusht_multitask
# Writes results/exp01/<task>/seed<k>/{command.txt,run.log,vram.csv,ram.csv,exit_code,env_runner.yaml,eval_log_*.json,metrics.json,media/}
set -u

N_ENVS=${1:-10}
shift || true
TASKS=${@:-pusht pusht_multitask}
SEEDS="42 0 1"
EXTRA=${EXTRA:-}
TAG=${TAG:-}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/uva"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for task in $TASKS; do
  for seed in $SEEDS; do
    out="$ROOT/results/exp01/$task$TAG/seed$seed"
    if [ -f "$out/metrics.json" ]; then echo "skip $out (done)"; continue; fi
    mkdir -p "$out"

    cmd="python eval_sim.py -c checkpoints/$task.ckpt -o $out --seed $seed --n_test 50 --n_test_vis 10 --n_train 0 --n_envs $N_ENVS $EXTRA"
    {
      echo "date: $(date -Is)"
      echo "repo: $(git -C "$ROOT" rev-parse HEAD) (dirty: $(git -C "$ROOT" status --porcelain | wc -l) files)"
      echo "uva:  $(git rev-parse HEAD) $(git rev-parse --abbrev-ref HEAD) (dirty: $(git status --porcelain | wc -l) files)"
      echo "ckpt: $(grep " $task.ckpt" checkpoints/SHA256SUMS)"
      echo "cwd:  $ROOT/uva"
      echo "$cmd"
    } > "$out/command.txt"

    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 2 > "$out/vram.csv" &
    vram_pid=$!
    (echo "timestamp,ram_used_mb,ram_available_mb"; while true; do
       echo "$(date +%T),$(free -m | awk '/Mem:/{print $3","$7}')"; sleep 2; done) > "$out/ram.csv" &
    ram_pid=$!

    echo "=== $task seed=$seed n_envs=$N_ENVS -> $out"
    /usr/bin/time -v $cmd > "$out/run.log" 2>&1
    code=$?
    kill $vram_pid $ram_pid 2>/dev/null
    echo "$code" > "$out/exit_code"

    peak_vram=$(tail -n +2 "$out/vram.csv" | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m+0) m=$2+0} END {print m}')
    min_avail=$(tail -n +2 "$out/ram.csv" | awk -F, 'NR==1||$3+0<m+0{m=$3+0} END {print m}')
    wall=$(grep "Elapsed (wall" "$out/run.log" | awk '{print $NF}')
    oom=$(grep -c -i -E "out of memory|OutOfMemoryError|Killed|MemoryError" "$out/run.log")
    echo "exit=$code wall=$wall peak_vram_mib=$peak_vram min_ram_available_mib=$min_avail oom_lines=$oom"

    if [ "$code" = "0" ]; then
      python "$ROOT/scripts/summarize_eval.py" "$out/eval_log_$task.ckpt.json"
    else
      tail -n 20 "$out/run.log"
    fi
  done
done
