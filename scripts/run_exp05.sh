#!/usr/bin/env bash
# EXP-05: UVA libero10.ckpt on LIBERO-10, one process per task so a crash loses one task only.
# Run from the repository root on the VM (LIBERO venv, see docs/ENVIRONMENT.md 2b):
#   bash scripts/run_exp05.sh [n_test] [n_envs] [seed]
# Writes results/exp05/<task>/seed<k>/{command.txt,run.log,vram.csv,ram.csv,exit_code,env_runner.yaml,eval_log_*.json,media/}
set -u

N_TEST=${1:-20}
N_ENVS=${2:-5}
SEED=${3:-42}
PY=${PY:-$HOME/venvs/uva-libero/bin/python}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/uva"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for f in $(ls data/libero_10/*.hdf5 | sort); do
  task=$(basename "$f" _demo.hdf5)
  out="$ROOT/results/exp05/$task/seed$SEED"
  if [ -f "$out/eval_log_libero10.ckpt.json" ]; then echo "skip $task (done)"; continue; fi
  mkdir -p "$out"

  cmd="$PY eval_sim.py -c checkpoints/libero10.ckpt -o $out --seed $SEED --libero_task $task --n_test $N_TEST --n_test_vis 2 --n_train 0 --n_envs $N_ENVS --mp_context spawn"
  {
    echo "date: $(date -Is)"
    echo "repo: $(git -C "$ROOT" rev-parse HEAD) (dirty: $(git -C "$ROOT" status --porcelain | wc -l) files)"
    echo "uva:  $(git rev-parse HEAD) $(git rev-parse --abbrev-ref HEAD) (dirty: $(git status --porcelain | wc -l) files)"
    echo "ckpt: $(grep " libero10.ckpt" checkpoints/SHA256SUMS)"
    echo "env:  MUJOCO_GL=egl"
    echo "$cmd"
  } > "$out/command.txt"

  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$out/vram.csv" &
  vram_pid=$!
  (echo "timestamp,ram_used_mb,ram_available_mb"; while true; do
     echo "$(date +%T),$(free -m | awk '/Mem:/{print $3","$7}')"; sleep 2; done) > "$out/ram.csv" &
  ram_pid=$!

  echo "=== $task seed=$SEED n_test=$N_TEST n_envs=$N_ENVS"
  /usr/bin/time -v $cmd > "$out/run.log" 2>&1
  code=$?
  kill $vram_pid $ram_pid 2>/dev/null
  echo "$code" > "$out/exit_code"

  peak_vram=$(tail -n +2 "$out/vram.csv" | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m+0) m=$2+0} END {print m}')
  min_avail=$(tail -n +2 "$out/ram.csv" | awk -F, 'NR==1||$3+0<m+0{m=$3+0} END {print m}')
  wall=$(grep "Elapsed (wall" "$out/run.log" | awk '{print $NF}')
  score=$(grep -a "^test_mean_score" "$out/run.log" | awk '{print $2}')
  echo "exit=$code wall=$wall peak_vram_mib=$peak_vram min_ram_available_mib=$min_avail success_rate=$score"
  [ "$code" = "0" ] || tail -n 15 "$out/run.log"
done
