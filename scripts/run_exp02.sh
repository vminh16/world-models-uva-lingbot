#!/usr/bin/env bash
# EXP-02: fine-tune UVA on PushT-M (random goals) from few demonstrations, and evaluate before / after.
# Run from the repository root on the VM, inside the `uva` conda env.
#
#   bash scripts/run_exp02.sh eval  <ckpt> <name> [n_envs]
#       PushT-M protocol of EXP-01 (--fix_goal False --seeded_goal True, 50 test seeds, policy seeds 42 0 1)
#       -> results/exp02/<name>/seed<k>/{command.txt,run.log,vram.csv,ram.csv,exit_code,eval_log_*.json,metrics.json,media/}
#   bash scripts/run_exp02.sh train <name> <k_demos> <num_epochs> [extra hydra overrides...]
#       fine-tune checkpoints/pusht.ckpt on <k_demos> training episodes of data/pusht_multitask chosen by upstream
#       `downsample_mask` (seed 42)
#   bash scripts/run_exp02.sh train <name> <subset> [extra hydra overrides...]
#       same on a nested subset of results/exp02/subsets.json (k10 / k25 / k50, from exp02_nested_subsets.py), with
#       the subset's epochs (same step budget as F2) and FVD every `val_every` epochs
#       -> results/exp02/train_<name>/{command.txt,run.log,logs.jsonl,vram.csv,ram.csv,checkpoints/latest.ckpt,...}
#
# <ckpt> is relative to uva/ or absolute.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/uva"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline

start_monitors() {  # $1 = output dir
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 2 > "$1/vram.csv" &
  vram_pid=$!
  (echo "timestamp,ram_used_mb,ram_available_mb"; while true; do
     echo "$(date +%T),$(free -m | awk '/Mem:/{print $3","$7}')"; sleep 2; done) > "$1/ram.csv" &
  ram_pid=$!
}

stop_monitors() {  # $1 = output dir
  kill $vram_pid $ram_pid 2>/dev/null
  peak_vram=$(tail -n +2 "$1/vram.csv" | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m+0) m=$2+0} END {print m}')
  min_avail=$(tail -n +2 "$1/ram.csv" | awk -F, 'NR==1||$3+0<m+0{m=$3+0} END {print m}')
  wall=$(grep "Elapsed (wall" "$1/run.log" | awk '{print $NF}')
  oom=$(grep -c -i -E "out of memory|OutOfMemoryError|Killed|MemoryError" "$1/run.log")
  echo "exit=$code wall=$wall peak_vram_mib=$peak_vram min_ram_available_mib=$min_avail oom_lines=$oom"
}

header() {  # $1 = output dir, $2 = command
  {
    echo "date: $(date -Is)"
    echo "repo: $(git -C "$ROOT" rev-parse HEAD) (dirty: $(git -C "$ROOT" status --porcelain | wc -l) files)"
    echo "uva:  $(git rev-parse HEAD) $(git rev-parse --abbrev-ref HEAD) (dirty: $(git status --porcelain | wc -l) files)"
    echo "cwd:  $ROOT/uva"
    echo "$2"
  } > "$1/command.txt"
}

mode=$1; shift
case $mode in
eval)
  CKPT=$1; NAME=$2; N_ENVS=${3:-10}
  sha=$(sha256sum "$CKPT" | awk '{print $1}')
  for seed in 42 0 1; do
    out="$ROOT/results/exp02/$NAME/seed$seed"
    if [ -f "$out/metrics.json" ]; then echo "skip $out (done)"; continue; fi
    mkdir -p "$out"
    cmd="python eval_sim.py -c $CKPT -o $out --seed $seed --n_test 50 --n_test_vis 10 --n_train 0 --n_envs $N_ENVS --fix_goal False --seeded_goal True"
    header "$out" "$cmd"
    echo "ckpt: $sha  $CKPT" >> "$out/command.txt"
    start_monitors "$out"
    echo "=== eval $NAME seed=$seed -> $out"
    /usr/bin/time -v $cmd > "$out/run.log" 2>&1
    code=$?
    echo "$code" > "$out/exit_code"
    stop_monitors "$out"
    if [ "$code" = "0" ]; then
      python "$ROOT/scripts/summarize_eval.py" "$out/eval_log_$(basename "$CKPT").json"
    else
      tail -n 20 "$out/run.log"; exit 1
    fi
  done
  ;;
train)
  NAME=$1; K=$2; shift 2
  if [[ $K == k* ]]; then
    subsets="$ROOT/results/exp02/subsets.json"
    read -r EPISODES EPOCHS VAL_EVERY < <(python -c "import json; s=json.load(open('$subsets'))['$K']; \
print(str(s['episodes']).replace(' ', ''), s['epochs'], s['val_every'])")
    DATA="+task.dataset.train_episodes=$EPISODES training.val_every=$VAL_EVERY"
  else
    EPOCHS=$1; shift
    DATA="+task.dataset.max_train_episodes=$K"
  fi
  out="$ROOT/results/exp02/train_$NAME"
  if [ -f "$out/exit_code" ] && [ "$(cat "$out/exit_code")" = "0" ]; then echo "skip $out (done)"; exit 0; fi
  mkdir -p "$out"
  # The env runner is only used by the rollout that always runs after epoch 0 (rollout_every is never hit again);
  # n_train=1 avoids the KeyError on train/mean_score. The final checkpoint is evaluated separately with `eval`.
  cmd="python train.py --config-dir=. --config-name=uva_pusht.yaml \
model.policy.autoregressive_model_params.pretrained_model_path=checkpoints/pusht.ckpt \
model.policy.action_model_params.predict_action=True \
task.dataset.dataset_path=data/pusht_multitask task.dataset.dataset_type=multitask $DATA \
task.env_runner.fix_goal=False +task.env_runner.seeded_goal=True \
task.env_runner.n_envs=10 task.env_runner.n_train=1 task.env_runner.n_train_vis=0 task.env_runner.n_test=10 \
training.num_epochs=$EPOCHS training.lr_warmup_steps=200 training.rollout_every=100000 \
training.checkpoint_every=$((EPOCHS - 1)) checkpoint.topk.k=0 \
dataloader.batch_size=16 dataloader.num_workers=4 val_dataloader.num_workers=2 \
logging.mode=offline hydra.run.dir=$out $*"
  header "$out" "$cmd"
  echo "init: $(grep " pusht.ckpt" checkpoints/SHA256SUMS)" >> "$out/command.txt"
  start_monitors "$out"
  echo "=== train $NAME k=$K epochs=$EPOCHS -> $out"
  /usr/bin/time -v $cmd > "$out/run.log" 2>&1
  code=$?
  echo "$code" > "$out/exit_code"
  stop_monitors "$out"
  if [ "$code" = "0" ]; then
    sha256sum "$out/checkpoints/latest.ckpt" > "$out/checkpoints/SHA256SUMS"
    cat "$out/checkpoints/SHA256SUMS"
  else
    tail -n 20 "$out/run.log"; exit 1
  fi
  ;;
*)
  echo "usage: $0 eval <ckpt> <name> [n_envs] | train <name> <k_demos> <num_epochs> [overrides...] | train <name> <k10|k25|k50> [overrides...]"; exit 2
  ;;
esac
