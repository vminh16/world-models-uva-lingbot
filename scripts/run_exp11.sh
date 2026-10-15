#!/bin/bash
# EXP-11: LingBot-VA image -> video + action generation (i2av) on one L4, offload on.
# Usage (from the repository root, venv ~/venvs/lingbot, see scripts/exp11_setup.sh):
#   [MODEL=<model_dir>] bash scripts/run_exp11.sh <name> <num_chunks> <seed> [KEY=VALUE ...]
# KEY=VALUE pairs override the robotwin_i2av config (wan_va_server.py --set), e.g. num_inference_steps=10.
# T5 prompt embeddings are cached in results/exp11/prompt_cache/ (the first run of a prompt pays about 270 s).
# Output in results/exp11/<name>/: demo.mp4, actions.npy, timing.json, run.log, vram.csv (GPU memory every 2 s),
# ram.csv (used RAM and swap every 2 s).
set -e
NAME=$1; CHUNKS=$2; SEED=$3; shift 3
MODEL=${MODEL:-$HOME/models/lingbot-va-posttrain-robotwin}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/results/exp11/$NAME
mkdir -p "$OUT"

nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 2 > "$OUT/vram.csv" &
VRAM_PID=$!
( echo "time,ram_used_mib,swap_used_mib"
  while true; do echo "$(date +%T),$(free -m | awk '/Mem:/{print $3}'),$(free -m | awk '/Swap:/{print $3}')"; sleep 2; done ) > "$OUT/ram.csv" &
RAM_PID=$!
trap 'kill $VRAM_PID $RAM_PID 2>/dev/null' EXIT

cd "$ROOT/lingbot-va"   # input_img_path in the config is relative (example/robotwin)
git rev-parse HEAD > "$OUT/commit.txt"
START=$(date +%s)
# same command as script/run_launch_va_server_sync.sh with NGPU=1, called directly because that script passes its
# arguments unquoted ($overrides), which splits a prompt with spaces
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=http://localhost:29510 \
python -m torch.distributed.run --nproc_per_node=1 --local-ranks-filter=0 --master_port 29501 --tee 3 \
    -m wan_va.wan_va_server --config-name robotwin_i2av \
    --model_path "$MODEL" --enable_offload true --num_chunks "$CHUNKS" --seed "$SEED" --save_root "$OUT" \
    --set prompt_embeds_cache_dir="$ROOT/results/exp11/prompt_cache" "$@" \
    2>&1 | tee "$OUT/run.log"
echo "wall_seconds=$(( $(date +%s) - START ))" | tee -a "$OUT/run.log"
