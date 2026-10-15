source ~/miniconda3/etc/profile.d/conda.sh; conda activate uva
cd ~/world-models-uva-lingbot
set -o pipefail
{
date -Is
bash scripts/run_exp02.sh eval checkpoints/pusht.ckpt f1_pusht_zeroshot 10 &&
bash scripts/run_exp02.sh train f2_k25 25 18 &&
bash scripts/run_exp02.sh eval $HOME/world-models-uva-lingbot/results/exp02/train_f2_k25/checkpoints/latest.ckpt f2_k25 10 &&
python scripts/aggregate_seeds.py results/exp02
echo "driver exit=$?"; date -Is
} 2>&1 | tee -a results/exp02/driver_f1_f2.log
