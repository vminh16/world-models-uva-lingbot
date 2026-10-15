# F4 then F3, started after the F1/F2 driver (tmux session exp02) has finished.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate uva
cd ~/world-models-uva-lingbot
while tmux has-session -t =exp02 2>/dev/null; do sleep 60; done
[ -f scripts/run_exp02.sh.next ] && mv scripts/run_exp02.sh.next scripts/run_exp02.sh
set -o pipefail
R=$HOME/world-models-uva-lingbot/results/exp02
{
date -Is
grep -q "driver exit=0" results/exp02/driver_f1_f2.log || { echo "F1/F2 driver did not finish cleanly, stop"; exit 1; }
# F4: same data, budget and flags as F2, action objective only
bash scripts/run_exp02.sh train f4_k25_action_only 25 18 model.policy.selected_training_mode=policy_model &&
bash scripts/run_exp02.sh eval $R/train_f4_k25_action_only/checkpoints/latest.ckpt f4_k25_action_only 10 &&
# F3: nested subsets k50 and k10 (k25 = F2), same step budget
bash scripts/run_exp02.sh train f3_k50 k50 &&
bash scripts/run_exp02.sh eval $R/train_f3_k50/checkpoints/latest.ckpt f3_k50 10 &&
bash scripts/run_exp02.sh train f3_k10 k10 &&
bash scripts/run_exp02.sh eval $R/train_f3_k10/checkpoints/latest.ckpt f3_k10 10 &&
python scripts/aggregate_seeds.py results/exp02
echo "driver exit=$?"; date -Is
} 2>&1 | tee -a results/exp02/driver_f3_f4.log
