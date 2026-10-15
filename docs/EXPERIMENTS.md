# Experiments

Every train or evaluation run that produces a number has a row here; the report quotes numbers only from this file and the
`metrics.json` files under `results/`. IDs: `EXP-0x` UVA, `EXP-1x` LingBot-VA, `EXP-2x` other.
"(ours)" = measured here, "(reported)" = from the paper or the official README.

## Plan

| ID | Priority | Model | Question | Setup | Metric | Output |
|---|---|---|---|---|---|---|
| EXP-01 | P0 | UVA | Do the released checkpoints reproduce the paper? | `pusht.ckpt` on PushT, `pusht_multitask.ckpt` on PushT-M, fixed seeds | success rate, mean target coverage | `results/exp01/` |
| EXP-02 | P0 | UVA | Can a checkpoint trained with one fixed goal (`pusht.ckpt`) adapt to random goals (PushT-M) with little data and compute? | fine-tune on k = 25 released PushT-M demonstrations, about 3 GPU hours on one L4; evaluate before (F1) and after (F2) with the EXP-01 PushT-M protocol; upper bound `pusht_multitask.ckpt` | success rate, mean score before / after | `results/exp02/` |
| EXP-03 | P1 | UVA | How much data does adaptation need (F3), and does the video objective help (F4)? | EXP-02 with nested 10 / 25 / 50 demonstrations, same step budget; F4 = 25 demonstrations, action objective only | mean score and success rate vs number of demonstrations | `results/exp02/` (same folder as EXP-02) |
| EXP-04 | P1 | UVA | Does the imagined future match the simulator, and does the mismatch explain failures? | compare UVA's predicted frames with the simulator's frames along rollouts, before and after fine-tuning | per-frame error (for example PSNR), linked to success / failure | `results/exp04/` |
| EXP-05 | P2 | UVA | Same benchmark as LingBot-VA | `libero10.ckpt` on LIBERO-10, `MUJOCO_GL=egl` | success rate | `results/exp05/` |
| EXP-11 | P0 | LingBot-VA | What does it imagine, how fast, with how much memory on one L4? | `robotwin_i2av`, offload on | latency per chunk, peak GPU memory, generated videos | `results/exp11/` |
| EXP-12 | P2 | LingBot-VA | Same benchmark as UVA, if it fits 24 GB | `lingbot-va-posttrain-libero-long` on a subset of LIBERO-Long tasks | success rate | `results/exp12/` |

Reported reference numbers (official LingBot-VA README): RoboTwin 2.0 92.9% (easy) / 91.6% (hard) averaged over 50 tasks,
LIBERO 98.5 average. UVA reference numbers: to be copied from the paper's tables when EXP-01 is written up.

## Training cost on one L4 (Phase 2, 2026-10-14)

Fine-tuning `pusht.ckpt` on 25 PushT-M demonstrations (joint video + action, all 5 task modes), fp16, 4 dataloader
workers, 50 steps each (`results/exp02/smoke_bs*/`):

| Batch size | Time per step | Samples / s | Peak GPU memory | Steps per epoch (25 demos) |
|---|---|---|---|---|
| 8 | 1.56 s | 5.1 | 19.1 GB | 335 |
| 16 | 3.17 s | 5.0 | 21.6 GB | 168 |
| 32 (paper, per GPU) | – | – | does not fit without gradient checkpointing | – |

Other costs per run: start-up and checkpoint loading about 1 min; FVD on 4 validation batches after every epoch about
1 min; one rollout after epoch 0 (11 episodes) about 2 min; `latest.ckpt` 4.7 GB (with optimizer state). The trained
checkpoint loads in `eval_sim.py` (checked). Throughput is the same for batch 8 and 16, so about **1100 steps of batch 16
(18k samples, about 7 epochs over 25 demonstrations) per GPU hour**.

## Results

| ID | Checkpoint | Task | Change from upstream | Metric | Result (ours) | Reported | Run time | Peak GPU memory | Output |
|---|---|---|---|---|---|---|---|---|---|
| EXP-07 | `pusht.ckpt`, action head denoising steps 100 (checkpoint) / 50 / 20 / 10 / 5 | PushT, 50 seeds × policy seed 42, 10 envs; batch-1 latency on 1 seed | `--act_diff_testing_steps`, `latency.json` in `eval_sim.py` | mean score; seconds per `predict_action` call (8 executed actions) | 0.957 / 0.973 / 0.972 / 0.976 / 0.995 (paired differences vs 100 steps all include 0); batch 1: 0.48 / 0.30 / 0.19 / 0.16 / 0.14 s; batch 10: 1.65 → 1.30 s | – | about 6 min per 50 seeds | 1.9 GiB (batch 1), 8.0 GiB (batch 10) | `results/exp07/` |
| EXP-11 | `robbyant/lingbot-va-posttrain-robotwin` (24.4 GB), i2av, offload on, 3 seeds × 10 chunks + 1 × 40 chunks | RoboTwin example image (3 cameras) + mug-on-rack prompt, no simulator (open-loop imagination) | `--model_path --enable_offload --num_chunks --seed` options, flash-attn import fix (ISSUE-013) | seconds per chunk (2 latent frames = 8 video frames, 32 actions); peak GPU memory | 17.1 s per chunk (chunks 1–9, 3 seeds), rising linearly to 23.6 s at chunk 36, flat after; first chunk 39–40 s; prompt encoding on CPU 268 s; peak 17.0 GiB allocated / 18.1 GB nvidia-smi | about 18 GB VRAM with offload (README) | 10 min per 10-chunk run, 20 min for 40 | 18.1 GB | `results/exp11/` |
| EXP-05 | `libero10.ckpt` | LIBERO-10, 10 tasks × 20 episodes, policy seed 42 | `eval_sim.py` options, LIBERO venv (ISSUE-006) | mean success over tasks | 0.865 [0.811, 0.906] | 0.90 | about 55 min | 10.3 GB | `results/exp05/` |
| EXP-03 F3 | `pusht.ckpt` fine-tuned on 10 PushT-M demos (nested subset), 3034 steps (sha256 `22a5a53c…`) | PushT-M, same protocol as F1 | `train_episodes` option, 37 epochs, FVD every 2 | success / mean score | 0.000 [0.000, 0.025] / 0.421 ± 0.023 | – | train 2 h 53 min; eval 6 min per 50 seeds | train 21.6 GB, eval 9.3 GiB | `results/exp02/f3_k10/`, `train_f3_k10/` |
| EXP-03 F3 | `pusht.ckpt` fine-tuned on 50 PushT-M demos (nested superset), 3144 steps (sha256 `d694bf13…`) | PushT-M, same protocol as F1 | `train_episodes` option, 8 epochs | success / mean score | 0.013 [0.004, 0.047] / 0.593 ± 0.040 | – | train 2 h 53 min; eval 6 min per 50 seeds | train 21.6 GB, eval 9.3 GiB | `results/exp02/f3_k50/`, `train_f3_k50/` |
| EXP-03 F4 | `pusht.ckpt` fine-tuned on the same 25 demos, 3024 steps, **action objective only** (`selected_training_mode=policy_model`, sha256 `a4df5f51…`) | PushT-M, same protocol as F1 | F2 command + `policy_model` | success / mean score | 0.013 [0.004, 0.047] / 0.454 ± 0.042 | – | train 2 h 47 min; eval 6 min per 50 seeds | train 21.5 GB, eval 9.3 GiB | `results/exp02/f4_k25_action_only/`, `train_f4_k25_action_only/` |
| EXP-02 F2 | `pusht.ckpt` fine-tuned on 25 PushT-M demos, 3024 steps (sha256 `8bd50547…`) | PushT-M, same protocol as F1 | training overrides in EXP-02 details | success / mean score | 0.007 [0.001, 0.037] / 0.517 ± 0.037 | – | train 2 h 52 min; eval 6 min per 50 seeds | train 21.6 GB, eval 9.3 GiB | `results/exp02/f2_k25/`, `train_f2_k25/` |
| EXP-02 F1 | `pusht.ckpt` (never trained on PushT-M) | PushT-M, `--fix_goal False --seeded_goal True`, 50 goals × 3 policy seeds | `eval_sim.py` overrides + `seeded_goal` | success / mean score | 0.000 [0.000, 0.025] / 0.180 ± 0.008 | – | 6 min per 50 seeds | 9.3 GiB | `results/exp02/f1_pusht_zeroshot/` |
| EXP-01 | `pusht.ckpt` | PushT, 50 seeds × 3 policy seeds | `eval_sim.py` overrides only | success / mean score | 0.733 ± 0.058 / 0.974 ± 0.016 | – / 0.98 | 5 min 50 s per 50 seeds | 9.3 GiB | `results/exp01/` |
| EXP-01 | `pusht_multitask.ckpt` | PushT-M, `--seeded_goal True`, 50 goals × 3 policy seeds | `eval_sim.py` overrides + `seeded_goal` | success / mean score | 0.253 ± 0.012 / 0.813 ± 0.030 | – | 6 min per 50 seeds | 9.3 GiB | `results/exp01/` |
| EXP-01 | `pusht_multitask.ckpt` | PushT-M, 50 seeds × 3 policy seeds | `eval_sim.py` overrides only | success / mean score | 0.220 ± 0.035 / 0.787 ± 0.008 | – / 0.88 (1 goal, ISSUE-005) | 5 min 55 s per 50 seeds | 9.3 GiB | `results/exp01/` |

## Details

### EXP-02 — Fine-tuning `pusht.ckpt` to random goals (PushT-M)
- Date / commit: 2026-10-14 / `aad9f1d` + uncommitted / `uva` `fe3dec8` + changes listed in the README
- Question: can a checkpoint trained with one fixed goal adapt to random goals from 25 demonstrations and about 3 hours
  on one L4?
- Exact commands (driver `results/exp02/driver_f1_f2.sh`):
  ```bash
  bash scripts/run_exp02.sh eval checkpoints/pusht.ckpt f1_pusht_zeroshot 10          # F1, lower bound
  bash scripts/run_exp02.sh train f2_k25 25 18                                         # F2, fine-tuning
  bash scripts/run_exp02.sh eval <abs>/results/exp02/train_f2_k25/checkpoints/latest.ckpt f2_k25 10
  python scripts/aggregate_seeds.py results/exp02
  ```
- Training overrides (F2): init `pusht.ckpt` (EMA weights), data `data/pusht_multitask` with
  `max_train_episodes=25` (episodes in `results/exp02/dataset_k25.json`: 3248 frames, 2673 windows of 32 steps),
  all 5 task modes (joint video + action), batch 16, 18 epochs = 3024 steps, AdamW lr 1e-4 with 200 warmup steps and
  cosine decay to 0, EMA, fp16. Normalizer refit on the whole PushT-M zarr (action / position ranges of all 247
  episodes, not only the 25 training ones). The other 222 episodes are the validation set (FVD on 4 batches per epoch).
- Evaluation: EXP-01 PushT-M protocol (`--fix_goal False --seeded_goal True`, test seeds 100000–100049, policy seeds 42,
  0, 1), the same 50 goals for F1, F2 and the upper bound.
- F1 result: 0/150 successes, mean score 0.180 ± 0.008 (`results/exp02/summary.json`). Where the block ends
  (`scripts/goal_distance.py`, block centroid in the last frame of the 30 recorded videos, 96 px frames): median 11.3 px
  from the fixed PushT goal and 21.7 px from the episode's goal; `pusht_multitask.ckpt` on the same goals ends 2.3 px from
  the episode goal. `pusht.ckpt` ignores the drawn goal and drifts toward where the goal always was during its training.
- F2 training: 2 h 52 min wall, peak GPU 21.6 GB, minimum available RAM 3.5 GB, no OOM. Action loss (mean over the steps
  whose sampled mode has an action target) 0.151 in epoch 0 → 0.038 in epoch 11; video loss 0.037 → 0.027 (the video
  model already knew PushT dynamics). FVD on 4 validation batches jumps between 1170 and 1760 with no trend: too few
  samples to read anything from it. Curves: `results/exp02/figures/train_curves_f2.png`. Rollout after epoch 0
  (10 goals, one policy seed): mean score 0.465.
- F2 result: success 1/150, mean score 0.517 ± 0.037 (repeats 0.536 / 0.474 / 0.540). Paired over the 50 goals
  (bootstrap 95%): F2 − F1 = +0.337 [0.262, 0.411], upper bound − F2 = 0.296 [0.220, 0.377]; F2 closes 53% of the
  gap between the lower and the upper bound, and beats F1 on 45/50 goals. Median max reward per episode 0.114 (F1) →
  0.609 (F2) → 0.887 (upper bound); 75/150 F2 episodes reach 0.6–0.95 coverage, 1 reaches 0.95.
- Where the block ends (`scripts/goal_distance.py`): median 4.3 px from the episode goal (F1 21.7 px, upper bound
  2.3 px), 16.3 px from the old fixed goal. F2 has learned to read the goal from the image and brings the T onto it,
  but not the final alignment that the success threshold (95% coverage) needs: last frames show the T on the goal with
  a residual rotation or offset (`results/exp02/figures/last_frames_f1_f2_upper.png`). Lowest mean reward over
  the 3 repeats: goals 100049, 100026, 100023 (≤ 0.01; no video, only the first 10 goals are recorded); among the
  recorded ones, 100008 (reward 0) ends with the T pushed into the bottom-left corner.
- Conclusion: 25 demonstrations and 3 L4 hours turn a fixed-goal policy into a goal-conditioned one at the coarse level
  (score 0.18 → 0.52, 53% of the way to the model trained on all 247 demos with full compute), but not at the precision
  level (success 0 → 0.007 vs 0.253).

### EXP-03 F3 — How many demonstrations does the adaptation need?
- Date: 2026-10-14/15. Driver `results/exp02/driver_f3_f4.sh`: `bash scripts/run_exp02.sh train f3_k50 k50`,
  `... train f3_k10 k10`, then `eval` as F2. Subsets are nested, k10 ⊂ k25 (= F2) ⊂ k50
  (`results/exp02/subsets.json`, ISSUE-010); same step budget: 3034 / 3024 / 3144 steps of batch 16.
- Result (mean score; success): k10 0.421 ± 0.023, 0/150; k25 0.517 ± 0.037, 1/150; k50 0.593 ± 0.040, 2/150.
  Share of the lower-to-upper gap: 38% / 53% / 65%. Paired over goals: k25 − k10 = +0.096 [0.027, 0.164],
  k50 − k25 = +0.076 [0.005, 0.147], k50 − k10 = +0.172 [0.093, 0.251]. About +0.1 mean score per doubling of
  the data; success stays at 0–1% for every k (upper bound 25%). Figure: `results/exp02/figures/data_scaling.png`.
- Training: final-epoch action loss is lowest for k10 (0.025) and highest for k50 (0.041), the reverse of the
  simulator ranking: with a fixed step budget, fewer demonstrations are seen more often (k10: about 37 passes) and
  fitted better, but generalize worse to new goals. Block centroid at the end: 9.4 px (k10), 4.3 px (k25), 5.3 px
  (k50) from the episode goal (30 videos each, noisy). Curves: `figures/train_curves_all.png`; last frames:
  `figures/last_frames_data_scaling.png`. FVD from the training logs is not comparable between k50 and the others
  (ISSUE-011).
- Conclusion: more demonstrations help steadily (+0.17 from 10 to 50) but even 50 demonstrations leave a third of the
  gap; and no amount tested here makes the policy precise enough to cross the 95% coverage threshold. One training run
  per k.

### EXP-03 F4 — Does the video objective help the policy with 25 demonstrations?
- Date: 2026-10-14. Driver `results/exp02/driver_f3_f4.sh`:
  `bash scripts/run_exp02.sh train f4_k25_action_only 25 18 model.policy.selected_training_mode=policy_model`, then
  `eval` as F2. Everything else equal to F2: same 25 episodes, init, 3024 steps of batch 16, lr schedule, seed 42;
  same speed (3.13 s/step), so same compute. Every F4 step has an action loss, against about 3/5 of the F2 steps (the
  other F2 steps train video only): equal compute, not equal number of action updates.
- Result: mean score 0.454 ± 0.042 (repeats 0.427 / 0.432 / 0.502) vs F2 0.517 ± 0.037; success 2/150 vs 1/150.
  Paired over the 50 goals: F2 − F4 = +0.063, bootstrap 95% [0.003, 0.124], sign-flip permutation p = 0.047; F2 is
  better on 30/50 goals. F4 closes 43% of the lower-to-upper gap (F2 53%). Median max reward 0.529 (F4) vs 0.609 (F2).
  Block centroid at the end: 11.7 px from the episode goal (F2 4.3 px).
- Training: F4 reaches a lower action loss on its own training data (0.0258 vs 0.0294 in the last epoch; F2's value
  mixes the policy, inverse and joint modes, so the two are not strictly comparable) but does worse in the simulator.
  Rollout after epoch 0: 0.440 (F4) vs 0.465 (F2). FVD: F4's video head is not trained, mean 1604 vs 1449 (noisy, 4
  batches). Figures: `figures/train_curves_f2_f4.png`, `figures/last_frames_f1_f2_f4_upper.png`.
- Caveat: one training run per condition. The ± is over policy-sampling seeds of one checkpoint; the variance between
  training seeds is not measured, and the effect (0.06) is about the size of the spread between policy seeds.
- Conclusion: with 25 demonstrations and equal compute, keeping the video objective gives a small but measurable gain
  (+0.06 mean score, p ≈ 0.05) and a T placed closer to the goal, consistent with the paper's claim that the video
  objective helps the policy. Weak evidence: one training seed, borderline significance. On the F3 curve, action-only
  with 25 demonstrations (0.454) is not distinguishable from joint training with 10 (0.421; paired difference +0.032
  [−0.043, 0.106]).

### EXP-03 addendum — Position vs orientation error at the end of PushT-M episodes
- Date: 2026-10-16. `python scripts/pose_error.py` → `results/exp02/pose_error.json`. No GPU: fits a rasterized T template
  (pusht_env.py geometry) to the goal (green, union of 5 frames) and to the block (grey, last frame) of the 30 recorded
  videos per run (first 10 goals × 3 seeds, 96-px frames, bars 5.6 px thick). Successful episodes fit to 0–4°.
- Result (median position error; episodes ending within 2.8 px and 15° of the goal pose): not fine-tuned 21.3 px, 0/30;
  k10 8.8 px, 3/30; F4 9.4 px, 0/29; F2 (k25) 4.2 px, 2/30; k50 5.0 px, 4/30; authors' model 1.4 px, 12/30.
- Near the goal (≤ 2.8 px), about half of the endings are rotated by 100–140° (F2 4/8, k50 4/10, authors' model 5/17),
  all with reward ≈ 0.7: the T's three arms make a ~120° rotation a local optimum of coverage.
- Conclusion: this corrects the earlier "right place, wrong angle" reading. Fine-tuned models bring the block near the
  goal but are imprecise in both position (median about one bar width) and orientation; the 0.72 reward plateau seen in
  EXP-01 is the ~120° rotation.

### EXP-07 (W3) — UVA inference cost vs action denoising steps

Command (VM, `uva` env): `bash scripts/run_exp07.sh` (2026-10-15, about 35 min). The policy samples only actions (no
video) at test time, so the step count of the action diffusion head (`act_diff_testing_steps`, 100 in the checkpoint) is
the knob. `eval_sim.py` times every `predict_action` call with CUDA synchronization (`latency.json`). Each call returns an
action chunk, of which 8 actions are executed (`n_action_steps`).

| steps | mean score (50 seeds) | success | paired difference vs 100 steps, 95% bootstrap | s / call, batch 1 | s / call, batch 10 |
|---|---|---|---|---|---|
| 100 | 0.957 | 0.70 | – | 0.48 | 1.65 |
| 50 | 0.973 | 0.74 | +0.016 [−0.007, 0.059] | 0.30 | 1.46 |
| 20 | 0.972 | 0.68 | +0.015 [−0.010, 0.058] | 0.19 | 1.35 |
| 10 | 0.976 | 0.78 | +0.019 [−0.004, 0.061] | 0.16 | 1.32 |
| 5 | 0.995 | 0.70 | +0.038 [−0.003, 0.099] | 0.14 | 1.30 |

- The 100-step run gives exactly the EXP-01 seed-42 result (0.9567), so the timing wrapper does not change the policy.
- No measurable loss down to 5 steps on PushT; the action head can run 3.5× faster at batch 1 (0.48 → 0.14 s).
- At batch 10 the time per call barely changes (1.65 → 1.30 s): the shared encoder and MAR transformer dominate, the action
  head is a small part. Peak GPU memory 1.9 GiB at batch 1.
- Batch-1 latency is from one episode (19–38 calls), so it is a timing, not a score; the batch-1 scores are not used.
- Against LingBot-VA on the same L4 (EXP-11): UVA executes 8 actions per 0.14–0.48 s (about 17–59 actions/s) with 1.9 GiB;
  LingBot-VA generates 32 actions per 3.8–17.1 s (about 2–8 actions/s) with 17 GiB, and also generates the video. Tasks,
  images and action spaces differ (PushT 2-D vs RoboTwin dual-arm), so this is an order-of-magnitude comparison.

### EXP-11 — LingBot-VA image → video + action generation on one L4

Command (VM, venv `~/venvs/lingbot`, setup `scripts/exp11_setup.sh`): `bash results/exp11/driver.sh`, which runs
`scripts/run_exp11.sh i2av_c10_s{0,1,2} 10 {0,1,2}` and `scripts/run_exp11.sh i2av_c40_s0 40 0`; summary and figures:
`python scripts/summarize_exp11.py` → `results/exp11/summary.json`, `results/exp11/figures/`.

What the run does: `robotwin_i2av` config of the authors (25 video denoising steps with classifier-free guidance 5,
50 action denoising steps, 2 latent frames and 16 actions per frame per chunk). One start image from 3 cameras and the
prompt "Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack."
Every chunk is conditioned on all earlier chunks through the KV cache; no simulator feedback (open-loop imagination).

| | value | source |
|---|---|---|
| model load (transformer to GPU, T5 and VAE on CPU) | about 1 min | `run.log` |
| prompt encoding, T5 (11.4 GB) on 4 vCPUs | 268 ± 2 s | `prompt_encoding_cpu_s` |
| first chunk (includes VAE encoding of the start image on the CPU) | 39–40 s | `timing.json` |
| later chunks, chunks 1–9 | 17.1 s (16.4–18.1), of which video ≈ 6.3 s, actions ≈ 9.8 s | `timing.json`, tqdm rates |
| chunk 36 onwards (KV cache full, oldest entries evicted) | 23.6 s, flat | `i2av_c40_s0/timing.json` |
| VAE decoding on the GPU | 10.7 s for 77 frames, 41.5 s for 317 frames | `decode_s` |
| peak GPU memory | 16.9–17.0 GiB allocated, 18.0–18.1 GB nvidia-smi; same for 2, 10 and 40 chunks | `timing.json`, `vram.csv` |
| peak RAM / swap | 4.1–5.1 GB / 0.6–2.3 GB (16 GB machine) | `ram.csv` |

Findings:
- Fits on the L4 with offload; no OOM. GPU memory does not grow with the number of chunks because the KV cache is
  allocated up front for 36 chunks (`attn_window` 72 / 2) and the oldest entries are evicted after that.
- Time per chunk grows linearly with the history (+0.21 s per chunk, 16.3 s → 23.6 s) because attention covers every
  filled cache slot, then stays flat once the cache is full: the cost of the long memory that UVA does not have.
- At about 17 s per 32 actions the L4 produces about 2 actions per second without the authors' asynchronous execution;
  the CPU prompt encoding (268 s) is paid once per episode.
- Seeds: generated actions differ by 0.001 (mean absolute, normalized units) in chunk 0 and up to 0.08 in chunk 9 between
  seeds; the difference grows with the chunk index (errors compound autoregressively). Seeds 1 and 2 stay close
  (≤ 0.008), seed 0 diverges from chunk 6 on, visible in the videos (`figures/frames_i2av_c10_s*.png`). The 40-chunk run
  with seed 0 reproduces the 10-chunk run with seed 0 (mean absolute pixel difference 0.5 / 255 over the first 77 frames).
- Qualitative (`figures/frames_i2av_c40_s0.png`): in the first 10 chunks the arm reaches, grasps and lifts the mug toward
  the rack; after about chunk 12 the imagined high-camera view loses the rack and the arms while the mug stays near the
  right edge, and later frames are mostly the empty table with shadows. Not quantified; past the authors' default length
  of 10 chunks, open-loop imagination drifts (in closed loop the real observations refresh the cache).

Ablations (2026-10-15, seed 0, same initial noise as the reference; `results/exp11/driver_ablation.sh`, comparison
`python scripts/compare_exp11.py` → `results/exp11/ablations.json`, `figures/ablations.png`). T5 embeddings cached
(`--set prompt_embeds_cache_dir=...`), so only the first run of a prompt pays the 268 s. Yardstick: another seed of the
reference config differs by 0.015–0.016 in actions and has 16–17 dB PSNR.

| run | change | action difference vs reference | video PSNR vs reference | s / chunk (ref) | peak GiB |
|---|---|---|---|---|---|
| `steps_v10_a20` | 10 video / 20 action denoising steps (default 25 / 50) | 0.002 | 28.7 dB | **7.1** (17.1) | 16.9 |
| `steps_v5_a10` | 5 / 10 steps | 0.015 (≈ another seed) | 16.8 dB | **3.8** (17.1) | 16.9 |
| `cfg1` | no classifier-free guidance on video (guidance 5 → 1, batch 2 → 1) | 0.012 | 19.1 dB | 15.1 (17.1) | 15.2 |
| `prompt_left` | prompt "…place it on the left side of the table." | 0.028 | 15.3 dB | 17.1 (17.1) | 16.9 |
| `win8_c40` | KV window 4 chunks (`attn_window` 8, default 72 = 36 chunks), 40 chunks, vs `i2av_c40_s0` | 0.135 (0 in chunks 0–3, ≤ 0.005 up to chunk 10, 0.05–0.34 from chunk 13) | 19.3 dB | 16.6 flat (20.3, rising) | 15.4 |

- Denoising steps: 10 / 20 steps give the same video and actions as 25 / 50 (difference 7× below seed noise) at 2.4× the
  speed; 5 / 10 steps are 4.5× faster and stay coherent, but differ from the reference as much as another seed does.
  With 5 / 10 steps the L4 produces 32 actions in 3.8 s, about 8 actions per second.
- Classifier-free guidance doubles the video batch but costs only 12% of the time per chunk (action denoising, without
  guidance, is the larger part) and 1.7 GiB; without it the sample stays coherent.
- Prompt: with the same start image the imagined video still takes the mug to the rack (frames in
  `figures/frames_prompt_left.png`); the actions differ from the reference by 1.7× seed noise. On this RoboTwin scene the
  post-trained model follows the visual context more than the new instruction (one prompt, one seed: an observation, not
  a measured rate).
- Memory window: with 4 chunks of memory the time per chunk stays at 16.6 s instead of rising to 23.6 s, and peak memory
  drops by 1.6 GiB. The first 4 chunks are identical to the full-window run (the window is not yet full); from about
  chunk 13 the two runs diverge far beyond seed noise, and from about chunk 15 the short-window video shows broken arm
  geometry (overlapping gripper fragments, `figures/frames_win8_c40.png`), which the full-window run does not. Short memory
  is cheaper but loses consistency over long horizons; this is the trade-off behind the UVA / LingBot-VA comparison
  (Phase 6, A3).

### EXP-05 — Released checkpoint, LIBERO-10
- Date / commit: 2026-10-14 / `aad9f1d` + uncommitted / `uva` `fe3dec8` + changes listed in the README
- Question: does `libero10.ckpt` reproduce the paper's LIBERO-10 number on one L4?
- Exact command: `bash scripts/run_exp05.sh 20 5 42`, then `python scripts/summarize_libero.py results/exp05`
- Seeds, number of rollouts: 10 tasks × 20 episodes (seeds 100000–100019), policy seed 42
- Result: 0.865 mean success, 95% Wilson [0.811, 0.906], reported 0.90 → reproduced; per task 0.50–1.00
  (`results/exp05/summary.json`, `notes.md`)
- Peak GPU memory / run time: 10.3 GB; about 55 min; first attempt hit the RAM limit (ISSUE-007)
- Qualitative: the two weakest tasks move two objects of the same kind to one place (basket, stove)
- Conclusion: reproduced within the interval; the hard cases are the multi-object, long-horizon tasks.

### EXP-01 — Released checkpoints, PushT and PushT-M
- Date / commit: 2026-10-14 / `aad9f1d` + uncommitted / `uva` `fe3dec8` + `eval_sim.py` options
- Question: do the released checkpoints reproduce the paper on one L4?
- Exact command: `bash scripts/run_exp01.sh 10`, then `python scripts/aggregate_seeds.py results/exp01`
  (each run: `python eval_sim.py -c checkpoints/<task>.ckpt -o results/exp01/<task>/seed<k> --seed <k> --n_test 50
  --n_test_vis 10 --n_train 0 --n_envs 10`)
- Seeds, number of rollouts: test seeds 100000–100049; policy seeds 42, 0, 1; 150 episodes per checkpoint
- Result: `results/exp01/summary.json` (table above), per run `metrics.json`
- Peak GPU memory / run time: 9534 MiB; about 5 min 50 s per 50 seeds; no OOM (GPU or RAM)
- Qualitative: PushT failures are near-misses (92–95% coverage); PushT-M failures often end with the T at the goal
  position but wrongly rotated (reward about 0.72) — `results/exp01/notes.md`, `figures/failure_last_frames.png`
- Conclusion: PushT reproduces the paper (mean score 0.974 vs 0.98). PushT-M is 0.09 lower, but the upstream evaluation
  draws one goal per chunk of environments (ISSUE-005), so the paper's number is most likely from a single goal. With 50
  distinct goals the score is 0.813 ± 0.030; single-goal scores range 0.68–0.92, so 0.88 is within reach of one goal.

<!--
Template:

### EXP-NN — <name>
- Date / commit (this repository + submodule): YYYY-MM-DD / <sha> / <sha>
- Question:
- Exact command:
  ```bash
  ```
- Overrides of the default config:
- Seeds, number of rollouts:
- Result (number + path of the json/csv written by the code):
- Peak GPU memory / run time:
- Qualitative (successful and failed rollouts, with video paths):
- Conclusion in one or two sentences:
-->
