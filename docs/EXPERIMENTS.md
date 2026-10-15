# Experiments

Every train or evaluation run that produces a number has a row here; the report quotes numbers only from this file and the
`metrics.json` files under `results/`. IDs: `EXP-0x` UVA, `EXP-1x` LingBot-VA, `EXP-2x` other.
"(ours)" = measured here, "(reported)" = from the paper or the official README.

## Plan

| ID | Priority | Model | Question | Setup | Metric | Output |
|---|---|---|---|---|---|---|
| EXP-01 | P0 | UVA | Do the released checkpoints reproduce the paper? | `pusht.ckpt` on PushT, `pusht_multitask.ckpt` on PushT-M, fixed seeds | success rate, mean target coverage | `results/exp01/` |
| EXP-02 | P0 | UVA | Can a released checkpoint adapt to a PushT variant it has never seen, with little data and compute? | new goal poses (or another controlled change) in the PushT simulator; demonstrations generated in the simulator; fine-tune a few GPU hours; evaluate before and after | success rate before / after | `results/exp02/` |
| EXP-03 | P1 | UVA | How much data does adaptation need? | EXP-02 with 10 / 25 / 50 demonstrations | success rate vs number of demonstrations | `results/exp03/` |
| EXP-04 | P1 | UVA | Does the imagined future match the simulator, and does the mismatch explain failures? | compare UVA's predicted frames with the simulator's frames along rollouts, before and after fine-tuning | per-frame error (for example PSNR), linked to success / failure | `results/exp04/` |
| EXP-05 | P2 | UVA | Same benchmark as LingBot-VA | `libero10.ckpt` on LIBERO-10, `MUJOCO_GL=egl` | success rate | `results/exp05/` |
| EXP-11 | P0 | LingBot-VA | What does it imagine, how fast, with how much memory on one L4? | `robotwin_i2av`, offload on | latency per chunk, peak GPU memory, generated videos | `results/exp11/` |
| EXP-12 | P2 | LingBot-VA | Same benchmark as UVA, if it fits 24 GB | `lingbot-va-posttrain-libero-long` on a subset of LIBERO-Long tasks | success rate | `results/exp12/` |

Reported reference numbers (official LingBot-VA README): RoboTwin 2.0 92.9% (easy) / 91.6% (hard) averaged over 50 tasks,
LIBERO 98.5 average. UVA reference numbers: to be copied from the paper's tables when EXP-01 is written up.

## Results

| ID | Checkpoint | Task | Change from upstream | Metric | Result (ours) | Reported | Run time | Peak GPU memory | Output |
|---|---|---|---|---|---|---|---|---|---|
| EXP-05 | `libero10.ckpt` | LIBERO-10, 10 tasks × 20 episodes, policy seed 42 | `eval_sim.py` options, LIBERO venv (ISSUE-006) | mean success over tasks | 0.865 [0.811, 0.906] | 0.90 | about 55 min | 10.3 GB | `results/exp05/` |
| EXP-01 | `pusht.ckpt` | PushT, 50 seeds × 3 policy seeds | `eval_sim.py` overrides only | success / mean score | 0.733 ± 0.058 / 0.974 ± 0.016 | – / 0.98 | 5 min 50 s per 50 seeds | 9.3 GiB | `results/exp01/` |
| EXP-01 | `pusht_multitask.ckpt` | PushT-M, `--seeded_goal True`, 50 goals × 3 policy seeds | `eval_sim.py` overrides + `seeded_goal` | success / mean score | 0.253 ± 0.012 / 0.813 ± 0.030 | – | 6 min per 50 seeds | 9.3 GiB | `results/exp01/` |
| EXP-01 | `pusht_multitask.ckpt` | PushT-M, 50 seeds × 3 policy seeds | `eval_sim.py` overrides only | success / mean score | 0.220 ± 0.035 / 0.787 ± 0.008 | – / 0.88 (1 goal, ISSUE-005) | 5 min 55 s per 50 seeds | 9.3 GiB | `results/exp01/` |

## Details

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
