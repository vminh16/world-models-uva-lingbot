# World Models on One GPU: UVA and LingBot-VA

Part 2 of a robot-learning assignment: study, reproduce and extend two video-action world models on a single
NVIDIA L4 (24 GB). Part 1 (ROSOrin Pro simulation, SLAM and Nav2) is in
[vminh16/rosorin-autonomy-sim](https://github.com/vminh16/rosorin-autonomy-sim).

| Model | Paper | What is done here |
|---|---|---|
| **UVA**, Unified Video Action model | RSS 2025 | Reproduce the released simulation results, then fine-tune on new task variants generated in the simulator |
| **LingBot-VA**, the official code of CauVA (causal video-action world model) | RSS 2026 | Inference only: the model is too large to train on one GPU (post-training uses 8 GPUs) |
| Motus | – | Not run (does not fit 24 GB); compared in the report with reported numbers only |

## Status

| Step | Status |
|---|---|
| GCP L4 machine and environments | done: 1× L4, `uva` env, LIBERO venv (`docs/ENVIRONMENT.md`) |
| UVA: evaluate released checkpoints (EXP-01, EXP-05) | done: PushT, PushT-M (50 seeds × 3, and with 50 distinct goals), LIBERO-10 (10 tasks × 20) |
| UVA: fine-tune `pusht.ckpt` to random goals (EXP-02, EXP-03) | done: 0 / 10 / 25 / 50 demonstrations and action-only ablation, 3k steps each on one L4 (`results/exp02/`) |
| UVA: inference cost vs denoising steps (EXP-07) | done: PushT 50 seeds at 100 / 50 / 20 / 10 / 5 steps, latency at batch 1 and 10 (`results/exp07/`) |
| LingBot-VA: inference (EXP-11) | done: i2av on one L4 with offload, 3 seeds × 10 chunks and 40 chunks, latency and memory; ablations of denoising steps, CFG, prompt, KV window (`results/exp11/`) |
| Report | Part 2 of the technical report: `report/PART2_en.md` (Vietnamese: `report/PART2_vi.md`); full results `report/REPORT_vi.md`; paper analysis `docs/PAPER_ANALYSIS_vi.md`; demo video `results/rollouts/part2_demo_small.mp4` |

## What is new compared with the papers

Reproduction only checks the authors' numbers. The extension asks a different question from theirs:

- **The authors** train UVA (starting from a pretrained VAE and MAR image model) on 4 to 8 GPUs, about 2 days for the video
  stage and 2 more for the joint stage, on fixed datasets: PushT (fixed goal) and PushT-M (their own PushT variant with varied
  goal positions, 247 demonstrations).
- **Here** the released fixed-goal checkpoint (`pusht.ckpt`) is **adapted** to random goals (PushT-M), which it has never
  seen, with 10 to 50 of the released PushT-M demonstrations and about 3 hours on one L4 per run. The questions are: how
  much data does adaptation need (EXP-02, F3), does the video objective help the policy in this low-data regime (EXP-03,
  F4), and does the model's *imagined* future, compared frame by frame with the simulator, explain where the policy fails
  (EXP-04)?
- Both models have a checkpoint for the same simulated benchmark (UVA `libero10.ckpt`, LingBot-VA
  `lingbot-va-posttrain-libero-long`), which allows a direct comparison in one simulator if memory allows (EXP-05, EXP-12).

## Repository layout

```
uva/            git submodule: fork of ShuangLI59/unified_video_action, branch hw-small-scale
lingbot-va/     git submodule: fork of Robbyant/lingbot-va, branch hw-small-scale
scripts/        task variants, data generation, figure generation (make_figures.py)
results/        one folder per experiment: metrics.json, rollout videos, figures, logs (see results/README.md)
docs/           ENVIRONMENT.md (GCP L4 setup), EXPERIMENTS.md (every run), DEBUG_LOG.md (every problem)
report/         Part 2 of the technical report (PART2_en.md, PART2_vi.md), full results (REPORT_vi.md), figures/
```

## Getting started

```bash
git clone --recursive https://github.com/vminh16/world-models-uva-lingbot.git
cd world-models-uva-lingbot
```

Machine, drivers and the two conda environments (`uva`, `lingbot`): [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).
The commands that have been run and verified are added to the "Reproduce" section below as each experiment is done.

## Reproduce

Each experiment in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) adds its exact command here.

**EXP-01** (2026-10-14, one L4, about 35 min for 6 runs), from the repository root in the `uva` env:

```bash
bash scripts/run_exp01.sh 10                       # PushT and PushT-M, 50 test seeds × policy seeds 42, 0, 1
python scripts/aggregate_seeds.py results/exp01    # -> results/exp01/summary.json
TAG=_seeded_goal EXTRA="--seeded_goal True" bash scripts/run_exp01.sh 10 pusht_multitask   # PushT-M, 50 distinct goals
```

**EXP-07** (2026-10-15, about 35 min), on the VM in the `uva` env:

```bash
bash scripts/run_exp07.sh                          # PushT 50 seeds and batch-1 latency, 100 / 50 / 20 / 10 / 5 action steps
```

**EXP-11** (2026-10-15, about 50 min + 40 min of ablations), on the VM with the venv from `scripts/exp11_setup.sh`:

```bash
bash results/exp11/driver.sh                       # i2av, 3 seeds x 10 chunks, then 40 chunks (scripts/run_exp11.sh)
python scripts/summarize_exp11.py                  # -> results/exp11/summary.json, results/exp11/figures/
bash results/exp11/driver_ablation.sh              # steps, CFG, KV window, prompt (seed 0)
python scripts/compare_exp11.py                    # -> results/exp11/ablations.json, results/exp11/figures/ablations.png
```

Example videos of the report (labelled, side by side, under `results/**/rollouts/*_small.mp4`):

```bash
python scripts/make_videos.py                      # -> results/rollouts/part2_demo_small.mp4 and per-experiment clips
PYTHONPATH=scripts python scripts/plot_report_figure.py -o report/figures/fig2_part2.png   # report Figure 2
python scripts/plot_report_qualitative.py -o report/figures/fig3_part2.png                 # report Figure 3
python scripts/pose_error.py                       # final position / angle error of PushT-M runs -> results/exp02/pose_error.json
```

**EXP-05** (2026-10-14, about 55 min), on the VM with the LIBERO venv (`docs/ENVIRONMENT.md` 2b):

```bash
bash scripts/run_exp05.sh 20 5 42                  # LIBERO-10, 10 tasks × 20 episodes, one process per task
python scripts/summarize_libero.py results/exp05   # -> results/exp05/summary.json
```

## Evidence

Every number in the report has an experiment ID and a file under `results/`. Each experiment folder keeps: the exact command,
the Hydra config, the commit of the repository and of the submodule, `metrics.json` written by the code (not typed by hand),
rollout videos from the simulator, a GPU memory log (`nvidia-smi`), and figures regenerated by one command
(`python scripts/make_figures.py`). Numbers from the papers are marked "(reported)".

## Changes from upstream

| Repository | File | Change | Why |
|---|---|---|---|
| uva | `env_runner/libero_image_runner.py`, `eval_sim.py` | option `mp_context` (default `None` = upstream fork): start method of the LIBERO simulator workers | with `MUJOCO_GL=egl`, forked workers fail in `eglCreateContext` (`EGL_BAD_ALLOC`); `spawn` works (ISSUE-006) |
| uva | `eval_sim.py` | checkpoint loaded with `map_location="cpu"`; env runners created before the policy is moved to the GPU; option `--debug` (LIBERO: first task only) | forked LIBERO workers cannot create an EGL context once CUDA is initialized (ISSUE-006) |
| uva | `eval_sim.py` | option `--seed` (policy sampling seed, default `cfg.training.seed`); options `--n_test`, `--n_test_vis`, `--n_train`, `--n_train_vis`, `--n_envs`, `--test_start_seed`, `--fix_goal` override `cfg.task.env_runner` (default: keep the checkpoint's value); the final runner config is saved to `<output_dir>/env_runner.yaml` | run the paper's 50 test seeds in chunks on a 16 GB-RAM machine without editing the checkpoint files |
| uva | `env/pusht/pusht_env.py`, `env/pusht/pusht_image_env.py`, `env_runner/pusht_image_runner.py`, `eval_sim.py` | option `seeded_goal` (default `False` = upstream): with `fix_goal=False`, the goal is drawn from `RandomState([env_seed, 1])` instead of the global `np.random` | upstream gives every env of a chunk the same goal (ISSUE-005) |
| uva | `workspace/train_unified_video_action_workspace.py` | scalar entries of every step and epoch log (train / video / action loss, lr, FVD, rollout scores) also appended to `<output_dir>/logs.jsonl` | training curves without reading wandb's offline files (EXP-02) |
| uva | `workspace/train_unified_video_action_workspace.py` | FVD on the validation batches runs only when `epoch % training.val_every == 0` (default 1 = upstream, every epoch) | the k = 10 run has 37 short epochs; FVD every epoch would add about 40 min (EXP-02 F3) |
| uva | `dataset/pusht_image_dataset.py` | option `train_episodes` (default `None` = upstream `max_train_episodes` sampling): explicit training episode indices | nested subsets k10 ⊂ k25 ⊂ k50 for the data-scaling runs; upstream sampling gives disjoint sets (EXP-02 F3) |
| uva | `eval_sim.py` | option `--act_diff_testing_steps` (default: checkpoint value); every `predict_action` call timed with CUDA synchronization, written to `<output_dir>/latency.json` with the peak GPU memory | inference cost vs denoising steps (EXP-07) |
| lingbot-va | `wan_va/modules/model.py` | `flash_attn` import optional (fallback `None`; `attn_mode="flashattn"` raises a clear error without it) | `attn_mode="torch"` could not start without flash-attn, which takes hours to build on 4 vCPUs (ISSUE-013) |
| lingbot-va | `wan_va/wan_va_server.py` | options `--model_path`, `--enable_offload`, `--num_chunks`, `--seed`, `--set KEY=VALUE` (config overrides; key `prompt_embeds_cache_dir` reuses T5 prompt embeddings across runs) (default: config values, unseeded, no cache = upstream); i2av mode writes `timing.json` (seconds per chunk with CUDA sync, VAE decode time, peak allocated GPU memory) and `actions.npy` next to `demo.mp4` | run EXP-11 without editing the config files, and measure latency and memory (EXP-11) |

## Acknowledgements

UVA: [ShuangLI59/unified_video_action](https://github.com/ShuangLI59/unified_video_action).
LingBot-VA: [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va).
PushT and ToolHang come from [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), LIBERO from
[LIBERO](https://libero-project.github.io/). Neither upstream repository declares a license; the forks are kept unchanged on
`main` and all work is on the `hw-small-scale` branch, for study and reproduction only.
