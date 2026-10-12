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

## Details

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
