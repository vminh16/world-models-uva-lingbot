# EXP-01 — Released UVA checkpoints on PushT and PushT-M

Date 2026-10-14. Machine: GCP `g2-standard-4`, 1× L4. Repository `aad9f1d` + uncommitted changes, submodule `uva`
`fe3dec8` (`hw-small-scale`) + the `eval_sim.py` options (README, "Changes from upstream").

Checkpoints (sha256, `uva/checkpoints/SHA256SUMS`):
- `pusht.ckpt` `114dd771f946311fc72208439937009ad8a8f027a1e843ec42625d1cfe38bcbc`
- `pusht_multitask.ckpt` `bfba3cc9d4633e8426b5c06cbfa1c6dc28447e703efef55a848edbd1f09bcc33`

## Protocol

50 test seeds 100000–100049, no train seeds, 300 steps, `--n_envs 10` (5 chunks of 10). Repeated with policy-sampling
seeds 42, 0, 1. The seed-42 runs used `eval_sim.py` before the ISSUE-004 fix; for seed 42 the fix changes nothing.

```bash
bash scripts/run_exp01.sh 10          # all runs; skips runs that have metrics.json
python scripts/aggregate_seeds.py results/exp01
```

## Results (`summary.json`, written by the scripts)

| Task | Success rate, mean ± std over 3 repeats | Pooled success, 150 episodes [95% Wilson] | Mean score ± std | Reported |
|---|---|---|---|---|
| PushT (`pusht.ckpt`) | 0.733 ± 0.058 (0.70 / 0.80 / 0.70) | 0.733 [0.657, 0.798] | 0.974 ± 0.016 | 0.98 (Table II) |
| PushT-M (`pusht_multitask.ckpt`), upstream goals | 0.220 ± 0.035 (0.18 / 0.24 / 0.24) | 0.220 [0.161, 0.293] | 0.787 ± 0.008 | 0.88 (Table II) |
| PushT-M, `--seeded_goal True` (50 distinct goals) | 0.253 ± 0.012 (0.24 / 0.26 / 0.26) | 0.253 [0.190, 0.328] | 0.813 ± 0.030 | – |

Success = coverage above 95% (`sim_max_reward == 1`). Mean score = mean over seeds of the episode's maximum reward,
`clip(coverage / 0.95, 0, 1)`; this is the metric of Diffusion Policy's PushT tables, so compare it with the paper first.

Cost per 50-seed run: 5 min 47 s – 5 min 55 s wall time, peak GPU memory 9534 MiB of 23034, minimum available RAM
about 9.7 GB of 16 GB. **No out-of-memory**: all exit codes 0, no "out of memory" / "Killed" in the logs, no OOM line in
the kernel log.

Reported numbers: paper Table II, "Policy Learning Results in Simulation", best checkpoint, 50 rollouts. The paper calls the
PushT number a success rate, but the code logs `test/mean_score` (mean of the episode max reward), and the DP-C baseline in
the same column (0.91) is Diffusion Policy's coverage number, so the comparable column here is the mean score.
PushT-M is reported as "average reward".

- PushT: 0.974 ± 0.016 vs 0.98: reproduced.
- PushT-M: 0.787 ± 0.008 vs 0.88: lower by 0.09, but the two numbers are not measured on the same goals (ISSUE-005):
  our runs average over 5 goals per repeat (15 over 3 repeats), the default protocol over a single goal. The paper's value
  also comes from the best checkpoint selected on these rollouts, which biases it upwards.
- PushT-M with one goal per test seed (`seeded_goal`, ISSUE-005 fix): 0.813 ± 0.030. This is the number to use as the
  upper bound in Phase 3.
- How much a single-goal evaluation varies: in the upstream-protocol runs each chunk of 10 envs shares one goal, so the
  15 chunks are 15 small single-goal evaluations. Their mean scores range 0.678–0.922 (mean 0.786, std 0.064); 1 of 15
  is ≥ 0.88. A paper-style evaluation (one goal, best checkpoint chosen on the same rollouts) can therefore reach 0.88
  without the policy being better than about 0.8 on average. This is consistent with the gap; it does not prove it.

## Observations

- Policy sampling noise matters: the same 50 seeds give 0.70–0.80 success on PushT. A single 50-seed evaluation
  cannot separate two policies that differ by less than about 0.1.
- PushT: 21/50 seeds succeed in all 3 repeats, only 1 (100017) in none. Failures are near-misses: rewards 0.97–0.999,
  i.e. 92–95% coverage, just under the threshold. The two clear failures are seeds 100035 (mean reward 0.33) and
  100041 (0.67).
- PushT-M: only 1/50 seeds succeeds in all repeats, 28/50 in none. Many failure rewards sit at about 0.72.
  In the last frames looked at so far (`figures/failure_last_frames.png`, panels 1–2) the T is at the goal position but
  at the wrong rotation; panel 3 (seed 100008, reward 0.28) shows the T pushed away from the goal; panel 4 is a PushT
  near-miss. This is a hypothesis from 4 frames; Phase 6 counts failure types over all videos.
- Earlier 4-seed run (2026-10-13, all 4 seeds in one batch) gave PushT-M mean score 0.889 on seeds 100000–100003; the
  same seeds here have lower rewards. Expected from sampling noise on 4 seeds, but checking whether `n_envs` changes the
  mean (for example `--n_envs 25`) is a cheap follow-up.

## Files

- `<task>/seed<k>/`: `command.txt` (commits, checkpoint hash, command), `env_runner.yaml`, `eval_log_*.json`,
  `metrics.json`, `run.log`, `vram.csv`, `ram.csv`, `exit_code`, `media/` (videos of seeds 100000–100009)
- `rollouts/<task>/env<seed>_seed<k>_<success|fail>_r<reward>.mp4`: the same videos, named by outcome
- `driver.log`; `driver_run1_seed_bug.log` (first attempt, ISSUE-004)
