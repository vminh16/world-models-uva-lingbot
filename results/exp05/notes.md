# EXP-05 — UVA `libero10.ckpt` on LIBERO-10 (LIBERO-Long)

Date 2026-10-14. Machine: GCP `g2-standard-4` (4 vCPU, 16 GB RAM), 1× L4. Simulator setup: `docs/ENVIRONMENT.md` 2b,
problems: ISSUE-006 (setup, 8 steps) and ISSUE-007 (out of memory).

Checkpoint `libero10.ckpt` sha256 `4554b9bb2f4f3d44a7e51d5c7bd42a7740451874ac6cf1fea91e088a949f0dbf`.

## Protocol

10 tasks × 20 test episodes (seeds 100000–100019, random initial object placement through `env.seed`), max 500 steps,
policy seed 42, 5 spawned simulator workers, one process per task. The released config evaluates only 3 episodes per
task; the paper says 50 per task (main text) or 10 per task (appendix).

```bash
bash scripts/run_exp05.sh 20 5 42                  # on the VM, LIBERO venv
python scripts/summarize_libero.py results/exp05   # -> summary.json
```

## Results (`summary.json`)

| | Ours (200 episodes) | Reported (Table II) |
|---|---|---|
| Mean success over 10 tasks | **0.865**, 95% Wilson interval [0.811, 0.906] | 0.90 |

Reproduced: the reported value lies inside our interval.

| Task | Success |
|---|---|
| LIVING_ROOM_SCENE2 alphabet soup + tomato sauce in basket | 0.50 (10/20) |
| KITCHEN_SCENE8 both moka pots on stove | 0.70 (14/20) |
| LIVING_ROOM_SCENE1 alphabet soup + cream cheese in basket | 0.85 (17/20) |
| LIVING_ROOM_SCENE2 cream cheese + butter in basket | 0.85 (17/20) |
| LIVING_ROOM_SCENE6 white mug on plate + pudding right of plate | 0.90 (18/20) |
| KITCHEN_SCENE4 black bowl in bottom drawer + close | 0.95 (19/20) |
| KITCHEN_SCENE6 mug in microwave + close | 0.95 (19/20) |
| LIVING_ROOM_SCENE5 two mugs on two plates | 0.95 (19/20) |
| KITCHEN_SCENE3 stove on + moka pot on it | 1.00 (20/20) |
| STUDY_SCENE1 book in caddy back compartment | 1.00 (20/20) |

Cost: 2 min 53 s – 7 min 33 s per task, about 55 min in total; peak GPU memory 10.3 GB (LIVING_ROOM_SCENE5); minimum available RAM 1.2 GB
(after the ISSUE-007 fix).

## Observations

- The two weakest tasks both require moving **two objects of the same kind** to one place (two items into the basket,
  two moka pots onto the stove). Tasks with one object, or two sub-steps on different objects (open/close, turn on +
  place), are at 0.95–1.00.
- In the failed soup + tomato sauce episode with a video (seed 100000), the episode ends at the 500-step limit with the
  gripper still at the objects (`figures/soup_tomato_last_frames_success_fail.png`, right; left is the successful seed
  100001). One video per outcome only: to be checked on more episodes before it goes into the report as a failure mode.
- 4 tasks were first killed by the out-of-memory killer (ISSUE-007, logs in `<task>/seed42_oom_nenvs5/`) and rerun with
  the same settings after `eval_sim.py` freed the unused checkpoint data; the other 6 tasks ran before that change, which
  does not touch the computation.
- Only 2 videos per task (`--n_test_vis 2`), one policy seed: the per-task numbers have wide intervals (20 episodes).
