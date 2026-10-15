# Debug Log

Source for the "problems and solutions" part of the report. Newest entries first. Each entry records the command, the error,
the cause (verified, or marked as a hypothesis) and the fix.

## ISSUE-008 — Peak GPU memory printed by the run scripts was wrong when values had 4 and 5 digits
- Date: 2026-10-14
- Component: our scripts (`scripts/run_exp01.sh`, `scripts/run_exp05.sh`)
- Status: FIXED
- Cause: `awk '{... if ($2>m) m=$2}'` compared strings ("4390" > "21560"). Fix: numeric comparison (`$2+0>m+0`).
  The minimum-available-RAM line had the same bug (`$3<m`); recomputed from `ram.csv`: all printed values were correct.
- Effect: recomputed from the saved `vram.csv` files: EXP-01 unchanged (9534 MiB everywhere); EXP-05 one task was
  10304 MiB, not 7078, so the EXP-05 peak is 10.3 GB (notes and EXPERIMENTS.md corrected). `aggregate_seeds.py` parses
  the numbers in Python and was correct.
- In report: no

## ISSUE-007 — LIBERO-10 evaluation killed by the out-of-memory killer
- Date: 2026-10-14
- Component: UVA / Infra
- Status: FIXED
- Command: `bash scripts/run_exp05.sh 20 5 42` (one process per task, 5 spawned simulator workers)
- Symptom: 6/10 tasks finished; the 4 LIVING_ROOM_SCENE2/5/6 tasks exited with code 137, available RAM reached 0–85 MB,
  28 OOM lines in the kernel log. GPU memory was not the limit (at most 9.1 GB of 23 GB).
- Cause (verified with `ps`): (1) since ISSUE-006 (7) the checkpoint is loaded on the CPU, so the parent process kept
  the 1.8 GB payload, the non-EMA model and the optimizer state in RAM (7.3 GB RSS); (2) each `spawn` worker re-imports
  torch/CUDA and the simulator, about 2 GB each; 16 GB RAM in total. Living-room scenes need more memory than kitchen ones.
- Fix: `eval_sim.py` deletes the payload, `workspace.model` and `workspace.optimizer` after loading (only the EMA policy
  is used): parent 7.3 → 3.9 GB. The 4 failed tasks are rerun with the same `n_envs=5`; logs of the killed runs kept in
  `results/exp05/<task>/seed42_oom_nenvs5/`. The rerun finished without OOM (minimum available RAM 1.2 GB).
- In report: yes (resource limits of one 16 GB machine)

## ISSUE-006 — Setting up the LIBERO simulator for UVA
- Date: 2026-10-14
- Component: Infra / UVA / Simulator
- Status: FIXED
- Context: the upstream `conda_environment.yml` has no robosuite, robomimic or LIBERO, and
  `env_runner/libero_image_runner.py:36` imports LIBERO from the hard-coded path `../LIBERO`.
- Problems, in order:
  1. `conda create --clone uva -n uva-libero` failed: `LookupError("didn't find info-x264-...-h7cc23a3_3 component in
     .../pkgs/x264-...conda")` (incomplete package cache). The script went on, `conda activate` failed silently and `pip`
     installed into the system Python 3.10 user site (`~/.local`). The `uva` env (Python 3.9) was not touched; `~/.local`
     had been created by that script and was deleted.
  2. In a venv, `pip` tried to build the newest `mujoco` from source (no Python 3.9 wheel): `MUJOCO_PATH environment
     variable is not set`. Fix: pin `mujoco==3.1.6`.
  3. `import mujoco` with `MUJOCO_GL=egl`: `Cannot initialize a EGL device display`. The GCP image has the compute-only
     NVIDIA driver (`libnvidia-compute-580-server`), no EGL/GL libraries. Fix: `sudo apt-get install
     libnvidia-gl-580-server libegl1 libgl1 libglvnd0` (same version 580.178.04 as the running driver; a dry run showed
     only new userspace libraries, no removal, no kernel module change).
  4. `robomimic==0.2.0` (and 0.3.0, the newest on PyPI) imports `mujoco_py` at module level
     (`robomimic/envs/env_robosuite.py:10`); `mujoco_py` does not exist for robosuite 1.4 / `mujoco` 3. The only use is
     `mujoco_py.builder.MujocoException` (line 378). Fix: robomimic 0.3.0 plus a two-file stand-in package `mujoco_py`
     in the venv that only defines `builder.MujocoException`; simulation runs through robosuite 1.4 and `mujoco` 3.1.6.
  5. `ModuleNotFoundError: No module named 'pytorch3d'` (`model/common/rotation_transformer.py:2`, used to convert the
     absolute LIBERO actions to the 6D rotation representation). No prebuilt wheel for py39 / cu121 / torch 2.1.0 on the
     official index. Only `pytorch3d.transforms` is used, and `transforms/` + `common/` are pure PyTorch (no `_C`
     import). Fix: copy `pytorch3d/__init__.py`, `common/`, `transforms/` of the official v0.7.5 tag into the venv
     (axis-angle → matrix → axis-angle round trip checked: error 0.0).
  6. Simulator worker died: `ModuleNotFoundError: No module named 'egl_probe'` (robomimic picks the render GPU with it).
     The earlier install had failed silently in the script: `egl_probe` 1.0.2 declares `cmake_minimum_required` below
     3.5, which CMake 4.x rejects ("Compatibility with CMake < 3.5 has been removed"). Fix: `pip install "cmake<4"`
     (3.31.10) in the venv, then `pip install --no-build-isolation egl_probe`; it finds GPU 0.
  7. Worker crash in `eglCreateContext`: `EGLError(err = EGL_BAD_ALLOC)`. `eval_sim.py` loaded the checkpoint with
     `torch.load` without `map_location` (straight to CUDA) and moved the policy to the GPU before
     `load_env_runner`, which forks the simulator workers; a child forked from a process with an initialized CUDA
     context cannot create its EGL context. (The runner already avoids OpenGL in the parent with a non-rendering dummy
     env; Diffusion Policy creates env runners before moving the model to the GPU.) Fix in `eval_sim.py`: load the
     checkpoint with `map_location="cpu"`, create the env runners, then move the policy to the GPU; the script prints
     `cuda initialized before env runners: False`. The moved calls (`.to`, `.eval`) draw no random numbers; regression
     check: PushT-M upstream protocol, seed 42, rerun after the change (`results/exp01/_regression_reorder/`): identical.
  8. Still `EGL_BAD_ALLOC` after (7), although `cuda initialized before env runners: False` and a fresh process can
     create a robosuite `EGLGLContext`. Cause (verified with `/tmp/egl_fork_test.py`: parent imports robosuite with
     `MUJOCO_GL=egl`, child creates an `EGLGLContext`): `fork` → `EGL_BAD_ALLOC`, `spawn` → OK, `forkserver` → OK.
     `import mujoco` with `MUJOCO_GL=egl` opens an EGL display at import time in the parent, and forked children inherit
     that state. Fix: `LiberoImageRunner(mp_context=...)` passed to `AsyncVectorEnv(context=...)`, default `None`
     (upstream behaviour); `eval_sim.py --mp_context spawn` on this machine.
  Side note: the regression check of (7) passed: PushT-M upstream protocol, seed 42, identical rewards on all 50 seeds
  (max difference 2.2e-16).
- Final setup: venv `~/venvs/uva-libero` created from the `uva` Python with `--system-site-packages` (shares torch 2.1.0,
  numpy 1.26.4; `uva` unchanged), plus robosuite 1.4.0, bddl 1.0.1, robomimic 0.3.0 (`--no-deps`) with the `mujoco_py` stand-in, mujoco 3.1.6, the pure-Python part of pytorch3d v0.7.5,
  egl_probe, LIBERO (`pip install -e`, `--no-deps`) cloned at `~/world-models-uva-lingbot/LIBERO`,
  `~/.libero/config.yaml` written beforehand. Script: `uva/data/_downloads/setup_libero_venv.sh` on the VM.
- Time spent: about 1 h
- In report: yes (undocumented dependencies of the official code)

## ISSUE-005 — PushT-M evaluation: all environments of a chunk get the same goal
- Date: 2026-10-14
- Component: UVA (upstream evaluation code)
- Status: FIXED (option; upstream behaviour kept as default)
- Symptom: in the first frames of one PushT-M run, the green goal T is at the same pose for different test seeds
  (`results/exp01/figures/pushtm_seed42_first_frames.png`).
- Cause (verified): with `fix_goal=False`, `PushTEnv._setup` (`unified_video_action/env/pusht/pusht_env.py:348`) samples
  the goal from the global `np.random`, not from the per-seed `RandomState(seed)` used for the initial state (line 106).
  `AsyncVectorEnv` forks its workers, so every worker starts with the same global random state and draws the same goal at
  each reset. Measured with `scripts/check_pushtm_goals.py` (runner setup only, no model): 50 test seeds give 5 distinct
  goals with `n_envs=10`, 10 with `n_envs=5`, and the goal of a test seed changes with the global seed and `n_envs`.
- Consequences: (1) a PushT-M score is an average over `ceil(50 / n_envs)` goals, not 50; (2) with the default
  `n_envs = n_train + n_test = 56` (one chunk) all test episodes share one goal, which is most likely the setting of the
  paper's 0.88; (3) "same seeds" does not mean the same goals across runs with different seeds or `n_envs`.
- Fix: option `seeded_goal` (env, runner, `eval_sim.py --seeded_goal True`) draws the goal from
  `RandomState([env_seed, 1])`. Checked: 50 distinct goals for 50 seeds, identical for any global seed and `n_envs`.
  Default `False` keeps the upstream behaviour for paper-protocol numbers. Results: `results/exp01/pusht_multitask_seeded_goal/`.
- In report: yes

## ISSUE-004 — `eval_sim.py --seed` had no effect
- Date: 2026-10-14
- Component: UVA (our change to `eval_sim.py`)
- Status: FIXED
- Symptom: EXP-01 with `--seed 42`, `0` and `1` gave identical per-seed rewards for both PushT and PushT-M.
- Cause (verified): `TrainUnifiedVideoActionWorkspace.__init__`
  (`unified_video_action/workspace/train_unified_video_action_workspace.py:48`) reseeds torch / numpy / random from
  `cfg.training.seed` after `eval_sim.py` has seeded them, so every run used 42.
- Fix: `eval_sim.py` also writes the chosen seed to `cfg.training.seed`. The seed-0 and seed-1 runs were deleted and rerun;
  the seed-42 runs were valid and kept. Driver log of the first attempt: `results/exp01/driver_run1_seed_bug.log`.
- In report: no

## ISSUE-003 — Libero10 evaluation returns `test_mean_score: NaN`
- Date: 2026-10-14
- Component: UVA
- Status: FIXED (data downloaded; the loader now fails loudly when no task file is found)
- Command that fails:
  ```bash
  python eval_sim.py --checkpoint checkpoints/libero10.ckpt --output_dir checkpoints/libero10
  ```
- Error: no exception; `eval_log_libero10.ckpt.json` contains only `"test_mean_score": NaN`.
- Cause (verified by reading the code): `load_env_runner` (`unified_video_action/utils/load_env.py:9`) builds one runner
  per file of `glob(cfg.task.dataset.dataset_path + "/*hdf5")`. `uva/data/` does not exist, so the list is empty, no
  rollout runs, and `eval_sim.py` takes the mean of an empty list.
- Fix: download the Libero10 hdf5 files (link in the upstream README) into the dataset path of the checkpoint's config;
  install LIBERO if the runner then fails to import it.
- In report: yes (a silent failure: the script finishes and writes a number)

## ISSUE-002 — Evaluation settings were changed by overwriting the checkpoint files
- Date: 2026-10-14
- Component: UVA
- Status: FIXED
- What was done: to make evaluation fit, `cfg.task.env_runner` (`n_test=4`, `n_train=0`, ...) was edited inside
  `pusht.ckpt`, `pusht_multitask.ckpt` and `libero10.ckpt` with `torch.save`, after copying them to `*_backup.ckpt`.
- Problem: the evaluation protocol was no longer visible in any command, and the 4-seed results are too few to compare with
  the paper (4/4 successes gives a 95% interval of [0.51, 1.00]).
- Fix: the `*_backup.ckpt` originals were checked (`n_test=50`, `test_start_seed=100000` for PushT and PushT-M) and moved
  back to `*.ckpt`; sha256 in `uva/checkpoints/SHA256SUMS`. Overrides are now command-line options of `eval_sim.py`.
- In report: no

## ISSUE-001 — Default PushT evaluation does not run on the L4 machine
- Date: 2026-10-13
- Component: UVA / Infra
- Status: FIXED
- Command that fails:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python eval_sim.py --checkpoint checkpoints/pusht.ckpt --output_dir checkpoints/pusht
  ```
- Error: not recorded.
- Cause (hypothesis, from the code and the machine): `n_envs` is `null`, so `PushTImageRunner` starts
  `n_train + n_test = 56` simulator processes and runs the policy on a batch of 56 at once; the machine has 16 GB RAM.
- Fix: `eval_sim.py --n_envs <k>` runs the same seeds in chunks of `k`.
- In report: yes

<!--
Entry template:

## ISSUE-NNN — <short title>
- Date: YYYY-MM-DD
- Component: Infra (GCP) | UVA | LingBot-VA | Simulator | Repo
- Status: OPEN | FIXED | WORKAROUND | WONTFIX
- Command that fails:
  ```bash
  <exact command>
  ```
- Error (the most important lines only):
  ```
  <message>
  ```
- Library versions involved (`pip freeze | grep -i <pkg>`):
- Cause: <verified, or "hypothesis:">
- Fix: <file, line, version, command>
- Time spent: <hours>
- In report: yes | no
-->
