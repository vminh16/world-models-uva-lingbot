"""Nested PushT-M training subsets for the data-scaling runs (F3): k10 ⊂ k25 ⊂ k50, where k25 is exactly the set F2
trained on (results/exp02/dataset_k25.json, chosen by upstream `downsample_mask`).

k10 = 10 episodes drawn from k25, k50 = k25 + 25 episodes drawn from the remaining non-validation episodes (numpy
default_rng(0)). For each subset: number of 32-step training windows and the number of epochs that gives the same
step budget as F2 (3024 steps of batch 16). Writes results/exp02/subsets.json.

Usage (from the repository root, `uva` env, after exp02_dataset_info.py 25):
    python scripts/exp02_nested_subsets.py
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "uva"))

from unified_video_action.common.replay_buffer import ReplayBuffer  # noqa: E402

BATCH, BUDGET, HORIZON, PAD = 16, 3024, 32, 1 + 7  # pad_before + pad_after of task/pusht.yaml


def windows(lengths):
    return int(sum(max(n + PAD - HORIZON + 1, 0) for n in lengths))


def main():
    k25_info = json.load(open(os.path.join(ROOT, "results", "exp02", "dataset_k25.json")))
    k25 = sorted(k25_info["train_episodes"])
    val = set(k25_info["val_ratio_episodes"])
    rb = ReplayBuffer.copy_from_path(os.path.join(ROOT, "uva", "data", "pusht_multitask"), keys=["action"])
    lengths = np.diff(np.concatenate([[0], rb.episode_ends[:]]))

    rng = np.random.default_rng(0)
    k10 = sorted(rng.choice(k25, size=10, replace=False).tolist())
    rest = [e for e in range(rb.n_episodes) if e not in val and e not in k25]
    k50 = sorted(k25 + rng.choice(rest, size=25, replace=False).tolist())

    out = {}
    for name, eps in [("k10", k10), ("k25", k25), ("k50", k50)]:
        n_win = windows(lengths[eps])
        steps_per_epoch = math.ceil(n_win / BATCH)
        epochs = round(BUDGET / steps_per_epoch)
        out[name] = {
            "episodes": eps,
            "frames": int(lengths[eps].sum()),
            "windows": n_win,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "steps": epochs * steps_per_epoch,
            "val_every": max(1, round(epochs / 18)),  # about 18 FVD evaluations per run, as in F2
        }
        print(name, {k: v for k, v in out[name].items() if k != "episodes"})
    assert set(k10) <= set(k25) <= set(k50)
    path = os.path.join(ROOT, "results", "exp02", "subsets.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("->", path)


if __name__ == "__main__":
    main()
