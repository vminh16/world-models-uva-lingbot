"""Which PushT-M episodes a fine-tuning run trains on, and how long they are.

Rebuilds the train / validation split exactly as PushTImageDataset does (val_ratio 0.02, seed 42, then
downsample_mask to k episodes) and writes results/exp02/dataset_k<k>.json.

Usage (from the repository root, `uva` env):
    python scripts/exp02_dataset_info.py 25
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "uva"))

from unified_video_action.common.replay_buffer import ReplayBuffer  # noqa: E402
from unified_video_action.common.sampler import downsample_mask, get_val_mask  # noqa: E402


def main():
    k = int(sys.argv[1])
    path = os.path.join(ROOT, "uva", "data", "pusht_multitask")
    rb = ReplayBuffer.copy_from_path(path, keys=["state", "action"])
    lengths = np.diff(np.concatenate([[0], rb.episode_ends[:]]))
    val_mask = get_val_mask(n_episodes=rb.n_episodes, val_ratio=0.02, seed=42)
    train_mask = downsample_mask(mask=~val_mask, max_n=k, seed=42)
    train_idx = np.where(train_mask)[0]
    info = {
        "dataset": "uva/data/pusht_multitask",
        "n_episodes": int(rb.n_episodes),
        "episode_length_all": {"mean": float(lengths.mean()), "min": int(lengths.min()), "max": int(lengths.max())},
        "k": k,
        "train_episodes": train_idx.tolist(),
        "train_episode_lengths": lengths[train_idx].tolist(),
        "train_frames": int(lengths[train_idx].sum()),
        "val_ratio_episodes": np.where(val_mask)[0].tolist(),
    }
    out = os.path.join(ROOT, "results", "exp02", f"dataset_k{k}.json")
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({key: v for key, v in info.items() if key != "val_ratio_episodes"}))
    print("->", out)


if __name__ == "__main__":
    main()
