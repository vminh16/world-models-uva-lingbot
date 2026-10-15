"""Training curves of a fine-tuning run from its logs.jsonl (written by the training workspace).

Per-step rows carry train_loss, diffusion_loss (video, 0 when the sampled mode has no video target), action_loss
(0 when the mode has no action target) and lr; end-of-epoch rows also carry video_fvd. Each loss is plotted only over
the steps where it was computed, with a moving average.

Usage:
    python scripts/plot_train_curves.py results/exp02/train_f2_k25 [more run dirs...] -o results/exp02/figures/train_curves.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(run_dir):
    with open(os.path.join(run_dir, "logs.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def smooth(y, w=50):
    if len(y) < w:
        return np.asarray(y)
    return np.convolve(y, np.ones(w) / w, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for run in args.runs:
        rows = load(run)
        name = os.path.basename(run.rstrip("/"))
        steps = [r for r in rows if "video_fvd" not in r]
        epochs = [r for r in rows if "video_fvd" in r]
        for ax, key, title in [(axes[0], "action_loss", "action loss"), (axes[1], "diffusion_loss", "video loss")]:
            pts = [(r["global_step"], r[key]) for r in steps if r.get(key, 0.0) > 0.0]
            if pts:
                x, y = map(np.asarray, zip(*pts))
                ax.plot(x, y, alpha=0.15)
                ys = smooth(y)
                ax.plot(x[len(x) - len(ys):], ys, label=name, color=ax.lines[-1].get_color())
            ax.set_title(f"{title} (steps where computed, MA 50)")
            ax.set_xlabel("training step")
            ax.set_yscale("log")
        axes[2].plot([r["global_step"] for r in epochs], [r["video_fvd"] for r in epochs], "o-", label=name)
        axes[2].set_title("FVD on 4 validation batches (lower is better)")
        axes[2].set_xlabel("training step")
        axes[3].plot([r["global_step"] for r in steps], [r["lr"] for r in steps], label=name)
        axes[3].set_title("learning rate")
        axes[3].set_xlabel("training step")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("->", args.out)


if __name__ == "__main__":
    main()
