"""Figure 3 of the technical report (Part 2): qualitative success and failure cases.

(a) UVA on PushT-M, last frame of three goals (rows) for four models (columns), cut from
    results/exp02/figures/last_frames_f1_f2_f4_upper.png (policy seed 42); the reward under each frame is read from
    the evaluation logs.
(b) LingBot-VA, 40 open-loop chunks from one image: full memory vs a 4-chunk KV window
    (results/exp11/figures/frames_{i2av_c40_s0,win8_c40}.png, 8 evenly spaced frames each).

Usage:
    python scripts/plot_report_qualitative.py -o report/figures/fig3_part2.png
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

GRID = "results/exp02/figures/last_frames_f1_f2_f4_upper.png"
X0, DX, W = 102, 196, 190  # cell geometry of GRID (pixels): first cell, period, width
Y0, DY, H = 20, 212, 188  # first cell below its title, period, height
GOALS = [0, 4, 8]  # columns of GRID: goals 100000, 100004, 100008
MODELS = [  # (row of GRID, label, run directory)
    (0, "before\nfine-tuning", "results/exp02/f1_pusht_zeroshot/seed42"),
    (1, "25 demos,\nvideo + action", "results/exp02/f2_k25/seed42"),
    (2, "25 demos,\naction only", "results/exp02/f4_k25_action_only/seed42"),
    (3, "author model,\n247 demos", "results/exp01/pusht_multitask_seeded_goal/seed42"),
]
FRAMES_PER_CHUNK = 316 / 40  # 1 conditioning frame + 40 chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    grid = np.array(Image.open(GRID).convert("RGB"))
    s11 = json.load(open("results/exp11/summary.json"))

    fig = plt.figure(figsize=(10, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.75], wspace=0.06)
    ga = gs[0].subgridspec(len(GOALS), len(MODELS), wspace=0.05, hspace=0.32)
    for j, (row, label, run) in enumerate(MODELS):
        log = json.load(open(glob.glob(os.path.join(run, "eval_log_*.json"))[0]))
        for i, c in enumerate(GOALS):
            ax = fig.add_subplot(ga[i, j])
            x, y = X0 + c * DX, Y0 + row * DY
            ax.imshow(grid[y:y + H, x:x + W])
            ax.set_xticks([])
            ax.set_yticks([])
            r = log[f"test/sim_max_reward_{100000 + c}"]
            ax.set_xlabel(f"r = {r:.2f}", fontsize=6.5, labelpad=1, color="green" if r >= 1 else "black")
            if i == 0:
                ax.set_title(label, fontsize=6.5)
            if j == 0:
                ax.set_ylabel(f"goal {i + 1}", fontsize=7)
    fig.text(0.125, 1.0, "(a) UVA on PushT-M: last frame (green T = goal)", fontsize=9, ha="left")

    gb = gs[1].subgridspec(2, 1, hspace=0.35)
    for i, (run, label) in enumerate([("i2av_c40_s0", "full memory"), ("win8_c40", "4-chunk memory")]):
        ax = fig.add_subplot(gb[i])
        ax.imshow(Image.open(f"results/exp11/figures/frames_{run}.png"))
        idx = s11[run]["strip_frame_indices"]
        tile = 1920 / len(idx)
        ax.set_xticks([tile * (k + 0.5) for k in range(len(idx))])
        ax.set_xticklabels([f"chunk {round(f / FRAMES_PER_CHUNK)}" for f in idx], fontsize=7)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(length=0)
    fig.text(0.44, 1.0, "(b) LingBot-VA: 40 chunks imagined from one image (open loop)", fontsize=9, ha="left")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
