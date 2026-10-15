"""Figure 2 of the technical report (Part 2): one row, two panels, sized for half a page.

(a) UVA fine-tuning: PushT-M mean score vs number of demonstrations (EXP-02/03), with the action-only run, the
    zero-shot checkpoint and the fully trained upper bound. Error bars: 95% bootstrap over the 50 goals.
(b) LingBot-VA: seconds per chunk over 40 open-loop chunks for full memory vs a 4-chunk KV window, and over 10 chunks
    for fewer denoising steps (EXP-11).

Usage:
    python scripts/plot_report_figure.py -o report/figures/fig2_part2.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_data_scaling import SCALING, goal_ci  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    s2 = json.load(open("results/exp02/summary.json"))
    up = json.load(open("results/exp01/summary.json"))["pusht_multitask_seeded_goal"]
    s11 = json.load(open("results/exp11/summary.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw=dict(width_ratios=[1, 1.35]))

    ks = [k for k, _ in SCALING]
    m = [s2[n]["mean_score_mean"] for _, n in SCALING]
    ci = np.array([goal_ci(s2[n]) for _, n in SCALING])
    ax1.errorbar(ks, m, yerr=[np.array(m) - ci[:, 0], ci[:, 1] - np.array(m)], marker="o", capsize=3,
                 label="video + action")
    f4 = s2["f4_k25_action_only"]
    c4 = goal_ci(f4)
    ax1.errorbar([26.5], [f4["mean_score_mean"]], yerr=[[f4["mean_score_mean"] - c4[0]], [c4[1] - f4["mean_score_mean"]]],
                 marker="s", capsize=3, color="tab:orange", ls="none", label="action only")
    ax1.axhline(up["mean_score_mean"], color="tab:green", ls="--", lw=1, label="author model, 247 demos")
    ax1.axhline(s2["f1_pusht_zeroshot"]["mean_score_mean"], color="gray", ls="--", lw=1, label="before fine-tuning")
    ax1.set_xticks(ks)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("demonstrations used for fine-tuning")
    ax1.set_ylabel("PushT-M mean score")
    ax1.set_title("(a) UVA: fine-tuning with few demos", fontsize=10)
    ax1.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1, 0.24))
    ax1.grid(alpha=0.3)

    runs = [("i2av_c40_s0", "full memory, 25/50 steps (default)", "tab:purple"),
            ("win8_c40", "4-chunk memory, 25/50 steps", "tab:olive"),
            ("steps_v10_a20", "full memory, 10/20 steps", "tab:pink"),
            ("steps_v5_a10", "full memory, 5/10 steps", "tab:gray")]
    for run, label, c in runs:
        t = s11[run]["chunk_seconds"][1:]
        ax2.plot(range(1, len(t) + 1), t, marker="o", ms=2, lw=1.2, color=c, label=label)
    ax2.axvline(35.5, color="k", ls=":", lw=1)
    ax2.text(35.3, 10.5, "cache full", ha="right", fontsize=7)
    ax2.set_ylim(0, 26)
    ax2.set_xlabel("chunk index (1 chunk = 32 actions)")
    ax2.set_ylabel("seconds per chunk")
    ax2.set_title("(b) LingBot-VA on one L4: time per chunk", fontsize=10)
    ax2.legend(fontsize=7, loc="lower right", ncol=2)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
