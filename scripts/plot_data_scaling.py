"""EXP-02/03 summary figure: PushT-M mean score and success rate vs number of fine-tuning demonstrations.

Reads results/exp02/summary.json (aggregate_seeds.py) and the upper bound from results/exp01/summary.json.
Points: mean over the 3 policy seeds; error bars: 95% bootstrap interval over the 50 goals (goal = resampling unit,
reward averaged over the policy seeds first). F1 and the upper bound are horizontal lines, F4 a separate marker.

Usage:
    python scripts/plot_data_scaling.py -o results/exp02/figures/data_scaling.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GOALS = [str(e) for e in range(100000, 100050)]
SCALING = [(10, "f3_k10"), (25, "f2_k25"), (50, "f3_k50")]


def goal_ci(summary, n_boot=10000, seed=0):
    r = np.array([summary["mean_reward_per_env_seed"][g] for g in GOALS])
    idx = np.random.default_rng(seed).integers(0, len(r), (n_boot, len(r)))
    return np.percentile(r[idx].mean(1), [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    s2 = json.load(open("results/exp02/summary.json"))
    up = json.load(open("results/exp01/summary.json"))["pusht_multitask_seeded_goal"]
    f1 = s2["f1_pusht_zeroshot"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    pts = [(k, s2[n]) for k, n in SCALING if n in s2]
    ks = [k for k, _ in pts]
    score = [s["mean_score_mean"] for _, s in pts]
    ci = np.array([goal_ci(s) for _, s in pts])
    ax1.errorbar(ks, score, yerr=[np.array(score) - ci[:, 0], ci[:, 1] - np.array(score)], fmt="o-", capsize=4,
                 label="fine-tuned, video + action (F2/F3)")
    if "f4_k25_action_only" in s2:
        f4 = s2["f4_k25_action_only"]
        lo, hi = goal_ci(f4)
        ax1.errorbar([25.8], [f4["mean_score_mean"]], yerr=[[f4["mean_score_mean"] - lo], [hi - f4["mean_score_mean"]]],
                     fmt="s", color="tab:orange", capsize=4, label="fine-tuned, action only (F4)")
    ax1.axhline(f1["mean_score_mean"], color="gray", ls="--", label="pusht.ckpt zero-shot (F1)")
    ax1.axhline(up["mean_score_mean"], color="green", ls="--", label="pusht_multitask.ckpt, 247 demos (upper)")
    ax1.set_xlabel("PushT-M demonstrations used for fine-tuning (same 3k-step budget)")
    ax1.set_ylabel("mean score (max coverage)")
    ax1.set_xticks([10, 25, 50])
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(alpha=0.3)

    succ = [s["pooled_success_rate"] for _, s in pts]
    sci = np.array([s["pooled_success_rate_ci95_wilson"] for _, s in pts])
    ax2.errorbar(ks, succ, yerr=[np.array(succ) - sci[:, 0], sci[:, 1] - np.array(succ)], fmt="o-", capsize=4)
    ax2.axhline(f1["pooled_success_rate"], color="gray", ls="--")
    ax2.axhline(up["pooled_success_rate"], color="green", ls="--")
    ax2.set_xlabel("PushT-M demonstrations")
    ax2.set_ylabel("success rate (coverage ≥ 95%), Wilson 95%")
    ax2.set_xticks([10, 25, 50])
    ax2.set_ylim(0, 0.4)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("->", a.out)


if __name__ == "__main__":
    main()
