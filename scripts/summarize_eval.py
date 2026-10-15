"""Summarize an eval_sim.py log into metrics.json.

Success = coverage above the 95% threshold of the PushT env, i.e. sim_max_reward == 1.0
(reward = clip(coverage / 0.95, 0, 1), see unified_video_action/env/pusht/pusht_env.py).

Usage:
    python scripts/summarize_eval.py results/exp01/pusht/eval_log_pusht.ckpt.json [--out metrics.json] [--prefix test/]
"""

import argparse
import json
import math
import os
import re


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(log, prefix="test/"):
    pattern = re.compile(re.escape(prefix) + r"sim_max_reward_(\d+)$")
    rewards = {int(m.group(1)): float(v) for k, v in log.items() if (m := pattern.match(k))}
    if not rewards:
        raise ValueError(f"no '{prefix}sim_max_reward_<seed>' keys in the log")

    seeds = sorted(rewards)
    values = [rewards[s] for s in seeds]
    n = len(values)
    successes = sum(v >= 1.0 for v in values)
    mean_cov = sum(values) / n
    std_cov = math.sqrt(sum((v - mean_cov) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    lo, hi = wilson_interval(successes, n)

    return {
        "n_seeds": n,
        "seed_range": [seeds[0], seeds[-1]],
        "success_rate": successes / n,
        "success_rate_ci95_wilson": [lo, hi],
        "n_success": successes,
        "mean_score": mean_cov,
        "mean_score_std": std_cov,
        "mean_score_sem": std_cov / math.sqrt(n),
        "mean_score_logged": log.get(prefix + "mean_score"),
        "per_seed_max_reward": {str(s): rewards[s] for s in seeds},
        "failed_seeds_sorted": [s for s in sorted(seeds, key=lambda s: rewards[s]) if rewards[s] < 1.0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--out", default=None, help="default: metrics.json next to the log")
    parser.add_argument("--prefix", default="test/")
    args = parser.parse_args()

    with open(args.log) as f:
        log = json.load(f)
    metrics = {"source_log": os.path.abspath(args.log), **summarize(log, args.prefix)}

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.log)), "metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    lo, hi = metrics["success_rate_ci95_wilson"]
    print(
        f"n={metrics['n_seeds']}  success={metrics['success_rate']:.3f} [{lo:.3f}, {hi:.3f}]  "
        f"mean_score={metrics['mean_score']:.3f} ± {metrics['mean_score_sem']:.3f} (sem)  -> {out}"
    )


if __name__ == "__main__":
    main()
