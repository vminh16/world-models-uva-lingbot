"""Aggregate repeated evaluations (results/<exp>/<task>/seed<k>/metrics.json) into results/<exp>/summary.json.

Per task: success rate and mean score per repeat, mean and std over repeats, success rate pooled over all episodes with
a 95% Wilson interval, how many repeats each test seed succeeded in, peak GPU memory and wall time.

Usage:
    python scripts/aggregate_seeds.py results/exp01
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics

from summarize_eval import wilson_interval


def peak_vram_mib(path):
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    return max(int(r[1].strip().split()[0]) for r in rows if len(r) > 1)


def wall_time(path):
    with open(path) as f:
        for line in f:
            if "Elapsed (wall clock)" in line:
                return line.split()[-1]
    return None


def aggregate_task(task_dir):
    runs = {}
    for metrics_path in sorted(glob.glob(os.path.join(task_dir, "seed*", "metrics.json"))):
        run_dir = os.path.dirname(metrics_path)
        seed = int(re.search(r"seed(\d+)$", run_dir).group(1))
        with open(metrics_path) as f:
            m = json.load(f)
        m["peak_vram_mib"] = peak_vram_mib(os.path.join(run_dir, "vram.csv"))
        m["wall_time"] = wall_time(os.path.join(run_dir, "run.log"))
        runs[seed] = m

    success = [runs[s]["success_rate"] for s in runs]
    score = [runs[s]["mean_score"] for s in runs]
    n_total = sum(runs[s]["n_seeds"] for s in runs)
    k_total = sum(runs[s]["n_success"] for s in runs)
    lo, hi = wilson_interval(k_total, n_total)

    env_seeds = sorted(runs[next(iter(runs))]["per_seed_max_reward"], key=int)
    successes_per_env_seed = {
        e: sum(runs[s]["per_seed_max_reward"][e] >= 1.0 for s in runs) for e in env_seeds
    }
    mean_reward_per_env_seed = {
        e: statistics.mean(runs[s]["per_seed_max_reward"][e] for s in runs) for e in env_seeds
    }

    return {
        "policy_seeds": sorted(runs),
        "per_run": {
            str(s): {k: runs[s][k] for k in ["success_rate", "mean_score", "n_seeds", "peak_vram_mib", "wall_time"]}
            for s in sorted(runs)
        },
        "success_rate_mean": statistics.mean(success),
        "success_rate_std": statistics.stdev(success) if len(success) > 1 else 0.0,
        "mean_score_mean": statistics.mean(score),
        "mean_score_std": statistics.stdev(score) if len(score) > 1 else 0.0,
        "pooled_episodes": n_total,
        "pooled_success_rate": k_total / n_total,
        "pooled_success_rate_ci95_wilson": [lo, hi],
        "env_seeds_never_successful": [e for e in env_seeds if successes_per_env_seed[e] == 0],
        "env_seeds_always_successful": [e for e in env_seeds if successes_per_env_seed[e] == len(runs)],
        "env_seeds_by_mean_reward": sorted(env_seeds, key=lambda e: mean_reward_per_env_seed[e]),
        "mean_reward_per_env_seed": mean_reward_per_env_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    args = parser.parse_args()

    tasks = sorted(d for d in os.listdir(args.exp_dir) if glob.glob(os.path.join(args.exp_dir, d, "seed*", "metrics.json")))
    summary = {t: aggregate_task(os.path.join(args.exp_dir, t)) for t in tasks}
    out = os.path.join(args.exp_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    for t, s in summary.items():
        lo, hi = s["pooled_success_rate_ci95_wilson"]
        print(
            f"{t:18s} success {s['success_rate_mean']:.3f} ± {s['success_rate_std']:.3f} (repeats {s['policy_seeds']}), "
            f"pooled {s['pooled_success_rate']:.3f} [{lo:.3f}, {hi:.3f}] n={s['pooled_episodes']} | "
            f"mean score {s['mean_score_mean']:.3f} ± {s['mean_score_std']:.3f} | "
            f"never/always successful env seeds: {len(s['env_seeds_never_successful'])}/{len(s['env_seeds_always_successful'])}"
        )
    print("->", out)


if __name__ == "__main__":
    main()
