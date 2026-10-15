"""Summarize EXP-05 (UVA on LIBERO-10) into results/exp05/summary.json.

Each LIBERO episode reward is 0/1 (task success), so a task's success rate is the mean of its
`test/<task>_sim_max_reward_<seed>` values. The overall number is the mean over tasks (as in the paper) and,
for an interval, the success rate pooled over all episodes with a 95% Wilson interval.

Usage:
    python scripts/summarize_libero.py results/exp05 [--seed 42]
"""

import argparse
import glob
import json
import os
import re

from summarize_eval import wilson_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reported", type=float, default=0.90, help="paper Table II, UVA on Libero10")
    args = parser.parse_args()

    tasks = {}
    for log_path in sorted(glob.glob(os.path.join(args.exp_dir, "*", f"seed{args.seed}", "eval_log_*.json"))):
        task = log_path.split(os.sep)[-3]
        with open(log_path) as f:
            log = json.load(f)
        rewards = [float(v) for k, v in log.items() if re.search(r"_sim_max_reward_\d+$", k)]
        tasks[task] = {
            "n_episodes": len(rewards),
            "n_success": int(sum(r >= 1.0 for r in rewards)),
            "success_rate": sum(r >= 1.0 for r in rewards) / len(rewards),
        }

    n = sum(t["n_episodes"] for t in tasks.values())
    k = sum(t["n_success"] for t in tasks.values())
    lo, hi = wilson_interval(k, n)
    summary = {
        "policy_seed": args.seed,
        "n_tasks": len(tasks),
        "mean_success_over_tasks": sum(t["success_rate"] for t in tasks.values()) / len(tasks),
        "pooled_episodes": n,
        "pooled_success_rate": k / n,
        "pooled_success_rate_ci95_wilson": [lo, hi],
        "reported_paper": args.reported,
        "per_task": tasks,
    }
    out = os.path.join(args.exp_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    for task, t in sorted(tasks.items(), key=lambda kv: kv[1]["success_rate"]):
        print(f"{t['success_rate']:.2f}  ({t['n_success']:2d}/{t['n_episodes']})  {task}")
    print(
        f"tasks={len(tasks)}  mean over tasks={summary['mean_success_over_tasks']:.3f}  "
        f"pooled={k}/{n}={k / n:.3f} [{lo:.3f}, {hi:.3f}]  reported={args.reported}  -> {out}"
    )


if __name__ == "__main__":
    main()
