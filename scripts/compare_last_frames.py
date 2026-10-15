"""Side-by-side last frames of several evaluations on the same goals (env seeds), with the episode's max reward.

Usage:
    python scripts/compare_last_frames.py -o results/exp02/figures/last_frames_f1_f2_upper.png --seed 42 \
        --run "F1 zero-shot=results/exp02/f1_pusht_zeroshot" --run "F2 k=25=results/exp02/f2_k25" \
        --run "upper bound=results/exp01/pusht_multitask_seeded_goal" --env_seeds 100000-100009
"""

import argparse
import json
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw


def last_frame(mp4, png):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-sseof", "-0.2", "-i", mp4,
                    "-update", "1", "-vframes", "1", png], check=True)
    return Image.open(png).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--seed", default="42", help="policy seed folder")
    p.add_argument("--run", action="append", required=True, help="label=eval folder")
    p.add_argument("--env_seeds", default="100000-100009")
    p.add_argument("--scale", type=int, default=2)
    a = p.parse_args()
    lo, hi = map(int, a.env_seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    runs = [r.rsplit("=", 1) for r in a.run]
    size, pad, head = 96 * a.scale, 4, 16
    canvas = Image.new("RGB", (100 + len(seeds) * (size + pad), len(runs) * (size + head + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    with tempfile.TemporaryDirectory() as tmp:
        for i, (label, folder) in enumerate(runs):
            run_dir = os.path.join(folder, f"seed{a.seed}")
            log = json.load(open([os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith("eval_log_")][0]))
            y = i * (size + head + pad)
            draw.text((4, y + head + size // 2), label, fill="black")
            for j, s in enumerate(seeds):
                x = 100 + j * (size + pad)
                mp4 = os.path.join(run_dir, "media", os.path.basename(log[f"test/sim_video_{s}"]))
                im = last_frame(mp4, os.path.join(tmp, "f.png")).resize((size, size), Image.NEAREST)
                canvas.paste(im, (x, y + head))
                r = log[f"test/sim_max_reward_{s}"]
                draw.text((x + 2, y + 2), f"{s} r={r:.2f}", fill="green" if r >= 1 else "black")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    canvas.save(a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
