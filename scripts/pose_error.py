"""Position vs orientation error of the T block at the end of PushT-M rollouts, from the recorded 96x96 videos.

The T of the environment (pusht_env.py add_tee: a 120x30 bar and a 30x90 stem in a 512-px window, i.e. bars about 5.6 px
thick in the 96-px frame) is rasterized as a template at 2-degree steps. Its pose (centre, angle) is fitted separately to
  - the goal: green pixels, union over 5 frames of the episode (the goal is static; the union removes occlusion), and
  - the block: grey pixels of the last frame,
by maximizing IoU over angle and over integer shifts of up to 3 px around the mask centroid. Reported per run: the median
position error (px, distance between fitted centres), the median absolute angle error, and how often each error is
large. Videos exist for the first 10 test goals of each policy seed (30 episodes per run).

Usage:
    python scripts/pose_error.py            # prints a table and writes results/exp02/pose_error.json
"""

import glob
import json
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

RUNS = [("not fine-tuned", "results/exp02/f1_pusht_zeroshot"),
        ("10 demos", "results/exp02/f3_k10"),
        ("25 demos, action only", "results/exp02/f4_k25_action_only"),
        ("25 demos", "results/exp02/f2_k25"),
        ("50 demos", "results/exp02/f3_k50"),
        ("authors' model (247)", "results/exp01/pusht_multitask_seeded_goal")]
SIZE, SS = 96, 8  # frame size, supersampling for the template
SCALE = SIZE / 512  # env window 512 px -> rendered 96 px
ANGLES = np.arange(0, 360, 2)
SHIFTS = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)]

grey = lambda im: np.abs(im - [119, 136, 153]).sum(-1) < 90  # noqa: E731
green = lambda im: np.abs(im - [144, 238, 144]).sum(-1) < 90  # noqa: E731


def tee_templates():
    """T masks centred (area centroid) on the frame centre, one per angle."""
    bar = [(-60, 30), (60, 30), (60, 0), (-60, 0)]
    stem = [(-15, 30), (-15, 120), (15, 120), (15, 30)]
    cy = (3600 * 15 + 2700 * 75) / 6300  # area centroid of the T, in env units
    out = []
    for a in np.deg2rad(ANGLES):
        im = Image.new("L", (SIZE * SS, SIZE * SS), 0)
        d = ImageDraw.Draw(im)
        for poly in (bar, stem):
            pts = []
            for x, y in poly:
                y = y - cy
                xr, yr = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
                pts.append(((SIZE / 2 + xr * SCALE) * SS, (SIZE / 2 + yr * SCALE) * SS))
            d.polygon(pts, fill=255)
        out.append(np.array(im.resize((SIZE, SIZE), Image.BOX)) > 127)
    return np.stack(out)


TEMPLATES = tee_templates()


def fit(mask):
    """Best (centre_yx, angle_deg, iou) of the T template for a mask."""
    ys, xs = np.nonzero(mask)
    c = np.round([ys.mean(), xs.mean()]).astype(int)
    best = (None, None, -1.0)
    for dy, dx in SHIFTS:
        sh = (c[0] + dy - SIZE // 2, c[1] + dx - SIZE // 2)
        t = np.roll(TEMPLATES, sh, axis=(1, 2))
        inter = (t & mask).sum((1, 2))
        union = (t | mask).sum((1, 2))
        iou = inter / np.maximum(union, 1)
        k = int(iou.argmax())
        if iou[k] > best[2]:
            best = (np.array([c[0] + dy, c[1] + dx], float), float(ANGLES[k]), float(iou[k]))
    return best


def frames(mp4, tmp):
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", mp4], capture_output=True, text=True).stdout)
    idx = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", mp4, "-vf", f"select={sel}", "-vsync", "0",
                    os.path.join(tmp, "f%d.png")], check=True)
    ims = [np.array(Image.open(os.path.join(tmp, f"f{i + 1}.png")).convert("RGB")).astype(int) for i in range(len(idx))]
    return ims


def analyse(mp4, tmp):
    ims = frames(mp4, tmp)
    g = np.any([green(im) for im in ims], axis=0)
    b = grey(ims[-1])
    if g.sum() < 20 or b.sum() < 20:
        return None
    gc, ga, giou = fit(g)
    bc, ba, biou = fit(b)
    da = abs((ba - ga + 180) % 360 - 180)
    return dict(pos_px=float(np.linalg.norm(bc - gc)), angle_deg=float(da), fit_iou_goal=giou, fit_iou_block=biou)


def main():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, run in RUNS:
            rows = []
            for log in sorted(glob.glob(os.path.join(run, "seed*", "eval_log_*.json"))):
                d = json.load(open(log))
                for k, v in d.items():
                    if not k.startswith("test/sim_video_"):
                        continue
                    seed = k.rsplit("_", 1)[1]
                    r = analyse(os.path.join(os.path.dirname(log), "media", os.path.basename(v)), tmp)
                    if r:
                        r.update(goal=seed, reward=d[f"test/sim_max_reward_{seed}"])
                        rows.append(r)
            a = np.array([r["angle_deg"] for r in rows])
            p = np.array([r["pos_px"] for r in rows])
            out[name] = dict(n=len(rows), median_pos_px=float(np.median(p)), median_angle_deg=float(np.median(a)),
                             frac_angle_over_10=float((a > 10).mean()), frac_pos_over_2_8px=float((p > 2.8).mean()),
                             median_fit_iou=float(np.median([r["fit_iou_block"] for r in rows])), episodes=rows)
            o = out[name]
            print(f"{name:24s} n={o['n']:2d}  position {o['median_pos_px']:4.1f}px (>half a bar: {o['frac_pos_over_2_8px']:.2f})"
                  f"  angle {o['median_angle_deg']:5.1f}deg (>10deg: {o['frac_angle_over_10']:.2f})  fit IoU {o['median_fit_iou']:.2f}")
    json.dump(out, open("results/exp02/pose_error.json", "w"), indent=1)


if __name__ == "__main__":
    main()
