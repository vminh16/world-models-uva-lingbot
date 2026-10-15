"""Where does the policy leave the block? Final-frame block centroid vs the fixed PushT goal and vs the episode's goal.

Reads the rollout videos (96x96) of an evaluation folder (results/<exp>/<name>/seed*/), segments the grey block in the
last frame and the green goal in the first frame by colour, and prints the median centroid distances in pixels.
The fixed PushT goal centroid (42.5,52.9) is the mean block centroid at the end of 4 successful PushT rollouts.

Usage:
    python scripts/goal_distance.py results/exp02/f1_pusht_zeroshot <tmp_dir> 42.5,52.9
"""
import glob, json, os, subprocess, sys
import numpy as np
from PIL import Image

def frame(mp4, png, first=False):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"] + (["-i", mp4, "-vframes", "1"] if first else ["-sseof", "-0.2", "-i", mp4, "-update", "1", "-vframes", "1"]) + [png]
    subprocess.run(cmd, check=True)
    return np.array(Image.open(png).convert("RGB")).astype(int)

def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])

grey = lambda im: (np.abs(im - [119, 136, 153]).sum(-1) < 90)
green = lambda im: (np.abs(im - [144, 238, 144]).sum(-1) < 90)
tmp = sys.argv[2]
FIXED = np.array([float(x) for x in sys.argv[3].split(",")])
d_fixed, d_goal = [], []
for log in sorted(glob.glob(os.path.join(sys.argv[1], "seed*", "eval_log_*.json"))):
    d = json.load(open(log))
    for k, v in d.items():
        if "sim_video" not in k:
            continue
        mp4 = os.path.join(os.path.dirname(log), "media", os.path.basename(v))
        goal_c = centroid(green(frame(mp4, os.path.join(tmp, "f.png"), first=True)))  # goal unoccluded at t=0 (mostly)
        blk_c = centroid(grey(frame(mp4, os.path.join(tmp, "l.png"))))
        d_fixed.append(np.linalg.norm(blk_c - FIXED)); d_goal.append(np.linalg.norm(blk_c - goal_c))
d_fixed, d_goal = np.array(d_fixed), np.array(d_goal)
print(f"n={len(d_fixed)}  dist to FIXED PushT goal: median {np.nanmedian(d_fixed):.1f}px  |  dist to episode goal: median {np.nanmedian(d_goal):.1f}px  (frame 96px)")
