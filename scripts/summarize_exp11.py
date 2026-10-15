"""EXP-11 summary: LingBot-VA i2av latency and memory on one L4, plus figures.

Reads every results/exp11/<run>/ that has timing.json (written by wan_va_server.py), vram.csv, ram.csv and run.log
(written by scripts/run_exp11.sh). Writes results/exp11/summary.json and, in results/exp11/figures/:
  latency_per_chunk.png   seconds per chunk vs chunk index (KV cache window marked)
  frames_<run>.png        evenly spaced frames of each generated video (needs ffmpeg)

Prompt encoding time = from the "Reset." log line to the first denoising step ("FRAME START ID"), i.e. the T5 text
encoder on the CPU (offload) for the prompt and the empty negative prompt.

Usage:
    python scripts/summarize_exp11.py
"""

import csv
import glob
import json
import os
import re
import subprocess
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "results/exp11"
KV_WINDOW_CHUNKS = 72 // 2  # attn_window / 2 chunks fit in the preallocated cache (model.py create_empty_cache)
TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - root - INFO - (.*)")


def log_times(path):
    """Time of the 'Reset.' line and of the first denoising step."""
    reset = first = None
    for line in open(path, errors="ignore"):
        for ts, msg in TS.findall(line):
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
            if msg.startswith("Reset.") and reset is None:
                reset = t
            if msg.startswith("FRAME START ID") and first is None:
                first = t
    return reset, first


def peak_csv(path, col):
    vals = []
    with open(path) as f:
        for row in csv.reader(f):
            try:
                vals.append(float(row[col].split()[0]))
            except (ValueError, IndexError):
                pass
    return max(vals) if vals else None


def run_summary(d):
    t = json.load(open(os.path.join(d, "timing.json")))
    cs = np.array(t["chunk_seconds"])
    reset, first = log_times(os.path.join(d, "run.log"))
    wall = re.findall(r"wall_seconds=(\d+)", open(os.path.join(d, "run.log")).read())
    n_chunks = t["num_chunks"]
    actions_per_chunk = t["frame_chunk_size"] * t["action_per_frame"]
    return dict(
        num_chunks=n_chunks,
        seed=t["seed"],
        first_chunk_s=float(cs[0]),
        later_chunks_mean_s=float(cs[1:].mean()) if len(cs) > 1 else None,
        later_chunks_min_max_s=[float(cs[1:].min()), float(cs[1:].max())] if len(cs) > 1 else None,
        generation_total_s=float(cs.sum()),
        decode_s=t["decode_seconds"],
        prompt_encoding_cpu_s=(first - reset).total_seconds() if reset and first else None,
        wall_s=int(wall[-1]) if wall else None,
        actions_per_chunk=actions_per_chunk,
        video_frames=t["video_frames"],
        video_shape=t["video_shape"],
        peak_allocated_gib=t["peak_allocated_gb_total"],
        peak_nvidia_smi_mib=peak_csv(os.path.join(d, "vram.csv"), 1),
        peak_ram_used_mib=peak_csv(os.path.join(d, "ram.csv"), 1),
        peak_swap_used_mib=peak_csv(os.path.join(d, "ram.csv"), 2),
        chunk_seconds=t["chunk_seconds"],
    )


def frame_strip(video, out, n=8):
    total = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video],
                               capture_output=True, text=True).stdout.strip())
    idx = np.linspace(0, total - 1, n).round().astype(int)
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", video, "-vf",
                    f"select='{sel}',scale=240:-1,tile={n}x1", "-frames:v", "1", out], check=True)
    return idx.tolist()


def main():
    runs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "*", "timing.json"))):
        d = os.path.dirname(f)
        runs[os.path.basename(d)] = run_summary(d)
    json.dump(runs, open(os.path.join(ROOT, "summary.json"), "w"), indent=2)

    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, r in runs.items():
        if name.startswith("smoke"):
            continue
        cs = r["chunk_seconds"]
        ax.plot(range(len(cs)), cs, "o-" if len(cs) <= 10 else ".-", ms=3, lw=1,
                label=f"{name} ({r['num_chunks']} chunks)")
    ax.axvline(KV_WINDOW_CHUNKS - 0.5, color="gray", ls="--", lw=1)
    ax.text(KV_WINDOW_CHUNKS - 0.3, ax.get_ylim()[1] * 0.95, "KV cache full", fontsize=8, va="top", color="gray")
    ax.set_xlabel("chunk index (1 chunk = 2 latent frames + 32 actions)")
    ax.set_ylabel("seconds per chunk")
    ax.set_title("LingBot-VA i2av on one L4: generation time per chunk")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "latency_per_chunk.png"), dpi=150)

    for name in runs:
        video = os.path.join(ROOT, name, "demo.mp4")
        if os.path.exists(video):
            idx = frame_strip(video, os.path.join(ROOT, "figures", f"frames_{name}.png"))
            runs[name]["strip_frame_indices"] = idx
    json.dump(runs, open(os.path.join(ROOT, "summary.json"), "w"), indent=2)
    for name, r in runs.items():
        print(name, {k: v for k, v in r.items() if k != "chunk_seconds"})


if __name__ == "__main__":
    main()
