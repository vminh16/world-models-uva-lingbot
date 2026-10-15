"""EXP-11 ablations: compare each run with the reference run that has the same seed (same initial noise).

For every pair (run, reference): mean absolute difference of the generated actions per chunk (normalized action space as
written to actions.npy), PSNR of the decoded video per frame, and mean seconds per chunk (chunks >= 1). The seed-to-seed
difference of the reference config (i2av_c10_s1, i2av_c10_s2 vs i2av_c10_s0) is the yardstick: a change smaller than that
is within sampling noise. Writes results/exp11/ablations.json and results/exp11/figures/ablations.png.

Usage:
    python scripts/compare_exp11.py
"""

import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "results/exp11"
PAIRS = [  # (run, reference, label)
    ("i2av_c10_s1", "i2av_c10_s0", "other seed (1)"),
    ("i2av_c10_s2", "i2av_c10_s0", "other seed (2)"),
    ("steps_v10_a20", "i2av_c10_s0", "10 video / 20 action steps"),
    ("steps_v5_a10", "i2av_c10_s0", "5 video / 10 action steps"),
    ("cfg1", "i2av_c10_s0", "no CFG (guidance 1)"),
    ("prompt_left", "i2av_c10_s0", "other prompt (mug to the left)"),
    ("win8_c40", "i2av_c40_s0", "KV window 4 chunks (40 chunks)"),
]
ACTIONS_PER_CHUNK = 32


def load_video(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 384, 320, 3).astype(np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).reshape(len(a), -1).mean(1)
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))


def compare(run, ref):
    d, r = os.path.join(ROOT, run), os.path.join(ROOT, ref)
    a, b = np.load(os.path.join(d, "actions.npy")), np.load(os.path.join(r, "actions.npy"))
    n = min(a.shape[1], b.shape[1])
    act = np.abs(a[:, :n] - b[:, :n]).reshape(a.shape[0], -1, ACTIONS_PER_CHUNK).mean(axis=(0, 2))
    va, vb = load_video(os.path.join(d, "demo.mp4")), load_video(os.path.join(r, "demo.mp4"))
    m = min(len(va), len(vb))
    p = psnr(va[:m], vb[:m])
    t, tr = (json.load(open(os.path.join(x, "timing.json"))) for x in (d, r))
    return dict(
        reference=ref,
        action_mad_per_chunk=act.round(4).tolist(),
        action_mad_mean=float(act.mean()),
        psnr_per_frame=p.round(2).tolist(),
        psnr_mean=float(p[1:].mean()),  # frame 0 is the conditioning image
        sec_per_chunk=float(np.mean(t["chunk_seconds"][1:])),
        sec_per_chunk_reference=float(np.mean(tr["chunk_seconds"][1:])),
        chunk_seconds=t["chunk_seconds"],
        peak_allocated_gib=t["peak_allocated_gb_total"],
    )


def main():
    out = {}
    for run, ref, label in PAIRS:
        if all(os.path.exists(os.path.join(ROOT, x, "timing.json")) for x in (run, ref)):
            out[run] = dict(label=label, **compare(run, ref))
    json.dump(out, open(os.path.join(ROOT, "ablations.json"), "w"), indent=2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3.8))
    for run, r in out.items():
        ls = ":" if run.startswith("i2av") else "-"
        ax1.plot(r["action_mad_per_chunk"], ls, marker="o", ms=3, label=r["label"])
        ax2.plot(r["psnr_per_frame"], ls, lw=1, label=r["label"])
    ax1.set_xlabel("chunk index")
    ax1.set_ylabel("mean |action - reference|")
    ax1.set_title("Actions vs same-seed reference")
    ax2.set_xlabel("video frame")
    ax2.set_ylabel("PSNR vs reference (dB)")
    ax2.set_title("Video vs same-seed reference (frame 0 = shared start image)")
    ax2.set_ylim(0, 50)
    for a in (ax1, ax2):
        a.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    names = [r["label"] for r in out.values()]
    ax3.barh(names, [r["sec_per_chunk"] for r in out.values()], color="tab:blue", label="run")
    ax3.barh(names, [r["sec_per_chunk_reference"] for r in out.values()], fill=False, edgecolor="k",
             label="reference")
    ax3.set_xlabel("seconds per chunk (chunks >= 1)")
    ax3.set_title("Latency")
    ax3.tick_params(axis="y", labelsize=7)
    ax3.legend(fontsize=7, loc="lower right", framealpha=1)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "ablations.png"), dpi=150)
    for run, r in out.items():
        print(f"{run:15s} {r['label']:32s} act {r['action_mad_mean']:.4f}  psnr {r['psnr_mean']:5.1f} dB  "
              f"{r['sec_per_chunk']:5.1f} s/chunk (ref {r['sec_per_chunk_reference']:.1f})")


if __name__ == "__main__":
    main()
