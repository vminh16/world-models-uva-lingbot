"""Example videos for the Part 2 report: labelled side-by-side clips, small enough to commit.

Every output matches the .gitignore exception results/**/rollouts/*_small.mp4. Inputs are the rollout videos written by
eval_sim.py (mapped to env seeds through eval_log_*.json), the UVA training visualizations (real vs predicted clips) and
the LingBot-VA i2av videos. Needs ffmpeg with drawtext.

Usage:
    python scripts/make_videos.py            # writes results/exp0*/rollouts/*_small.mp4 and results/rollouts/part2_demo_small.mp4
"""

import glob
import json
import os
import subprocess
import tempfile

FONT = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"
FPS = 10


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def seed_video(run_dir, env_seed):
    log = json.load(open(glob.glob(os.path.join(run_dir, "eval_log_*.json"))[0]))
    path = log[f"test/sim_video_{env_seed}"]
    reward = log[f"test/sim_max_reward_{env_seed}"]
    return os.path.join(run_dir, "media", os.path.basename(path)), reward


def label(text, size):
    text = text.replace(":", r"\:").replace("'", "")
    return (f"drawtext=fontfile={FONT}:text='{text}':x=(w-text_w)/2:y=4:fontsize={size}:fontcolor=black:"
            f"box=1:boxcolor=white@0.8:boxborderw=3")


def panels(inputs, out, width, height, fontsize=14, title=None):
    """inputs: list of (video, label). Each panel is scaled to width x height, labelled, padded to the longest clip
    by holding its last frame, then stacked horizontally."""
    args, chains = [], []
    for i, (video, text) in enumerate(inputs):
        args += ["-i", video]
        chains.append(f"[{i}:v]fps={FPS},scale={width}:{height}:flags=neighbor,{label(text, fontsize)},"
                      f"tpad=stop_mode=clone:stop_duration=40[v{i}]")
    stack = "".join(f"[v{i}]" for i in range(len(inputs))) + f"hstack=inputs={len(inputs)}"
    if title:
        stack += f",pad=iw:ih+28:0:28:white,{label(title, 16).replace('y=4', 'y=6')}"
    # shortest real clip length decides nothing: cut at the longest input (tpad holds the others)
    longest = max(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                        v], capture_output=True, text=True).stdout) for v, _ in inputs)
    ff(*args, "-filter_complex", ";".join(chains) + ";" + stack + ",format=yuv420p", "-t", f"{longest:.2f}",
       "-c:v", "libx264", "-crf", "28", "-preset", "slow", "-r", str(FPS), out)


def card(text, out, size=(960, 240), seconds=2.5):
    lines = text.split("\n")
    draws = ",".join(
        f"drawtext=fontfile={FONT}:text='{l.replace(':', chr(92) + ':')}':x=(w-text_w)/2:"
        f"y=(h-{len(lines)}*34)/2+{i}*34:fontsize={26 if i == 0 else 20}:fontcolor=black" for i, l in enumerate(lines))
    ff("-f", "lavfi", "-i", f"color=white:s={size[0]}x{size[1]}:r={FPS}:d={seconds}", "-vf", draws + ",format=yuv420p",
       "-c:v", "libx264", "-crf", "28", out)


def fit(video, out, size=(960, 400)):
    """Scale into a fixed canvas (for concatenation)."""
    w, h = size
    ff("-i", video, "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=neighbor,"
       f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:white,fps={FPS},format=yuv420p", "-c:v", "libx264", "-crf", "28", out)


def main():
    os.makedirs("results/exp01/rollouts", exist_ok=True)
    os.makedirs("results/exp02/rollouts", exist_ok=True)
    os.makedirs("results/exp11/rollouts", exist_ok=True)
    os.makedirs("results/rollouts", exist_ok=True)
    made = []

    # 1. Reproduction: PushT success and near miss; PushT-M same goal, success and wrong rotation
    r = "results/exp01/rollouts"
    out = f"{r}/repro_pusht_and_pushtm_small.mp4"
    panels([(f"{r}/pusht/env100000_seed42_success_r1.000.mp4", "PushT success"),
            (f"{r}/pusht/env100003_seed1_fail_r0.972.mp4", "PushT near miss 0.97"),
            (f"{r}/pusht_multitask/env100000_seed1_success_r1.000.mp4", "PushT-M success"),
            (f"{r}/pusht_multitask/env100000_seed42_fail_r0.720.mp4", "PushT-M rotated 0.72")],
           out, 240, 240, title="Released checkpoints (EXP-01): reward = best coverage / 0.95")
    made.append(out)

    # 2. Fine-tuning: same goal, increasing data
    runs = [("results/exp02/f1_pusht_zeroshot/seed42", "0 demos"),
            ("results/exp02/f3_k10/seed42", "10 demos"),
            ("results/exp02/f2_k25/seed42", "25 demos"),
            ("results/exp02/f3_k50/seed42", "50 demos"),
            ("results/exp01/pusht_multitask_seeded_goal/seed42", "full 247")]
    for env_seed, note in ((100009, "data helps"), (100008, "failure: pushed to the wall")):
        inputs = []
        for d, name in runs:
            v, rew = seed_video(d, env_seed)
            inputs.append((v, f"{name} r={rew:.2f}"))
        out = f"results/exp02/rollouts/finetune_goal{env_seed}_small.mp4"
        panels(inputs, out, 200, 200, fontsize=13,
               title=f"Fine-tuning pusht.ckpt to random goals, goal {env_seed} ({note})")
        made.append(out)

    # 3. UVA imagination: real vs predicted clips, F2 last epoch (4 x 4 grid of 16 clips each)
    vis = "results/exp02/train_f2_k25/vis"
    last = max(int(p.split("_")[-1].split(".")[0]) for p in glob.glob(f"{vis}/predicted_*.mp4"))
    out = "results/exp02/rollouts/uva_real_vs_imagined_small.mp4"
    ff("-i", f"{vis}/real_{last}.mp4", "-i", f"{vis}/predicted_{last}.mp4", "-filter_complex",
       f"[0:v]scale=400:400,{label('real (simulator)', 16)}[a];[1:v]scale=400:400,{label('imagined by UVA', 16)}[b];"
       f"[a][b]hstack,loop=3:size=8,setpts=N/4/TB,format=yuv420p", "-r", "4", "-c:v", "libx264", "-crf", "28", out)
    made.append(out)

    # 4. LingBot-VA i2av
    e = "results/exp11"
    out = f"{e}/rollouts/lingbot_i2av_10chunks_small.mp4"
    panels([(f"{e}/i2av_c10_s0/demo.mp4", "seed 0"), (f"{e}/i2av_c10_s1/demo.mp4", "seed 1"),
            (f"{e}/prompt_left/demo.mp4", "prompt: mug to the left")],
           out, 320, 384, title="LingBot-VA imagines 10 chunks from one image (top: wrist cameras, bottom: head camera)")
    made.append(out)
    out = f"{e}/rollouts/lingbot_40chunks_memory_small.mp4"
    panels([(f"{e}/i2av_c40_s0/demo.mp4", "full memory (36 chunks)"),
            (f"{e}/win8_c40/demo.mp4", "short memory (4 chunks)")],
           out, 320, 384, title="40 chunks open loop: memory length")
    made.append(out)

    # 5. One demo video: title cards + the clips above
    with tempfile.TemporaryDirectory() as t:
        parts = []
        for i, (txt, clip) in enumerate([
            ("UVA: released checkpoints\nPushT 0.974 (paper 0.98)   PushT-M 0.813 on 50 distinct goals", made[0]),
            ("UVA fine-tuned to random goals\n0 / 10 / 25 / 50 demos, about 3 h on one L4 each", made[1]),
            ("Failure case\nthe block is pushed to the wall and never recovered", made[2]),
            ("What UVA imagines\nreal vs predicted clips after fine-tuning", made[3]),
            ("LingBot-VA on one L4\n17 s per chunk of 32 actions, 18 GB GPU memory", made[4]),
            ("LingBot-VA, 40 chunks\nshort memory is faster but loses consistency", made[5]),
        ]):
            c, f = f"{t}/c{i}.mp4", f"{t}/f{i}.mp4"
            card(txt, c, size=(960, 400))
            fit(clip, f)
            parts += [c, f]
        lst = f"{t}/list.txt"
        open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
        out = "results/rollouts/part2_demo_small.mp4"
        ff("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", "28", "-r", str(FPS), out)
        made.append(out)

    for m in made:
        print(f"{os.path.getsize(m) / 1024:7.0f} KB  {m}")


if __name__ == "__main__":
    main()
