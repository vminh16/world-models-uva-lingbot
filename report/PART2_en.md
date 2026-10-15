# Part 2 — Research Paper Reading and Implementation: World Models

**Code:** `github.com/vminh16/world-models-uva-lingbot` · **Part 2 demo video:**
[`results/rollouts/part2_demo_small.mp4`](../results/rollouts/part2_demo_small.mp4)

## 2.1 Problem

A *world model* learns the dynamics of the environment: given past observations and actions, it predicts the next
observation, i.e. it approximates *p(oₜ₊₁ | o≤ₜ, a≤ₜ)*. For a robot this is exactly the knowledge a policy needs: how an
action will change the scene. Two families of methods existed before these papers:
- **Reactive policies** (Diffusion Policy, OpenVLA, π0) map images directly to actions. They are fast, but they do not
  model scene dynamics, tend to memorize trajectories, and see only the last few frames.
- **Video-first methods** (UniPi) generate future frames and then infer actions from them. They capture dynamics, but they
  are slow because they synthesize pixels, and video errors propagate into the actions.

Both papers address the same question: **how can video prediction make a policy better while staying fast enough for the
control loop?**

## 2.2 UVA — Unified Video Action Model (RSS 2025)

**Main ideas.**
- **Joint latent.** Past images, past actions and partially masked future frames are tokenized and fused by one
  transformer (about 0.5B parameters) into a joint latent *Z*.
- **Decoupled decoding.** Two small diffusion heads read *Z*: one generates future frames, the other the next action
  chunk. Training minimizes *L = L_action + L_video*; at control time only the action head runs, so no video is generated.
- **Masked training.** Randomly masking video or actions in the input and output lets one set of weights act in several
  roles.

**Contributions to robot learning.**
1. **It separates the benefit of video from its cost.** Video is used only during training, to shape *Z*. The policy never
   renders frames at run time, so it is even faster than Diffusion Policy in simulation (0.23 s vs 0.36–0.50 s per
   inference).
2. **Video supervision reduces overfitting.** UVA keeps its performance as the history grows, while Diffusion Policy
   degrades. Removing the video loss lowers PushT-M from 0.88 to 0.46.
3. **One model, several roles:** policy, video generator, forward and inverse dynamics model, and policy combined with a
   planner. This opens the way to incomplete data, such as videos without action labels.

## 2.3 LingBot-VA — Causal World Modeling for Robot Control (RSS 2026)

**Main ideas.**
- **One interleaved causal sequence.** Video latents *z* and actions *a* are interleaved in time, *[zₜ, aₜ, zₜ₊₁, aₜ₊₁, …]*,
  under a causal attention mask: each token attends only to earlier tokens.
- **Predict, then act.** For each chunk the model first predicts the next video latents, *ẑₜ₊₁ ~ p(· | z≤ₜ, a<ₜ)*. It then
  infers the actions that lead there (inverse dynamics), *aₜ ~ g(· | ẑₜ₊₁, z≤ₜ, a<ₜ)*. Both are trained with flow matching.
- **Asymmetric Mixture-of-Transformers (5.3B).** A large video stream, initialized from the Wan2.2-5B video model, and an
  action stream four times narrower share the same depth and exchange information through joint attention.
- **Long-term memory through a KV cache.** Because attention is causal, the keys and values of past tokens never change
  when new tokens arrive, so they are stored and reused. Each step only computes the new tokens, yet still attends to the
  whole history (up to 36 chunks on RoboTwin). On the robot, the tokens of the **real** camera image replace the predicted
  ones in the cache, which keeps the prediction anchored to reality.
- **Noisy latent augmentation.** During training the video history is randomly mixed with noise, so the action stream
  learns to read partially denoised latents. At run time video denoising can stop early (5 steps).

**Contributions to robot learning.**
1. **A causal video world model that runs in closed loop.** At every step the model predicts the future and is corrected
   by the real observation, so it can recover from errors during execution. The causal formulation matters: on RoboTwin
   2.0 (50 tasks), a bidirectional variant reaches 81.5% against 92.9%.
2. **Long-term memory for partially observable tasks.** Chunk-based policies decide from the last few frames only.
   LingBot-VA remembers what it has done, for example how many times a plate was wiped or which box was already opened,
   and clearly outperforms π0.5 on such tasks.
3. **Large-scale video pretraining transfers to bimanual manipulation with little data.** With 50 demonstrations per
   real-world task, it beats π0.5 by more than 20% on hard long-horizon, high-precision and deformable-object tasks.
4. **A 5B model inside the control loop:** about 0.5 s per step on one 48 GB GPU.

## 2.4 Comparison

**Table 2.** Two ways of using video for robot control.

| | UVA | LingBot-VA |
|---|---|---|
| Video and action | Joint latent, two independent decoders | Interleaved causal sequence: predict video, then infer actions by inverse dynamics |
| Video at control time | **Skipped**: implicit world model, used only in training | **Predicted explicitly** at every step, denoising can stop early |
| Memory | Short fixed window (PushT: 4 frames over 1.6 s), re-encoded each call | KV cache, up to 72 latent frames (36 chunks) |
| Scale, data | 0.5B, trained on each benchmark's data | 5.3B, initialized from Wan2.2-5B, pretrained on 16k hours of robot data |
| Speed reported | 0.23 s (simulation, 100 denoising steps); 95 ms (real robot, 16 steps) | ≈ 0.5 s per step (RTX 5880 Ada 48 GB; 5 video / 10 action steps) |
| **Measured on 1× L4 24 GB** | **0.14–0.48 s, 1.9 GB** | **3.8–17 s per 32-action chunk, 18 GB** (with offloading) |

Speeds are not directly comparable across rows: the GPUs differ, I offload part of LingBot-VA to the CPU, its paper
counts 16 actions per step while my chunks hold 32, and open-loop generation does not include encoding new camera images.

The core difference is whether **video must be predicted explicitly at control time**, and the two papers reach
different answers. UVA skips video and still performs well. LingBot-VA's ablation shows that using video only as a
training signal (close to UVA's design) drops its success from 92.9% to 48.3%. The two results are not directly
comparable, since each ablation concerns its own architecture. The experiments below shed some light on this question
and on its cost in time and memory.

## 2.5 Setup, metrics and experiments

All experiments ran on one Google Cloud VM with 1× NVIDIA L4 (24 GB), 4 vCPUs and 16 GB RAM, about 17 GPU-hours in total.
The code is a fork of the official repositories; every change is a new option whose default keeps the original behavior.

**Metrics.** *Coverage* is the fraction of the goal region covered by the T block. The **mean score** (PushT, PushT-M)
is the highest coverage in an episode divided by 0.95 (capped at 1), averaged over episodes. **Success** is the fraction
of episodes whose highest coverage reaches 95%; for LIBERO it is the task completion rate. The paper labels its PushT
numbers "success rates", but its 0.98 matches my mean score (0.974), not my success rate (0.73), so I compare mean scores.
On PushT and PushT-M each configuration runs on 50 fixed test cases with 3 policy seeds; ± is the standard deviation over
the seeds. Two models are compared pairwise on the same cases, with a 95% bootstrap confidence interval over the 50 cases.
LIBERO-10 uses 20 episodes per task with one seed and a Wilson 95% interval. Fine-tuning the 5.3B LingBot-VA, or evaluating it in
closed loop in RoboTwin at 17 s per chunk, is beyond one 24 GB GPU, so I study it through open-loop generation, cost and
ablations. Without a simulator, each variant is compared with the default run of the same seed (same initial noise) through the
**action deviation**, the mean |a − a_ref| in normalized action space. The reference is the deviation caused by changing
only the seed (0.016).

**Table 3.** Experiments.

| ID | Question | Method |
|---|---|---|
| EXP-01, 05 | Do the authors' checkpoints reach the paper's numbers? | PushT, PushT-M, LIBERO-10 |
| EXP-02, 03 | Can UVA learn new goals from few demos? Does the video loss help? | Fine-tune `pusht.ckpt` (one fixed goal) on 10/25/50 PushT-M demos; one run without video loss |
| EXP-07 | How many denoising steps are needed? | UVA with 100, 50, 20, 10, 5 steps |
| EXP-11 | Does LingBot-VA run on 24 GB? How do steps, memory and the prompt matter? | Open-loop generation of 10–40 chunks from one RoboTwin image, with ablations |

**A bug in the original evaluation code that biases results.** In PushT-M the goal is drawn from the global random
generator instead of the test case's seed. The parallel simulators are forked with the same random state, so **every
episode in a batch gets the same goal**. With the default configuration (56 simulators in one batch), all 50 test cases
share a single goal. I added an option that ties the goal to each test case's seed.

## 2.6 Results

**Table 4.** Reproduction.

| Model | Environment | Measured | Paper |
|---|---|---|---|
| UVA | PushT | mean score **0.974 ± 0.016** (success 0.73) | 0.98 |
| UVA | LIBERO-10 | success **0.865** [0.811, 0.906] | 0.90 |
| UVA | PushT-M: original code (5 goals per run) / 50 distinct goals | 0.787 ± 0.008 / **0.813 ± 0.030** | 0.88 |
| LingBot-VA | RoboTwin, open-loop generation | 18 GB GPU memory, 17 s per 32-action chunk | – |

PushT and LIBERO-10 match the paper within statistical error. On 50 distinct goals, PushT-M reaches 0.813, 0.07 below
the paper's 0.88. Two factors make the paper's number hard to compare: with the default configuration it was most likely
measured on a single goal, and single-goal scores range from 0.68 to 0.92, so one goal adds a lot of variance; and the
paper reports the best checkpoint, selected on the evaluation score itself, which biases the number upwards. LingBot-VA fits in 24 GB when the text encoder and the VAE are offloaded to
the CPU; without offloading I estimate it exceeds 24 GB, but I did not try.

**Table 5.** Fine-tuning UVA on PushT-M. All runs start from `pusht.ckpt` and use the same budget: batch 16, about 3,030
steps (about 3 GPU-hours), AdamW with learning rate 1e-4, 200 warm-up steps and cosine decay, EMA, fp16; the demo subsets
are nested (10 ⊂ 25 ⊂ 50). The last two columns come from the final frames of the 30 recorded episodes (first 10 goals ×
3 seeds, 96-px frames, where a bar of the T is 5.6 px thick).

| Demos | Training loss | Mean score | Success | Final position error (median) | Ends placed (≤ 2.8 px and ≤ 15°) |
|---|---|---|---|---|---|
| 0 (not fine-tuned) | – | 0.180 | 0% | 21.3 px | 0/30 |
| 10 | video + action | 0.421 | 0% | 8.8 px | 3/30 |
| 25 | **action only** | 0.454 | 1.3% | 9.4 px | 0/29 |
| 25 | video + action | 0.517 | 0.7% | 4.2 px | 2/30 |
| 50 | video + action | **0.593** | 1.3% | 5.0 px | 4/30 |
| 247 (authors' model) | video + action | 0.813 | 25% | 1.4 px | 12/30 |

**Table 6.** Denoising steps and inference time (batch 1, one L4). UVA scores use one policy seed; no paired difference
from 100 steps is significant (all confidence intervals include 0).

| Model | Steps | Quality | Time |
|---|---|---|---|
| UVA (8 actions per call) | 100 (default) → 20 → 5 | mean score 0.957 → 0.972 → 0.995 | 0.48 → 0.19 → **0.14 s** |
| LingBot-VA (32 actions per chunk) | 25/50 (default) → 10/20 → 5/10 | action deviation 0.002 → 0.015 (seed change: 0.016) | 17.1 → 7.1 → **3.8 s** |

![Figure 2](figures/fig2_part2.png)

**Figure 2.** (a) UVA mean score on PushT-M versus the number of fine-tuning demos; error bars are 95% confidence intervals
over 50 goals. (b) LingBot-VA generation time per chunk over 40 consecutive chunks, for different memory and denoising
settings.

*Explanation.* (a) More demos help steadily: the mean score rises from 0.18 to 0.42, 0.52 and 0.59 with 10, 25 and 50
demos, and both increases are significant in paired comparisons over the same goals (+0.096 [0.027, 0.164] and +0.076
[0.005, 0.147]), even though the unpaired intervals in the figure overlap. At 25 demos, removing the
video loss (orange square) lowers the score by 0.063 [0.003, 0.124], to the level of 10 demos, even though both runs use
the same compute and the action-only run receives more action updates (every step instead of about 3 in 5). This agrees
with the regularizing role of video claimed by UVA, but the evidence is weak: each configuration was trained once, and the
gap is about the size of the spread between policy seeds. Success stays near 0 at every data level. (b) With full memory
the time per chunk grows linearly from 16 s to 24 s, as expected when attention cost scales with the cache length, and
flattens once the cache is full (chunk 36). Capping the cache at 4 chunks keeps the time constant. Fewer denoising steps
give the largest speed-up (2.4× at 10/20 steps, 4.5× at 5/10), with action deviations within the seed-to-seed variation.

## 2.7 Success and failure analysis

![Figure 3](figures/fig3_part2.png)

**Figure 3.** (a) Last frame of UVA on three PushT-M goals (rows) for four models (columns), policy seed 42; the goals
were chosen to show the three typical outcomes. r is the highest reward reached during the episode (r = 1 means
success), so it can differ from the last frame. (b) LingBot-VA imagining 40 consecutive chunks from one image, with no real observation
fed back; top row: full cache, bottom row: cache limited to the last 4 chunks.

*Explanation.*
- **Before fine-tuning, UVA ignores the goal in the image** and pushes the block towards the old fixed PushT goal: its
  final position is 11.3 px from that old goal but 21.7 px from the episode's goal (median, 30 episodes).
- **After fine-tuning, the model brings the block near the goal but rarely places it precisely** (Table 5). With 25–50
  demos the median position error is 4–5 px, close to one bar width, and only 2–4 of 30 episodes end both within half a
  bar and within 15° of the goal pose (authors' model: 12/30). This is why the mean score triples while success stays
  near 0.
- **A recurring local optimum is a rotation of about 120°.** The T has three arms of similar length, so a T rotated by
  about 120° around the goal centre still covers about two thirds of the goal (r ≈ 0.7), as on goal 2 (r = 0.72). Of the
  episodes that end within half a bar of the goal, about half are rotated in this way (4 of 8 at 25 demos, 4 of 10 at 50),
  and the authors' model shows it too (5 of 17).
- **The worst failures occur when the block is pushed against a wall or into a corner** (goal 3; goal 2 for the
  action-only model). The model never recovers, most likely because no demo contains such a state.
- **In my 40-chunk run, LingBot-VA stays coherent for about 10 chunks:** the arm reaches, grasps the mug and moves it towards the rack. After
  that the scene drifts even with a full cache (the rack and the arms fade away). With a 4-chunk cache, the gripper breaks
  into several fragments from about chunk 15.
- **The prompt has a weak effect:** changing it to "…place it on the left side of the table" shifts the actions (deviation 0.028), but the
  video still places the mug on the rack (one trial).
- **On LIBERO-10,** the two weakest tasks (0.50 and 0.70, 20 episodes each) both put two objects into one place (a
  basket, a stove); every other task reaches at least 0.85.

## 2.8 Discussion: reproduction versus paper claims

**Implicit or explicit: the value of video depends on the setting.** In UVA's own ablation, removing video costs 0.53
on PushT and 0.42 on PushT-M but only 0.04 on LIBERO-10; in my fine-tuning from a checkpoint that already learned video it
costs 0.063, and in LingBot-VA's ablation 45 points. Task length does not explain the differences, since LIBERO-10
consists of long tasks. One hypothesis is that the video loss matters most when the latent has not yet learned scene
dynamics, and less when it has (my fine-tuning). It fits PushT and PushT-M but not LIBERO-10, where the action-only model
(0.86) is close to UVA (0.90), possibly because both are near the ceiling. LingBot-VA's ablation is tied to its
architecture, in which a small action stream is designed to rely on the predicted video latents. As for cost, video
generation takes about 6 of the 17 s per chunk in LingBot-VA's default configuration; the much larger gap between the two
models in Table 2 also reflects a 10× difference in size (0.5B vs 5.3B), so it cannot be attributed to video prediction
alone. Both models use video in a fixed way and cannot decide *when* imagining is worth it.

**Denoising steps: default configurations are not model limits.** LingBot-VA reports that reducing video denoising from
25 to 15 steps keeps success unchanged and uses 5 steps on the real robot, while its repository defaults to 25/50. My
open-loop result agrees: at 10/20 steps the actions are almost identical to the default. UVA uses 100 steps in
simulation, yet 5 steps do not lower its PushT score and run 3.5× faster (one seed, single-goal task; untested on harder
tasks). Speed comparisons between methods should therefore be made at equal quality, not at default settings.

**The KV cache gives long memory, but at a cost, and open-loop imagination drifts quickly.** The cache removes
re-encoding of the history, but the time per step still grows linearly with its length. Truncating the cache breaks
consistency, and even with a full cache my open-loop run drifts after about 10 chunks. The long-horizon success reported in the
paper most likely relies on real image tokens continuously replacing imagined ones in the cache; I did not run closed
loop, so I could not verify this. In this run (one image, one prompt, one seed), the model on its own was not a reliable long-horizon simulator.

**The evaluation protocol is part of the result.** With the default configuration, the shared-goal bug makes the PushT-M
number most likely reflect a single goal, and the paper reports a checkpoint selected on the evaluation score itself. UVA's
table says "success rate", while its PushT-M description says "average reward"; on the same model the two metrics differ
widely (PushT: 0.97 vs 0.73). Multi-goal benchmarks should check the diversity of their test cases directly and report
both metrics.

## 2.9 Main limitation and how to address it

**Fine-tuned UVA brings the block near the goal but cannot place it precisely.** At every data level (10–50 demos)
success is only 0–1.3% (fully trained model: 25%). The block ends a median 4–9 px from the goal, and only 0–4 of 30
episodes end both within half a bar and within 15° (Table 5); near the goal, a 120° rotation is a frequent local optimum
of coverage. Three plausible causes: (1) 25–50 demos cover the space of goal positions and angles sparsely; (2) offline
imitation provides no data on how to recover once the block is off course, for example against a wall or after a wrong
rotation; (3) the video loss is averaged uniformly over all tokens of a frame, so the small details that decide precision
(the robot's tip, the block's edges) contribute little to it. They also looked blurred in the predicted frames I
inspected, but I did not measure this.

**Remedies, from cheapest to most expensive.**
1. **Hindsight goal relabeling.** Demos end exactly at the goal, so their final pose adds nothing new. Instead, use the
   block's pose at *intermediate* times as the goal for the preceding part of the trajectory, and redraw the goal in the
   images. Each demo then yields many (trajectory, goal) pairs with more diverse goal angles, and re-rendering needs only a
   CPU.
2. **Fine-tuning in the loop with the simulator** (DAgger, or RL), so the model learns from the off-course states it
   causes itself. For RL, a reward on the pose error (position and angle) would avoid the 120° local optimum that
   coverage has.
3. **Region-weighted loss:** higher weight on tokens covering the robot and the manipulated object.

**Future direction: an adaptive world model.** The discussion above suggests a model that runs like UVA by default
(skipping video, reacting quickly) and switches to explicit video prediction like LingBot-VA only when uncertainty is high:
before contact, during precise manipulation, or when the previous prediction disagrees with the real image. In
LingBot-VA's default configuration video takes about a third of each chunk's time; this is what could be saved on easy
steps, provided the action stream is trained to work without predicted video.

**Limitations of this study.** Each fine-tuning configuration was trained once, so the conclusion that the video loss
helps has a confidence interval close to 0. LingBot-VA was run in open loop only, and its ablations use one seed, so the
statements about memory and the prompt are qualitative.
