# Environment: one NVIDIA L4 on Google Cloud

Status: **plan, not yet verified**. Commands that have run successfully are copied, with their date and result, into the
"Reproduce" section of the README. Problems go to [`DEBUG_LOG.md`](DEBUG_LOG.md).

## 1. Virtual machine

| Item | Choice | Why |
|---|---|---|
| Machine type | `g2-standard-8` (8 vCPU, 32 GB RAM, 1× L4 24 GB attached) | the smallest G2 type; the L4 comes with the machine type |
| Region | one with L4 quota, for example `asia-southeast1` or `us-central1` | request the "NVIDIA L4 GPUs" quota for that region first |
| Image | Deep Learning VM with CUDA 12 (project `deeplearning-platform-release`) | NVIDIA driver and CUDA preinstalled |
| Boot disk | 200 GB | datasets, checkpoints, two conda environments |
| Provisioning | Spot for long runs (much cheaper, can be stopped by Google), standard for interactive work | save checkpoints often on Spot |

The image family names change over time, so list them first and pick a CUDA 12 family:

```bash
gcloud compute images list --project deeplearning-platform-release --no-standard-images --filter="family~cu12" \
  --format="value(family)" | sort -u
```

Then create the machine (replace the placeholders):

```bash
gcloud compute instances create uva-l4 \
  --zone=<ZONE> --machine-type=g2-standard-8 \
  --image-project=deeplearning-platform-release --image-family=<FAMILY> \
  --boot-disk-size=200GB --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True
# add --provisioning-model=SPOT for long training runs
gcloud compute ssh uva-l4 --zone=<ZONE>
nvidia-smi        # must show one NVIDIA L4 with about 23 GB
```

Stop the machine when it is idle (`gcloud compute instances stop uva-l4 --zone=<ZONE>`); the disk is still billed.

## 2. UVA environment (`uva`)

From the upstream README ([ShuangLI59/unified_video_action](https://github.com/ShuangLI59/unified_video_action)):

```bash
cd uva
conda install mamba -c conda-forge
mamba env create -f conda_environment.yml
conda activate uva                      # name as defined in conda_environment.yml
python unified_video_action/utils/download.py     # pretrained VAE and MAR, needed for training
```

Data and checkpoints (download links are in the upstream README): PushT (Diffusion Policy), PushT-M (authors), Libero10,
ToolHang, into `uva/data/`; `pusht.ckpt`, `pusht_multitask.ckpt`, `libero10.ckpt` into `uva/checkpoints/`.

## 3. LingBot-VA environment (`lingbot`)

From the upstream README ([Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va)): Python 3.10.16, PyTorch 2.9.0,
CUDA 12.6.

```bash
conda create -n lingbot python=3.10.16 -y && conda activate lingbot
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
```

Checkpoints on Hugging Face: `robbyant/lingbot-va-base`, `robbyant/lingbot-va-posttrain-robotwin`,
`robbyant/lingbot-va-posttrain-libero-long`. Before inference set `attn_mode` in `<model>/transformer/config.json` to
`"torch"` or `"flashattn"` (`"flex"` is for training and fails at evaluation). Offload mode keeps the VAE and text encoder on
the CPU; the upstream memory figures (about 18 GB for i2av, about 24 GB for single-GPU RoboTwin evaluation) assume it.

## 4. Simulators without a screen

| Simulator | Used by | Headless setting |
|---|---|---|
| PushT (2D physics, pymunk) | UVA | renders frames to arrays; set `SDL_VIDEODRIVER=dummy` if pygame asks for a display |
| LIBERO (MuJoCo) | UVA, LingBot-VA | `MUJOCO_GL=egl` (rendering on the GPU, no display) |
| RoboTwin 2.0 (SAPIEN) | LingBot-VA | Vulkan on the NVIDIA driver; optional, heaviest |

Videos and figures are written to `results/`. To look at them: VS Code Remote-SSH (opens mp4/png and forwards the
TensorBoard port), or copy them with `gcloud compute scp --recurse uva-l4:~/world-models-uva-lingbot/results ./results`.

## 5. Long runs

Run anything longer than 10 minutes in `tmux` or with `nohup ... > results/<exp>/run.log 2>&1 &`, and log GPU memory next to
it: `nvidia-smi --query-gpu=memory.used --format=csv -l 5 > results/<exp>/vram.csv`.
