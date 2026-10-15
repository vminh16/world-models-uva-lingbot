# Environment: one NVIDIA L4 on Google Cloud

Status: section 1 (machine) and section 2 (`uva`) verified; section 3 (`lingbot`) not yet. Commands that have run successfully are copied, with their date and result, into the
"Reproduce" section of the README. Problems go to [`DEBUG_LOG.md`](DEBUG_LOG.md).

## 1. Virtual machine

Machine actually used (checked 2026-10-14):

| Item | Value |
|---|---|
| Instance | `instance-20261013-044100`, zone `us-west1-a`, standard provisioning |
| Machine type | `g2-standard-4`: 4 vCPU, 16 GB RAM, 1× NVIDIA L4 (23 GB usable) |
| OS / driver | Ubuntu 22.04.5 LTS, NVIDIA driver 580.178.04 (CUDA 13.0 driver API) |
| Boot disk | 300 GB |
| `uva` env | Python 3.9.23, PyTorch 2.1.0 + CUDA 12.1 |
| User / repository | `minhsctn1608`, `~/world-models-uva-lingbot`, conda in `~/miniconda3` |

```bash
gcloud compute ssh minhsctn1608@instance-20261013-044100 --zone=us-west1-a
```

16 GB RAM limits how many simulator processes run at once: the default PushT runner starts `n_train + n_test = 56`
environments in parallel. Use `eval_sim.py --n_envs` (see README, "Changes from upstream").

Original plan, kept for reference:

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

If the local `gcloud` hangs on every command (seen 2026-10-15: `gcloud compute instances list` stalled after the auth
step while the REST API answered), SSH directly with the key gcloud created:
`ssh -i ~/.ssh/google_compute_engine minhsctn1608@<external IP>` (the external IP changes at every start; read it from
`gcloud compute instances describe ... --format='value(networkInterfaces[0].accessConfigs[0].natIP)'` or the console).

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

### 2b. LIBERO for UVA (`~/venvs/uva-libero`, verified 2026-10-14)

Not covered by the upstream environment file; details and the problems met are in ISSUE-006 of `DEBUG_LOG.md`.

```bash
sudo apt-get install -y libnvidia-gl-580-server libegl1 libgl1 libglvnd0   # EGL for headless MuJoCo
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/world-models-uva-lingbot/LIBERO   # runner expects ../LIBERO
~/miniconda3/envs/uva/bin/python -m venv --system-site-packages ~/venvs/uva-libero
printf "numpy==1.26.4\ntorch==2.1.0\nmujoco==3.1.6\n" > /tmp/c.txt
export PATH=~/venvs/uva-libero/bin:$PATH   # use the venv's cmake (<4) for egl_probe
~/venvs/uva-libero/bin/python -m pip install -c /tmp/c.txt "cmake<4" robosuite==1.4.0 bddl==1.0.1 easydict future h5py \
  termcolor tensorboardX imageio imageio-ffmpeg psutil
~/venvs/uva-libero/bin/python -m pip install --no-build-isolation egl_probe   # CMake 4 rejects its CMakeLists
~/venvs/uva-libero/bin/python -m pip install --no-deps robomimic==0.3.0
# robomimic imports mujoco_py at module level but only uses mujoco_py.builder.MujocoException: add a stand-in package
# site-packages/mujoco_py/{__init__.py: 'from . import builder', builder.py: 'class MujocoException(Exception): pass'}
~/venvs/uva-libero/bin/python -m pip install --no-deps -e ~/world-models-uva-lingbot/LIBERO
# pytorch3d (only pytorch3d.transforms is used, pure PyTorch): copy __init__.py, common/, transforms/ of tag v0.7.5
# from github.com/facebookresearch/pytorch3d into the venv's site-packages/pytorch3d/
# write ~/.libero/config.yaml first (benchmark_root, bddl_files, init_states, datasets, assets), else LIBERO prompts
cd uva && MUJOCO_GL=egl ~/venvs/uva-libero/bin/python eval_sim.py -c checkpoints/libero10.ckpt -o <dir> ...
```

Data: `uva/data/libero_10/*.hdf5` (10 task files, 13 GB), `uva/data/pusht/pusht_cchi_v7_replay.zarr` (206 episodes),
`uva/data/pusht_multitask` (247 episodes).

## 3. LingBot-VA environment (`~/venvs/lingbot`, verified 2026-10-15)

From the upstream README ([Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va)): Python 3.10.16, PyTorch 2.9.0,
CUDA 12.6. The driver on the VM (580, CUDA 13.0) runs the cu126 wheels. Conda on the VM cannot create environments
(ISSUE-012), so the environment is a [uv](https://docs.astral.sh/uv/) venv; the whole setup is `scripts/exp11_setup.sh`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv is missing
uv venv --python 3.10.16 ~/venvs/lingbot && source ~/venvs/lingbot/bin/activate
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib \
    ftfy easydict huggingface_hub hf_transfer imageio imageio-ffmpeg
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('robbyant/lingbot-va-posttrain-robotwin', local_dir='$HOME/models/lingbot-va-posttrain-robotwin')"
```

`flash-attn` is not installed (it builds from source for hours on 4 vCPUs); inference uses `attn_mode="torch"` (PyTorch
SDPA), which needs the import fix in the README table "Changes from upstream" (ISSUE-013). Exact package versions:
`results/exp11/pip_freeze.txt`.

Checkpoints on Hugging Face: `robbyant/lingbot-va-base`, `robbyant/lingbot-va-posttrain-robotwin`,
`robbyant/lingbot-va-posttrain-libero-long`, 24.4 GB each (transformer 10.2 GB, T5 text encoder 11.4 GB, VAE 2.8 GB; the
download took about 2 min on the VM). `attn_mode` in `<model>/transformer/config.json` is already `"torch"` in the
RoboTwin checkpoint, and `wan_va_server.py` passes `attn_mode="torch"` explicitly in any case (`"flex"` is for training).
Offload mode (`--enable_offload true`, upstream default in `shared_config.py` is off) keeps the VAE and the text encoder
on the CPU. Loading then holds T5, VAE and transformer on the CPU at the same time; a 32 GB swap file was added before the
first run (`sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`,
not persistent across reboots), but the measured peak was 4.8 GB RAM and 0.6 GB swap (safetensors are memory-mapped).

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
