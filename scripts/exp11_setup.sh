#!/bin/bash
# EXP-11 L1: lingbot venv + checkpoint download. Log: results/exp11/setup.log
# conda on the VM cannot create envs (broken plugins and a corrupted package cache in base), so the environment is a
# uv venv with the Python version the LingBot-VA README asks for.
set -x
set -e
export PYTHONNOUSERSITE=1
cd ~/world-models-uva-lingbot
mkdir -p results/exp11
date
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -d ~/venvs/lingbot ] || uv venv --python 3.10.16 ~/venvs/lingbot
source ~/venvs/lingbot/bin/activate
python -V
[ "$(which python)" = "$HOME/venvs/lingbot/bin/python" ] || { echo 'lingbot venv not active'; exit 1; }
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict huggingface_hub hf_transfer imageio imageio-ffmpeg
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv pip freeze > results/exp11/pip_freeze.txt
date
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('robbyant/lingbot-va-posttrain-robotwin', local_dir='$HOME/models/lingbot-va-posttrain-robotwin')
print('downloaded to', p)"
du -sh ~/models/lingbot-va-posttrain-robotwin
date
echo SETUP_DONE
