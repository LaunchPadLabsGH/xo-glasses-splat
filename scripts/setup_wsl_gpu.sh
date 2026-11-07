#!/usr/bin/env bash
# One-time setup for Gaussian Splatting training on an NVIDIA RTX 50-series GPU
# (Blackwell, sm_120) under WSL2 Ubuntu.
#
# Before running:
#   1. On Windows, install the latest NVIDIA GeForce Game Ready or Studio driver
#      (R570 or newer). GeForce drivers include CUDA support for WSL2; do NOT
#      install any NVIDIA driver inside WSL.
#   2. In PowerShell: `wsl --update`, then open Ubuntu (22.04 or 24.04).
#   3. Clone this repo inside the WSL filesystem (e.g. ~/xo-glasses-splat), not
#      under /mnt/c or /mnt/f, for much faster file access.
#   4. Short on space on C:? WSL stores Ubuntu on C: by default. Move it to
#      another NTFS drive first (see "Windows: running from another drive" in
#      the README). This setup needs about 15 GB.
#
# Usage:  bash scripts/setup_wsl_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

say "Checking the GPU is visible from WSL"
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found. Update the Windows NVIDIA driver and run 'wsl --update' in PowerShell." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

say "Checking free disk space (about 15 GB needed: CUDA toolkit ~5 GB, PyTorch ~4 GB, data and runs)"
avail_gb=$(df -Pk "$HOME" | awk 'NR==2 {print int($4 / 1048576)}')
echo "Free space for this Ubuntu install: ${avail_gb} GB"
if [ "$avail_gb" -lt 15 ]; then
  echo "Less than 15 GB free. Free up space, or move this Ubuntu install to a bigger drive" >&2
  echo "(README: 'Windows: running from another drive'). Set XGS_SKIP_SPACE_CHECK=1 to continue anyway." >&2
  [ "${XGS_SKIP_SPACE_CHECK:-0}" = 1 ] || exit 1
fi
# Do not keep pip's download cache (saves ~3 GB).
export PIP_NO_CACHE_DIR=1

say "Installing system packages (ffmpeg, build tools)"
sudo apt-get update -y
sudo apt-get install -y ffmpeg build-essential python3-venv python3-dev git wget

if ! command -v nvcc >/dev/null || ! nvcc --version | grep -qE "release 12\.(8|9)|release 13"; then
  say "Installing CUDA 12.8 toolkit for WSL (compiler only; the driver comes from Windows)"
  . /etc/os-release
  case "$VERSION_ID" in
    22.04|24.04) ;;
    *) echo "Untested Ubuntu $VERSION_ID; continuing with the WSL-Ubuntu repository" ;;
  esac
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -y
  sudo apt-get install -y cuda-toolkit-12-8
  sudo apt-get clean   # drop downloaded .deb packages (~3 GB)
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
grep -q "cuda-12.8/bin" ~/.bashrc || echo 'export CUDA_HOME=/usr/local/cuda-12.8; export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
nvcc --version | tail -2

say "Creating Python environment (.venv)"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel setuptools

say "Installing PyTorch built for CUDA 12.8 (includes Blackwell sm_120 kernels)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

say "Installing xgs and gsplat"
pip install -e ".[dev]"
# gsplat compiles its CUDA kernels for this GPU on first use.
export TORCH_CUDA_ARCH_LIST="12.0"
grep -q TORCH_CUDA_ARCH_LIST ~/.bashrc || echo 'export TORCH_CUDA_ARCH_LIST="12.0"' >> ~/.bashrc
pip install ninja jaxtyping "gsplat==1.5.3"

say "Sanity check: PyTorch sees the GPU, and gsplat can rasterise on it (first run compiles, ~5-10 min)"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA archs in this PyTorch build:", torch.cuda.get_arch_list())
assert any(a.startswith("sm_120") for a in torch.cuda.get_arch_list()), "PyTorch build lacks sm_120"
from gsplat import rasterization
d = "cuda"
img, alpha, _ = rasterization(
    means=torch.zeros(1, 3, device=d) + torch.tensor([0, 0, 2.0], device=d),
    quats=torch.tensor([[1.0, 0, 0, 0]], device=d), scales=torch.full((1, 3), 0.2, device=d),
    opacities=torch.ones(1, device=d), colors=torch.ones(1, 3, device=d),
    viewmats=torch.eye(4, device=d)[None], Ks=torch.tensor([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]], device=d),
    width=64, height=64)
assert alpha.max() > 0.5, "rasterised Gaussian not visible"
print("gsplat rasterisation OK, centre alpha =", round(alpha[0, 32, 32, 0].item(), 3))
PY

say "Done. Next:"
echo "  . .venv/bin/activate"
echo "  bash scripts/fetch_demo_data.sh          # public demo footage"
echo "  bash scripts/run_demo.sh                 # CPU stages, then GPU training and export"
