#!/usr/bin/env bash
# Run ON THE VM (Ubuntu 22.04). Installs the NVIDIA driver, Python env and the repo.
#   bash setup_vm.sh            # first run installs the driver and reboots
#   bash setup_vm.sh            # run again after the reboot to finish
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/YOUR_USER/dp-ctgan.git}"   # or copy the folder with gcloud compute scp
REPO_DIR="$HOME/dp-ctgan"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo ">>> Installing NVIDIA driver with Google's installer (VM will reboot)"
  sudo apt-get update -y && sudo apt-get install -y python3 curl
  curl -fsSL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o /tmp/cuda_installer.pyz
  sudo python3 /tmp/cuda_installer.pyz install_driver
  echo ">>> If the VM did not reboot automatically: sudo reboot ; then run this script again."
  exit 0
fi
nvidia-smi

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git tmux htop unzip
if [[ ! -d "$REPO_DIR" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
# uv provides the same Python 3.11 the pipeline was tested with (Ubuntu 22.04 ships 3.10)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt       # PyPI torch wheels already include the CUDA 12 runtime
uv pip install google-cloud-bigquery db-dtypes pyarrow kaggle imbalanced-learn
python - <<'PY'
import torch, opacus, sklearn
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("opacus", opacus.__version__, "sklearn", sklearn.__version__)
PY
echo ">>> Setup complete. Activate with: cd $REPO_DIR && source .venv/bin/activate"
