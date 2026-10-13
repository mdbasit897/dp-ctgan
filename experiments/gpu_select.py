"""Give each experiment process its own GPU on a multi-GPU machine.

Called by common.py before torch is imported. Every process claims the GPU that the fewest
running experiment processes have claimed (ties: least memory in use) and restricts itself to it
via CUDA_VISIBLE_DEVICES, so N jobs started at the same time spread over N GPUs. Inside the
process that GPU is cuda:0, which is what dp_ctgan_pipeline and synthcity use.

  * Set CUDA_VISIBLE_DEVICES yourself to choose a GPU manually; it is then left alone.
  * DPCTGAN_AUTO_GPU=0 disables the automatic choice.
  * Machines without nvidia-smi or with a single GPU are left unchanged.
"""
import fcntl
import os
import subprocess
import sys
import tempfile

CLAIM_DIR = os.path.join(tempfile.gettempdir(), "dpctgan_gpu_claims")


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, check=True).stdout
    return [(int(i), float(m)) for i, m in (line.split(",") for line in out.strip().splitlines())]


def assign():
    if "CUDA_VISIBLE_DEVICES" in os.environ or os.environ.get("DPCTGAN_AUTO_GPU", "1") == "0":
        return None
    try:
        gpus = _gpus()
    except Exception:
        return None
    if len(gpus) < 2:
        return None
    os.makedirs(CLAIM_DIR, exist_ok=True)
    with open(os.path.join(CLAIM_DIR, ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        claims = {i: 0 for i, _ in gpus}
        for name in os.listdir(CLAIM_DIR):
            if not name.startswith("gpu"):
                continue
            gpu, pid = map(int, name[3:].split("."))
            if _alive(pid) and gpu in claims:
                claims[gpu] += 1
            elif not _alive(pid):
                os.remove(os.path.join(CLAIM_DIR, name))
        gpu = min(gpus, key=lambda g: (claims[g[0]], g[1]))[0]
        open(os.path.join(CLAIM_DIR, f"gpu{gpu}.{os.getpid()}"), "w").close()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[gpu] process {os.getpid()} -> GPU {gpu} "
          f"({claims[gpu]} other experiment process(es) on it)", file=sys.stderr, flush=True)
    return gpu
