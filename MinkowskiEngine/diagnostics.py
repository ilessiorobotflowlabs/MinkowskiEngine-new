import sys
import os
import platform
import subprocess
import shutil


def parse_nvidia_smi():
    sp = subprocess.Popen(
        ["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_dict = dict()
    for item in sp.communicate()[0].decode("utf-8").split("\n"):
        if item.count(":") == 1:
            key, val = [i.strip() for i in item.split(":")]
            out_dict[key] = val
    return out_dict


def print_diagnostics():
    print("==========System==========")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")

    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith(("NAME=", "VERSION=")):
                    print(line.strip())

    print("==========Pytorch==========")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    except ImportError:
        print("torch not installed")

    print("==========NVIDIA-SMI==========")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        print(f"nvidia-smi: {nvidia_smi}")
        for k, v in parse_nvidia_smi().items():
            if "version" in k.lower():
                print(f"  {k}: {v}")
    else:
        print("nvidia-smi not found")

    print("==========NVCC==========")
    nvcc = shutil.which("nvcc")
    if nvcc:
        print(f"nvcc: {nvcc}")
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print(result.stdout.strip())
    else:
        print("nvcc not found")

    print("==========CC==========")
    CC = os.environ.get("CXX", os.environ.get("CC", "c++"))
    cc_path = shutil.which(CC)
    print(f"CC={CC}")
    if cc_path:
        print(f"  path: {cc_path}")
        result = subprocess.run([CC, "--version"], capture_output=True, text=True)
        print(result.stdout.strip())
    else:
        print(f"  {CC} not found")

    print("==========MinkowskiEngine==========")
    try:
        import MinkowskiEngine as ME

        print(f"MinkowskiEngine: {ME.__version__}")
        print(f"  compiled with CUDA: {ME.is_cuda_available()}")
        print(f"  NVCC version: {ME.cuda_version()}")
        print(f"  CUDART version: {ME.cudart_version()}")
    except ImportError:
        print("MinkowskiEngine not installed")


if __name__ == "__main__":
    print_diagnostics()
