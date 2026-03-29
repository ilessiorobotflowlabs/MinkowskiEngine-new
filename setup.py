"""
MinkowskiEngine build script.

Supports CUDA and CPU-only builds via torch.utils.cpp_extension.

Usage:
    pip install -e .                           # auto-detect CUDA
    pip install -e . --config-settings="--build-option=--cpu_only"
    MAX_JOBS=8 pip install -e .                # control parallelism

Environment variables:
    FORCE_CUDA=1        Force CUDA build even if torch.cuda.is_available() is False
    CPU_ONLY=1          Force CPU-only build
    CUDA_HOME=/path     Override CUDA toolkit path
    MAX_JOBS=N          Max parallel compilation threads (default: 12)
"""

import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import torch

if sys.platform == "win32":
    raise ImportError("Windows is currently not supported.")


HERE = Path(__file__).parent.resolve()
SRC_PATH = HERE / "src"
MAX_COMPILATION_THREADS = 12


def find_version():
    init_file = (HERE / "MinkowskiEngine" / "__init__.py").read_text()
    match = re.search(r'^__version__ = ["\']([^"\']*)["\']', init_file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


def find_blas_libraries():
    """Find BLAS library without using deprecated numpy.distutils."""
    libraries = []

    # Check for common BLAS libraries via pkg-config or environment
    blas_env = os.environ.get("BLAS", "").lower()
    if blas_env:
        if blas_env == "mkl":
            libraries.append("mkl_rt")
            return libraries, ["-DUSE_MKL"], ["-DUSE_MKL"]
        else:
            libraries.append(blas_env)
            return libraries, [], []

    # Try to detect from numpy config (works with numpy 1.x and 2.x)
    try:
        import numpy as np

        # NumPy 2.x: use numpy.show_config(mode='dicts')
        if hasattr(np, "__config__") and hasattr(np.__config__, "blas_opt_info"):
            info = np.__config__.blas_opt_info
            if "libraries" in info:
                return info["libraries"], [], []

        # Try numpy.show_config for detection
        # Fall through to manual detection
    except Exception:
        pass

    # Manual detection: try linking against common BLAS libraries
    blas_candidates = ["openblas", "blas", "flexiblas"]
    for blas in blas_candidates:
        # Check if library exists via ldconfig or common paths
        lib_paths = [
            f"/usr/lib/lib{blas}.so",
            f"/usr/lib/x86_64-linux-gnu/lib{blas}.so",
            f"/usr/lib64/lib{blas}.so",
        ]
        if any(Path(p).exists() for p in lib_paths):
            libraries.append(blas)
            return libraries, [], []

    # Last resort: just use openblas and hope for the best
    warnings.warn(
        "Could not auto-detect BLAS library. Defaulting to openblas. "
        "Set BLAS=openblas (or mkl/atlas) environment variable to override."
    )
    libraries.append("openblas")
    return libraries, [], []


# Determine CPU_ONLY vs CUDA
CPU_ONLY = os.environ.get("CPU_ONLY", "0") == "1"
FORCE_CUDA = os.environ.get("FORCE_CUDA", "0") == "1"

# Legacy argv-based flags (for backward compat with old install instructions)
if "--cpu_only" in sys.argv:
    CPU_ONLY = True
    sys.argv.remove("--cpu_only")
if "--force_cuda" in sys.argv:
    FORCE_CUDA = True
    sys.argv.remove("--force_cuda")

if not torch.cuda.is_available() and not FORCE_CUDA:
    warnings.warn(
        "torch.cuda.is_available() is False. Building CPU_ONLY. "
        "Set FORCE_CUDA=1 to override."
    )
    CPU_ONLY = True

if FORCE_CUDA:
    CPU_ONLY = False
    print("--- FORCE_CUDA: building with CUDA support ---")

# Compiler flags
CC_FLAGS = []
NVCC_FLAGS = []
libraries = []
include_dirs = [
    str(SRC_PATH),
    str(SRC_PATH / "3rdparty"),
    "/usr/include/x86_64-linux-gnu",  # cblas.h location on Ubuntu/Debian
    "/usr/include",
]

if not CPU_ONLY:
    libraries.append("cusparse")

if sys.platform != "win32":
    CC_FLAGS += ["-fopenmp"]

if sys.platform == "darwin":
    CC_FLAGS += ["-stdlib=libc++", "-std=c++17"]

NVCC_FLAGS += ["--expt-relaxed-constexpr", "--expt-extended-lambda"]

# Redirect nvcc temp files to the large NVMe mount to avoid filling root filesystem.
# nvcc respects TMPDIR env variable for its intermediate files.
_nvcc_tmpdir = os.environ.get("NVCC_TMPDIR", tempfile.gettempdir())
os.makedirs(_nvcc_tmpdir, exist_ok=True)
os.environ.setdefault("TMPDIR", _nvcc_tmpdir)

# BLAS
blas_libs, blas_cc_flags, blas_nvcc_flags = find_blas_libraries()
libraries += blas_libs
CC_FLAGS += blas_cc_flags
NVCC_FLAGS += blas_nvcc_flags

print(f"BLAS libraries: {blas_libs}")

# Compiler override
if "CC" in os.environ or "CXX" in os.environ:
    if "CXX" in os.environ:
        os.environ["CC"] = os.environ["CXX"]
        CC = os.environ["CXX"]
    else:
        CC = os.environ["CC"]
    print(f"Using compiler: {CC}")

# Optimization flags
debug = os.environ.get("DEBUG", "0") == "1"
if debug:
    CC_FLAGS += ["-g", "-DDEBUG"]
    NVCC_FLAGS += ["-g", "-DDEBUG", "-Xcompiler=-fno-gnu-unique"]
else:
    CC_FLAGS += ["-O3"]
    NVCC_FLAGS += ["-O3", "-Xcompiler=-fno-gnu-unique"]

# Parallelism
if "MAX_JOBS" not in os.environ and os.cpu_count() > MAX_COMPILATION_THREADS:
    os.environ["MAX_JOBS"] = str(MAX_COMPILATION_THREADS)

# Source files
CPU_SOURCES = [
    "math_functions_cpu.cpp",
    "coordinate_map_manager.cpp",
    "convolution_cpu.cpp",
    "convolution_transpose_cpu.cpp",
    "local_pooling_cpu.cpp",
    "local_pooling_transpose_cpu.cpp",
    "global_pooling_cpu.cpp",
    "broadcast_cpu.cpp",
    "pruning_cpu.cpp",
    "interpolation_cpu.cpp",
    "quantization.cpp",
    "direct_max_pool.cpp",
]

GPU_SOURCES = [
    "math_functions_cpu.cpp",
    "math_functions_gpu.cu",
    "coordinate_map_manager.cu",
    "coordinate_map_gpu.cu",
    "convolution_kernel.cu",
    "convolution_gpu.cu",
    "convolution_transpose_gpu.cu",
    "pooling_avg_kernel.cu",
    "pooling_max_kernel.cu",
    "local_pooling_gpu.cu",
    "local_pooling_transpose_gpu.cu",
    "global_pooling_gpu.cu",
    "broadcast_kernel.cu",
    "broadcast_gpu.cu",
    "pruning_gpu.cu",
    "interpolation_gpu.cu",
    "spmm.cu",
    "gpu.cu",
    "quantization.cpp",
    "direct_max_pool.cpp",
]

if CPU_ONLY:
    Extension = CppExtension
    src_files = CPU_SOURCES
    bind_file = "pybind/minkowski.cpp"
    CC_FLAGS += ["-DCPU_ONLY"]
    NVCC_FLAGS += ["-DCPU_ONLY"]
    print("--- Building CPU_ONLY ---")
else:
    Extension = CUDAExtension
    src_files = GPU_SOURCES
    bind_file = "pybind/minkowski.cu"
    print("--- Building with CUDA ---")

ext_modules = [
    Extension(
        name="MinkowskiEngineBackend._C",
        sources=[
            *[str(SRC_PATH / f) for f in src_files],
            bind_file,
        ],
        extra_compile_args={"cxx": CC_FLAGS, "nvcc": NVCC_FLAGS},
        libraries=libraries,
        include_dirs=include_dirs,
    ),
]

setup(
    name="MinkowskiEngine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
