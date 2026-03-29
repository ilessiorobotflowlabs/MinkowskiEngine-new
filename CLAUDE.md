# MinkowskiEngine-new — Modernized Sparse Tensor Library

## Overview
Sparse tensor auto-differentiation library for high-dimensional convolutions.
Fork of NVIDIA/MinkowskiEngine v0.5.4, modernized for PyTorch 2.x + CUDA 12.x.

## Architecture
```
MinkowskiEngine/       # Python package (sparse ops, convolutions, pooling)
src/                   # C++/CUDA kernels (sparse conv, coordinate hashing)
pybind/                # pybind11 bindings
tests/                 # pytest tests
examples/              # Training examples
```

## Dev Commands
```bash
uv pip install -e .                          # Install editable
FORCE_CUDA=1 uv pip install -e .             # Force CUDA build
uv run pytest tests/ -x -v                   # Run tests
ruff check MinkowskiEngine/                  # Lint
```

## Conventions
- Package manager: `uv`
- Search: `rg` (ripgrep)
- Python: >=3.10
- PyTorch: >=2.0
- Git prefix: `[MINKOWSKI]`

# currentDate
Today's date is 2026-03-29.
