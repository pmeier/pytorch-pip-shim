import subprocess

import pytest

from pytorch_pip_shim import shim

__all__ = ["skip_if_cuda_unavailable", "skip_if_pip_is_not_shimmed"]

try:
    subprocess.check_call(
        "nvcc --version",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    CUDA_AVAILABLE = True
except subprocess.CalledProcessError:
    CUDA_AVAILABLE = False


skip_if_cuda_unavailable = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="Requires CUDA."
)

skip_if_pip_is_not_shimmed = pytest.mark.skipif(
    not shim.is_inserted(), reason="pytorch-pip-shim is not inserted."
)
