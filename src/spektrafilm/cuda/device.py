"""CUDA device management and host-device transfer utilities."""
from __future__ import annotations

import logging
import sys
import warnings
from typing import Any

import numpy as np

# Suppress benign CuPy warning on Windows wheels when CUDA 12 packages use splayed layout
warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*", category=UserWarning)

logger = logging.getLogger(__name__)

_CUPY_AVAILABLE: bool | None = None
_CUDA_INIT_ERROR: str | None = None
_CUDA_DEVICE_INFO: dict[str, Any] | None = None
_MANAGED_POOL: Any = None


def _configure_windows_cuda_dlls() -> None:
    """Ensure all bundled and site-packages NVIDIA CUDA DLLs are registered on Windows."""
    if sys.platform != "win32" or not hasattr(sys, "add_dll_directory"):
        import os
        if not hasattr(os, "add_dll_directory"):
            return

    import os
    from pathlib import Path

    # 1. In frozen mode (PyInstaller onedir), register bundle_dir and all subdirectories with DLLs
    if getattr(sys, "frozen", False):
        bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        try:
            os.add_dll_directory(bundle_dir)
        except Exception:
            pass
        for root, dirs, files in os.walk(bundle_dir):
            if any(f.lower().endswith(".dll") for f in files):
                try:
                    os.add_dll_directory(root)
                except Exception:
                    pass
        cuda_runtime_dir = os.path.join(bundle_dir, "nvidia", "cuda_runtime")
        if os.path.isdir(cuda_runtime_dir):
            os.environ.setdefault("CUDA_PATH", cuda_runtime_dir)

    # 2. In normal Python or editable installs, register site-packages nvidia directories
    for p in list(sys.path):
        nvidia_cand = os.path.join(p, "nvidia")
        if os.path.isdir(nvidia_cand):
            for root, dirs, files in os.walk(nvidia_cand):
                if any(f.lower().endswith(".dll") for f in files):
                    try:
                        os.add_dll_directory(root)
                    except Exception:
                        pass


def get_cuda_init_error() -> str | None:
    """Return error message from CUDA initialization probe if it failed."""
    return _CUDA_INIT_ERROR


def is_cuda_available() -> bool:
    """Return True if CuPy and a functional CUDA GPU are available."""
    global _CUPY_AVAILABLE, _MANAGED_POOL, _CUDA_INIT_ERROR
    if _CUPY_AVAILABLE is not None:
        return _CUPY_AVAILABLE

    try:
        _configure_windows_cuda_dlls()
        import cupy as cp
        # Probe device count and properties
        count = cp.cuda.runtime.getDeviceCount()
        if count > 0:
            # Minimal test allocation and compute
            test_arr = cp.zeros(1, dtype=cp.float32)
            _ = float(test_arr.sum().get())
            _CUPY_AVAILABLE = True
            _CUDA_INIT_ERROR = None

            # Configure CuPy allocator.
            # IMPORTANT: Do NOT use cudaMallocManaged on Windows!
            # Under Windows Display Driver Model (WDDM), Unified Memory page faults and
            # PCIe migrations cause severe WDDM scheduling locks that stall the Desktop
            # Window Manager (DWM), freezing the mouse cursor and the entire Windows GUI.
            # Standard device memory pool (cudaMalloc) runs directly in physical VRAM with zero
            # PCIe page faulting, preventing system freezes.
            if sys.platform != "win32":
                try:
                    if hasattr(cp.cuda, "malloc_managed"):
                        _MANAGED_POOL = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
                        cp.cuda.set_allocator(_MANAGED_POOL.malloc)
                except Exception as exc:
                    logger.debug("Managed memory allocator initialization failed: %s", exc)
        else:
            _CUDA_INIT_ERROR = "No CUDA-capable GPU found (getDeviceCount returned 0)"
            _CUPY_AVAILABLE = False
    except Exception as exc:
        _CUDA_INIT_ERROR = f"{type(exc).__name__}: {exc}"
        logger.warning("CUDA acceleration not available: %s", exc)
        _CUPY_AVAILABLE = False

    return _CUPY_AVAILABLE


def get_device_info() -> dict[str, Any]:
    """Return a dictionary containing details about the CUDA device."""
    global _CUDA_DEVICE_INFO
    if _CUDA_DEVICE_INFO is not None:
        return _CUDA_DEVICE_INFO

    if not is_cuda_available():
        _CUDA_DEVICE_INFO = {
            "available": False,
            "device_name": "CPU (multi-core)",
            "total_memory_gb": 0.0,
            "compute_capability": (0, 0),
            "error": _CUDA_INIT_ERROR,
        }
        return _CUDA_DEVICE_INFO

    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode("utf-8") if isinstance(props["name"], bytes) else str(props["name"])
    total_mem = props["totalGlobalMem"] / (1024 ** 3)
    major = props.get("major", 0)
    minor = props.get("minor", 0)

    _CUDA_DEVICE_INFO = {
        "available": True,
        "device_name": name,
        "total_memory_gb": round(total_mem, 2),
        "compute_capability": (major, minor),
        "error": None,
    }
    return _CUDA_DEVICE_INFO



def resolve_device(device_setting: str = "auto") -> str:
    """Resolve 'auto', 'cuda', or 'cpu' to an active device string ('cuda' or 'cpu')."""
    setting = (device_setting or "auto").lower()
    if setting == "cuda":
        if is_cuda_available():
            return "cuda"
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    elif setting == "cpu":
        return "cpu"
    else:  # "auto"
        return "cuda" if is_cuda_available() else "cpu"


def to_device(array: Any, dtype: Any = None) -> Any:
    """Transfer an array to the CUDA device as a CuPy array.

    If CUDA is not available or array is not array-like, returns array (or np.asarray).
    If already a CuPy array, ensures contiguous layout and dtype.
    """
    if not is_cuda_available():
        arr = np.asarray(array)
        return arr.astype(dtype, copy=False) if dtype else arr

    import cupy as cp
    if isinstance(array, cp.ndarray):
        arr = cp.ascontiguousarray(array)
        return arr.astype(dtype, copy=False) if dtype else arr

    arr = cp.asarray(array)
    arr = cp.ascontiguousarray(arr)
    if dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def to_cpu(array: Any) -> np.ndarray:
    """Transfer an array back to the CPU as a NumPy array."""
    if is_cuda_available():
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return array.get()
    return np.asarray(array)


def is_cupy_array(array: Any) -> bool:
    """Check if the given object is a CuPy array."""
    if not is_cuda_available():
        return False
    import cupy as cp
    return isinstance(array, cp.ndarray)


def get_array_module(array: Any) -> Any:
    """Return cupy if array is a CuPy array, else numpy."""
    if is_cuda_available():
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp
    return np


def synchronize() -> None:
    """Synchronize the current CUDA stream if CUDA is active."""
    if is_cuda_available():
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def free_memory() -> None:
    """Free cached blocks from GPU and managed memory pools."""
    if is_cuda_available():
        import cupy as cp
        try:
            if _MANAGED_POOL is not None:
                _MANAGED_POOL.free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
