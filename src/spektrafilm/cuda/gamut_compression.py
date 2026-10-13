"""CUDA-accelerated output gamut compression."""
from __future__ import annotations

from typing import Any
import numpy as np

from spektrafilm.cuda.device import is_cuda_available, to_device, to_cpu

_REINHARD_KNEE_CUDA_SOURCE = r"""
extern "C" {

__global__ void reinhard_knee_kernel_f32(
    const float* __restrict__ d,
    float* __restrict__ out,
    const float t,
    const float l,
    const float p,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    float val = d[idx];
    if (val <= t) {
        out[idx] = val;
        return;
    }
    float s = l - t;
    float n = (val - t) / s;
    float n_p = powf(n, p);
    float denom = powf(1.0f + n_p, 1.0f / p);
    out[idx] = t + s * (n / denom);
}

__global__ void reinhard_knee_kernel_f64(
    const double* __restrict__ d,
    double* __restrict__ out,
    const double t,
    const double l,
    const double p,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    double val = d[idx];
    if (val <= t) {
        out[idx] = val;
        return;
    }
    double s = l - t;
    double n = (val - t) / s;
    double n_p = pow(n, p);
    double denom = pow(1.0 + n_p, 1.0 / p);
    out[idx] = t + s * (n / denom);
}

// Backwards compatibility alias
__global__ void reinhard_knee_kernel(
    const double* __restrict__ d,
    double* __restrict__ out,
    const double t,
    const double l,
    const double p,
    const int N
) {
    reinhard_knee_kernel_f64(d, out, t, l, p, N);
}

}
"""

_MODULE = None


def _get_knee_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    import cupy as cp
    _MODULE = cp.RawModule(code=_REINHARD_KNEE_CUDA_SOURCE, options=("--use_fast_math",))
    return _MODULE


def cuda_reinhard_knee(
    d: Any,
    threshold: float,
    limit: float,
    power: float,
) -> Any:
    """Evaluate the Reinhard knee on GPU with native f32/f64 precision."""
    import cupy as cp

    is_f64 = hasattr(d, "dtype") and (d.dtype == np.float64 or d.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32
    suffix = "f64" if is_f64 else "f32"

    d_in = to_device(d, dtype=target_dtype)
    out = cp.empty_like(d_in)

    N = d_in.size
    mod = _get_knee_module()
    kernel = mod.get_function(f"reinhard_knee_kernel_{suffix}")

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (d_in, out, target_dtype(threshold), target_dtype(limit), target_dtype(power), np.int32(N)),
    )
    return out


def cuda_compress_rgb_aces_rgc(
    rgb: Any,
    threshold: float,
    limit: float,
    power: float,
) -> Any:
    """ACES Reference Gamut Compression on GPU with native f32/f64 precision."""
    import cupy as cp

    is_f64 = hasattr(rgb, "dtype") and (rgb.dtype == np.float64 or rgb.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32

    d_rgb = to_device(rgb, dtype=target_dtype)
    ach = cp.max(d_rgb, axis=-1, keepdims=True)
    safe_ach = cp.where(ach > 1e-12, ach, 1.0)

    d = (ach - d_rgb) / safe_ach
    d_compressed = cuda_reinhard_knee(d, threshold, limit, power)
    del d

    rgb_compressed = ach * (1.0 - d_compressed)
    del d_compressed
    return cp.where(ach > 1e-12, rgb_compressed, d_rgb)


def cuda_compress_rgb(
    rgb: Any,
    spec: Any,
    output_color_space: str = "sRGB",
    preview_mode: bool = False,
) -> Any:
    """Dispatch output gamut compression on GPU."""
    if not spec.active or spec.algorithm == "off":
        return rgb

    import cupy as cp
    is_f64 = hasattr(rgb, "dtype") and (rgb.dtype == np.float64 or rgb.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32

    d_rgb = to_device(rgb, dtype=target_dtype)

    if spec.algorithm == "aces_rgc":
        t, l, p = spec.knee
        return cuda_compress_rgb_aces_rgc(d_rgb, threshold=t, limit=l, power=p)

    if preview_mode:
        # In interactive preview mode on GPU, preserve real-time responsiveness (40-60+ FPS)
        # by applying GPU ACES RGC instead of stalling the event loop on CPU appearance models.
        t, l, p = spec.knee
        return cuda_compress_rgb_aces_rgc(d_rgb, threshold=t, limit=l, power=p)

    # For other complex perceptual spaces (CAM16-UCS, OkLch, etc.) in final render / offline:
    # Transfer to CPU for the colour/SciPy transforms if GPU implementation not available,
    # and free GPU memory beforehand so CPU has maximum memory headroom.
    from spektrafilm.utils.gamut_compression import compress_rgb
    cpu_rgb = to_cpu(d_rgb)
    del d_rgb
    from spektrafilm.cuda.device import free_memory
    free_memory()

    result = compress_rgb(cpu_rgb, spec, output_color_space=output_color_space)
    del cpu_rgb
    return to_device(result, dtype=target_dtype)
