"""CUDA-accelerated spectral computations and tensor contractions."""
from __future__ import annotations

from typing import Any
import numpy as np

from spektrafilm.cuda.device import is_cuda_available, to_device, to_cpu


def cuda_compute_density_spectral(
    channel_density: Any,
    density_cmy: Any,
    base_density: Any = None,
) -> Any:
    """Compute spectral density from CMY density on GPU via tensor dot product.

    Equivalent to contract('ijk, lk->ijl', density_cmy, channel_density) + base_density.
    """
    import cupy as cp

    is_f64 = hasattr(density_cmy, "dtype") and (density_cmy.dtype == np.float64 or density_cmy.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32

    d_cmy = to_device(density_cmy, dtype=target_dtype)
    d_channel_density = to_device(channel_density, dtype=target_dtype)

    # density_cmy is (H, W, 3), channel_density is (N_wavelengths, 3)
    # Output is (H, W, N_wavelengths)
    density_spectral = cp.tensordot(d_cmy, d_channel_density, axes=([2], [1]))

    if base_density is not None:
        d_base = to_device(base_density, dtype=target_dtype)
        density_spectral += d_base

    return density_spectral


def cuda_density_to_light(density_spectral: Any, illuminant: Any) -> Any:
    """Convert spectral density to transmitted/reflected light irradiance on GPU.

    Equivalent to 10 ** (-density_spectral) * illuminant.
    """
    import cupy as cp

    is_f64 = hasattr(density_spectral, "dtype") and (density_spectral.dtype == np.float64 or density_spectral.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32

    d_density = to_device(density_spectral, dtype=target_dtype)
    d_illuminant = to_device(illuminant, dtype=target_dtype)

    # 10 ** (-density_spectral)
    transmittance = cp.exp(-d_density * cp.log(target_dtype(10.0)))
    res = transmittance * d_illuminant
    return cp.nan_to_num(res, nan=0.0)


def cuda_spectral_integrate(light: Any, response_curves: Any, normalization: Any = None) -> Any:
    """Integrate spectral light against response curves (sensitivities or CMFs) on GPU.

    Equivalent to contract('ijk, kl->ijl', light, response_curves) / normalization.
    """
    import cupy as cp

    is_f64 = hasattr(light, "dtype") and (light.dtype == np.float64 or light.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32

    d_light = to_device(light, dtype=target_dtype)
    d_response = to_device(response_curves, dtype=target_dtype)

    out = cp.tensordot(d_light, d_response, axes=([2], [0]))
    if normalization is not None:
        d_norm = to_device(normalization, dtype=target_dtype)
        out /= d_norm

    return out


_FUSED_SPECTRAL_SOURCE = r"""
extern "C" {

#define DEFINE_FUSED_SPECTRAL_KERNEL(TYPE, SUFFIX) \
__global__ void fused_spectral_integrate_kernel_##SUFFIX( \
    const TYPE* __restrict__ cmy,            /* (N, 3) */ \
    const TYPE* __restrict__ channel_density,/* (W, 3) */ \
    const TYPE* __restrict__ base_density,   /* (W,) */ \
    const TYPE* __restrict__ illuminant,     /* (W,) */ \
    const TYPE* __restrict__ response,       /* (W, C_out) */ \
    const TYPE* __restrict__ inv_norm,       /* (C_out,) */ \
    TYPE* __restrict__ output,               /* (N, C_out) */ \
    const int N, \
    const int W_count, \
    const int C_out, \
    const int has_base, \
    const int has_norm \
) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx >= N) return; \
\
    TYPE c = cmy[idx * 3 + 0]; \
    TYPE m = cmy[idx * 3 + 1]; \
    TYPE y = cmy[idx * 3 + 2]; \
\
    TYPE acc0 = (TYPE)0.0; \
    TYPE acc1 = (TYPE)0.0; \
    TYPE acc2 = (TYPE)0.0; \
\
    const TYPE ln10 = (TYPE)2.30258509299404568402; \
\
    for (int w = 0; w < W_count; ++w) { \
        TYPE d = c * channel_density[w * 3 + 0] + \
                 m * channel_density[w * 3 + 1] + \
                 y * channel_density[w * 3 + 2] + \
                 (has_base ? base_density[w] : (TYPE)0.0); \
        TYPE trans = exp(-d * ln10); \
        TYPE light = trans * illuminant[w]; \
        if (isnan(light)) light = (TYPE)0.0; \
\
        acc0 += light * response[w * C_out + 0]; \
        acc1 += light * response[w * C_out + 1]; \
        acc2 += light * response[w * C_out + 2]; \
    } \
\
    if (has_norm) { \
        acc0 *= inv_norm[0]; \
        acc1 *= inv_norm[1]; \
        acc2 *= inv_norm[2]; \
    } \
\
    output[idx * C_out + 0] = acc0; \
    output[idx * C_out + 1] = acc1; \
    output[idx * C_out + 2] = acc2; \
}

DEFINE_FUSED_SPECTRAL_KERNEL(float, f32)
DEFINE_FUSED_SPECTRAL_KERNEL(double, f64)

// Backwards compatibility alias
__global__ void fused_spectral_integrate_kernel(
    const double* __restrict__ cmy,
    const double* __restrict__ channel_density,
    const double* __restrict__ base_density,
    const double* __restrict__ illuminant,
    const double* __restrict__ response,
    const double* __restrict__ inv_norm,
    double* __restrict__ output,
    const int N,
    const int W_count,
    const int C_out,
    const int has_base,
    const int has_norm
) {
    fused_spectral_integrate_kernel_f64(
        cmy, channel_density, base_density, illuminant, response, inv_norm,
        output, N, W_count, C_out, has_base, has_norm
    );
}

}
"""

_FUSED_MODULE = None


def _get_fused_spectral_module():
    global _FUSED_MODULE
    if _FUSED_MODULE is not None:
        return _FUSED_MODULE

    import cupy as cp
    _FUSED_MODULE = cp.RawModule(code=_FUSED_SPECTRAL_SOURCE, options=("--use_fast_math",))
    return _FUSED_MODULE


def cuda_fused_spectral_integrate(
    density_cmy: Any,
    channel_density: Any,
    base_density: Any = None,
    illuminant: Any = None,
    response_curves: Any = None,
    normalization: Any = None,
) -> Any:
    """Fused single-pass spectral integration on GPU without allocating 3D intermediate arrays.

    Evaluates:
        density_spectral = cmy @ channel_density.T + base_density
        light = 10 ** (-density_spectral) * illuminant
        out = (light @ response_curves) / normalization

    in GPU registers per pixel, eliminating gigabytes of intermediate VRAM traffic.
    Supports both float32 and float64 dtypes natively for maximum speed and memory efficiency.
    """
    import cupy as cp

    is_f64 = hasattr(density_cmy, "dtype") and (density_cmy.dtype == np.float64 or density_cmy.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32
    suffix = "f64" if is_f64 else "f32"

    d_cmy = to_device(density_cmy, dtype=target_dtype)
    d_channel_density = to_device(channel_density, dtype=target_dtype)
    d_illuminant = to_device(illuminant, dtype=target_dtype)
    d_response = to_device(response_curves, dtype=target_dtype)

    orig_shape = d_cmy.shape
    N = int(np.prod(orig_shape[:-1]))
    flat_cmy = d_cmy.reshape(N, 3)

    W_count = d_channel_density.shape[0]
    C_out = d_response.shape[1]

    has_base = (base_density is not None)
    d_base = to_device(base_density, dtype=target_dtype) if has_base else cp.zeros(1, dtype=target_dtype)

    has_norm = (normalization is not None)
    if has_norm:
        norm_arr = np.asarray(normalization, dtype=target_dtype)
        if norm_arr.ndim == 0:
            inv_norm = cp.full(C_out, 1.0 / float(norm_arr), dtype=target_dtype)
        else:
            inv_norm = to_device(1.0 / norm_arr, dtype=target_dtype)
    else:
        inv_norm = cp.ones(C_out, dtype=target_dtype)

    output = cp.empty((N, C_out), dtype=target_dtype)

    mod = _get_fused_spectral_module()
    kernel = mod.get_function(f"fused_spectral_integrate_kernel_{suffix}")

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (
            flat_cmy,
            d_channel_density,
            d_base,
            d_illuminant,
            d_response,
            inv_norm,
            output,
            np.int32(N),
            np.int32(W_count),
            np.int32(C_out),
            np.int32(1 if has_base else 0),
            np.int32(1 if has_norm else 0),
        ),
    )

    return output.reshape(*orig_shape[:-1], C_out)
