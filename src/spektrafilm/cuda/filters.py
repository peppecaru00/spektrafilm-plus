"""CUDA-accelerated optical and spatial filters (Gaussian, exponential, FFT convolution)."""
from __future__ import annotations

from typing import Any
import numpy as np

from spektrafilm.cuda.device import is_cuda_available, to_device, to_cpu

# Gaussian-mixture approximations of 2D exponential PSF (same as CPU)
_EXPONENTIAL_GAUSSIAN_FITS: dict[int, np.ndarray] = {
    2: np.array([
        [0.6235, 0.9401],
        [0.3765, 2.5177],
    ], dtype=np.float64),
    3: np.array([
        [0.1633, 0.5360],
        [0.6496, 1.5236],
        [0.1870, 2.7684],
    ], dtype=np.float64),
}


_YVV_CUDA_SOURCE = r"""
extern "C" {

#define DEFINE_YVV_KERNELS(TYPE, SUFFIX) \
__global__ void yvv_horizontal_kernel_##SUFFIX( \
    const TYPE* __restrict__ src, \
    TYPE* __restrict__ dst, \
    int H, int W, int C, \
    const TYPE* __restrict__ B, \
    const TYPE* __restrict__ B1, \
    const TYPE* __restrict__ B2, \
    const TYPE* __restrict__ B3 \
) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    int total_rows = H * C; \
    if (idx >= total_rows) return; \
\
    int c = idx / H; \
    int r = idx % H; \
\
    TYPE b = B[c]; \
    TYPE b1 = B1[c]; \
    TYPE b2 = B2[c]; \
    TYPE b3 = B3[c]; \
\
    int row_offset = r * (W * C) + c; \
    int stride = C; \
\
    TYPE x0 = src[row_offset]; \
    TYPE w1 = x0; \
    TYPE w2 = x0; \
    TYPE w3 = x0; \
\
    for (int j = 0; j < W; ++j) { \
        int pos = row_offset + j * stride; \
        TYPE val = src[pos]; \
        TYPE w = b * val + b1 * w1 + b2 * w2 + b3 * w3; \
        dst[pos] = w; \
        w3 = w2; \
        w2 = w1; \
        w1 = w; \
    } \
\
    TYPE xn = dst[row_offset + (W - 1) * stride]; \
    TYPE y1 = xn; \
    TYPE y2 = xn; \
    TYPE y3 = xn; \
\
    for (int j = W - 1; j >= 0; --j) { \
        int pos = row_offset + j * stride; \
        TYPE val = dst[pos]; \
        TYPE y = b * val + b1 * y1 + b2 * y2 + b3 * y3; \
        dst[pos] = y; \
        y3 = y2; \
        y2 = y1; \
        y1 = y; \
    } \
} \
\
__global__ void yvv_vertical_kernel_##SUFFIX( \
    const TYPE* __restrict__ src, \
    TYPE* __restrict__ dst, \
    int H, int W, int C, \
    const TYPE* __restrict__ B, \
    const TYPE* __restrict__ B1, \
    const TYPE* __restrict__ B2, \
    const TYPE* __restrict__ B3 \
) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    int total_cols = W * C; \
    if (idx >= total_cols) return; \
\
    int c = idx % C; \
    int col = idx / C; \
\
    TYPE b = B[c]; \
    TYPE b1 = B1[c]; \
    TYPE b2 = B2[c]; \
    TYPE b3 = B3[c]; \
\
    int col_offset = col * C + c; \
    int row_stride = W * C; \
\
    TYPE x0 = src[col_offset]; \
    TYPE w1 = x0; \
    TYPE w2 = x0; \
    TYPE w3 = x0; \
\
    for (int i = 0; i < H; ++i) { \
        int pos = col_offset + i * row_stride; \
        TYPE val = src[pos]; \
        TYPE w = b * val + b1 * w1 + b2 * w2 + b3 * w3; \
        dst[pos] = w; \
        w3 = w2; \
        w2 = w1; \
        w1 = w; \
    } \
\
    TYPE xn = dst[col_offset + (H - 1) * row_stride]; \
    TYPE y1 = xn; \
    TYPE y2 = xn; \
    TYPE y3 = xn; \
\
    for (int i = H - 1; i >= 0; --i) { \
        int pos = col_offset + i * row_stride; \
        TYPE val = dst[pos]; \
        TYPE y = b * val + b1 * y1 + b2 * y2 + b3 * y3; \
        dst[pos] = y; \
        y3 = y2; \
        y2 = y1; \
        y1 = y; \
    } \
}

DEFINE_YVV_KERNELS(float, f32)
DEFINE_YVV_KERNELS(double, f64)

}
"""

_YVV_MODULE = None


def _get_yvv_module():
    global _YVV_MODULE
    if _YVV_MODULE is not None:
        return _YVV_MODULE
    import cupy as cp
    _YVV_MODULE = cp.RawModule(code=_YVV_CUDA_SOURCE, options=("--use_fast_math",))
    return _YVV_MODULE


def _yvv_coeffs(sigma: float) -> tuple[float, float, float, float]:
    if sigma >= 2.5:
        q = 0.98711 * sigma - 0.96330
    else:
        q = 3.97156 - 4.14554 * np.sqrt(1.0 - 0.26891 * sigma)
    q2 = q * q
    q3 = q2 * q
    b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3
    b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3
    b2 = -(1.4281 * q2 + 1.26661 * q3)
    b3 = 0.422205 * q3
    B = 1.0 - (b1 + b2 + b3) / b0
    return B, b1 / b0, b2 / b0, b3 / b0


def cuda_gaussian_filter(image: Any, sigma: Any, truncate: float = 3.0) -> Any:
    """Apply a separable 2D Gaussian filter on GPU using mode='reflect'.

    Uses Young & van Vliet (2002) 3rd-order recursive IIR filter for sigma >= 3.0 px
    to achieve O(1) performance independent of sigma. Small sigmas use FIR.
    Supports both float32 and float64 dtypes natively for maximum memory efficiency.
    """
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage

    is_f64 = hasattr(image, "dtype") and (image.dtype == np.float64 or image.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32
    suffix = "f64" if is_f64 else "f32"

    d_image = to_device(image, dtype=target_dtype)
    is_2d = (d_image.ndim == 2)
    if is_2d:
        d_image = d_image[:, :, None]

    H, W, C = d_image.shape

    if np.ndim(sigma) == 0:
        sigmas = np.full(C, float(sigma), dtype=target_dtype)
    else:
        sigmas = np.asarray(sigma, dtype=target_dtype).ravel()
        if sigmas.shape[0] != C:
            raise ValueError("Per-channel sigma length must match channel count")

    if np.all(sigmas <= 0.0):
        return d_image[:, :, 0].copy() if is_2d else d_image.copy()

    # Check if all positive sigmas qualify for fast Young & van Vliet IIR (>= 3.0)
    all_yvv = np.all(sigmas >= 3.0)
    if all_yvv:
        B_vals = np.zeros(C, dtype=target_dtype)
        B1_vals = np.zeros(C, dtype=target_dtype)
        B2_vals = np.zeros(C, dtype=target_dtype)
        B3_vals = np.zeros(C, dtype=target_dtype)
        for c in range(C):
            B_vals[c], B1_vals[c], B2_vals[c], B3_vals[c] = _yvv_coeffs(float(sigmas[c]))

        d_B = cp.asarray(B_vals)
        d_B1 = cp.asarray(B1_vals)
        d_B2 = cp.asarray(B2_vals)
        d_B3 = cp.asarray(B3_vals)

        mod = _get_yvv_module()
        kh = mod.get_function(f"yvv_horizontal_kernel_{suffix}")
        kv = mod.get_function(f"yvv_vertical_kernel_{suffix}")

        tmp = cp.empty_like(d_image)
        out = cp.empty_like(d_image)

        block_size = 256
        grid_h = (H * C + block_size - 1) // block_size
        grid_v = (W * C + block_size - 1) // block_size

        kh((grid_h,), (block_size,), (d_image, tmp, np.int32(H), np.int32(W), np.int32(C), d_B, d_B1, d_B2, d_B3))
        kv((grid_v,), (block_size,), (tmp, out, np.int32(H), np.int32(W), np.int32(C), d_B, d_B1, d_B2, d_B3))
        del tmp, d_B, d_B1, d_B2, d_B3
        return out[:, :, 0] if is_2d else out

    # Mixed or small sigmas: per-channel dispatch
    out = cp.empty_like(d_image)
    for c in range(C):
        s = float(sigmas[c])
        if s <= 0.0:
            out[:, :, c] = d_image[:, :, c]
        elif s >= 3.0:
            b, b1, b2, b3 = _yvv_coeffs(s)
            d_b = cp.array([b], dtype=target_dtype)
            d_b1 = cp.array([b1], dtype=target_dtype)
            d_b2 = cp.array([b2], dtype=target_dtype)
            d_b3 = cp.array([b3], dtype=target_dtype)

            ch_in = d_image[:, :, c:c+1]
            ch_tmp = cp.empty_like(ch_in)
            ch_out = cp.empty_like(ch_in)

            mod = _get_yvv_module()
            kh = mod.get_function(f"yvv_horizontal_kernel_{suffix}")
            kv = mod.get_function(f"yvv_vertical_kernel_{suffix}")

            block_size = 256
            grid_h = (H + block_size - 1) // block_size
            grid_v = (W + block_size - 1) // block_size

            kh((grid_h,), (block_size,), (ch_in, ch_tmp, np.int32(H), np.int32(W), np.int32(1), d_b, d_b1, d_b2, d_b3))
            kv((grid_v,), (block_size,), (ch_tmp, ch_out, np.int32(H), np.int32(W), np.int32(1), d_b, d_b1, d_b2, d_b3))
            del ch_tmp, d_b, d_b1, d_b2, d_b3
            out[:, :, c] = ch_out[:, :, 0]
        else:
            out[:, :, c] = ndimage.gaussian_filter(d_image[:, :, c], (s, s), mode="reflect", truncate=truncate)

    return out[:, :, 0] if is_2d else out


def cuda_exponential_filter(
    image: Any, decay_constant: Any, *, n_gaussians: int = 3, truncate: float = 3.0,
) -> Any:
    """Per-channel 2D exponential filter on GPU via Gaussian-mixture surrogate."""
    import cupy as cp

    if n_gaussians not in _EXPONENTIAL_GAUSSIAN_FITS:
        raise ValueError(f"No fit for n_gaussians={n_gaussians}")

    fit = _EXPONENTIAL_GAUSSIAN_FITS[n_gaussians]
    decay_constant = np.asarray(decay_constant, dtype=np.float64)

    result = None
    for amplitude, sigma_ratio in fit:
        sigma_k = sigma_ratio * decay_constant
        component = cuda_gaussian_filter(image, sigma_k, truncate=truncate)
        if result is None:
            result = amplitude * component
        else:
            result += amplitude * component
        del component
    return result


def cuda_unsharp_mask(image: Any, sigma: float = 0.0, amount: float = 0.0) -> Any:
    """Apply an unsharp mask on GPU."""
    if sigma <= 0.0 or amount <= 0.0:
        return image
    d_image = to_device(image)
    image_blur = cuda_gaussian_filter(d_image, sigma)
    return d_image + amount * (d_image - image_blur)


def cuda_fftconvolve_2d(image_2d: Any, kernel_2d: Any, mode: str = "same") -> Any:
    """Perform 2D FFT convolution on GPU via cuFFT."""
    import cupyx.scipy.signal as signal

    d_in = to_device(image_2d)
    d_k = to_device(kernel_2d)
    return signal.fftconvolve(d_in, d_k, mode=mode)


def cuda_apply_diffusion_filter_um(image: Any, diffusion_filter: Any, pixel_size_um: float) -> Any:
    """Apply a diffusion-filter PSF to an RGB image directly on GPU."""
    if not diffusion_filter.active:
        return image
    if diffusion_filter.strength <= 0 or diffusion_filter.spatial_scale <= 0:
        return image

    import cupy as cp
    from spektrafilm.model.diffusion import (
        _DIFFUSION_FILTER_SHAPES,
        _strength_to_scatter,
        _overrides_from_params,
        _bloom_max_lambda_um,
        diffusion_filter_psf,
    )

    family = diffusion_filter.filter_family
    if family not in _DIFFUSION_FILTER_SHAPES:
        raise ValueError(f"Unknown diffusion filter family: {family!r}")

    p_s = _strength_to_scatter(diffusion_filter.strength, family)
    if p_s <= 0:
        return image

    overrides = _overrides_from_params(diffusion_filter)
    bloom_max_lambda_px = (
        _bloom_max_lambda_um(family, overrides) * diffusion_filter.spatial_scale / pixel_size_um
    )
    radius = int(np.ceil(max(8.0 * bloom_max_lambda_px, 5.0)))
    radius = min(radius, max(min(image.shape[:2]) // 2 - 1, 1))

    psf_shape = (2 * radius + 1, 2 * radius + 1)
    halo_warmth = float(getattr(diffusion_filter, "halo_warmth", 0.0))
    psf_per_channel = diffusion_filter_psf(
        psf_shape,
        family=family,
        spatial_scale=diffusion_filter.spatial_scale,
        pixel_size_um=pixel_size_um,
        halo_warmth=halo_warmth,
        overrides=overrides,
    )

    is_f64 = hasattr(image, "dtype") and (image.dtype == np.float64 or image.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32

    d_image = to_device(image, dtype=target_dtype)
    padded = cp.pad(d_image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    blurred = cp.empty_like(padded)

    import cupyx.scipy.signal as signal
    for channel in range(d_image.shape[2]):
        d_kernel = to_device(psf_per_channel[..., channel], dtype=target_dtype)
        blurred[:, :, channel] = signal.fftconvolve(
            padded[:, :, channel], d_kernel, mode="same",
        )
    del padded
    blurred = blurred[radius:-radius, radius:-radius, :]
    out = (1.0 - p_s) * d_image + p_s * blurred
    del blurred
    from spektrafilm.cuda.device import free_memory
    free_memory()
    return out
