"""Edge-preserving Luminance-Guided Chrominance Denoising for Spektrafilm.

Smooths high-frequency sensor chrominance noise (color speckles) in digital images
while leaving luminance 100% untouched to preserve full sharpness, texture, and detail.
Supports both CuPy GPU arrays and NumPy CPU arrays.
"""

from __future__ import annotations

import numpy as np


def _guided_filter_channel(guide, src, sigma: float, eps: float, xp, ndimage):
    """Apply edge-preserving guided filter to a single 2D channel."""
    mean_guide = ndimage.gaussian_filter(guide, sigma)
    mean_src = ndimage.gaussian_filter(src, sigma)
    corr_guide = ndimage.gaussian_filter(guide * guide, sigma)
    corr_guide_src = ndimage.gaussian_filter(guide * src, sigma)

    var_guide = xp.maximum(corr_guide - mean_guide * mean_guide, 0.0)
    cov_guide_src = corr_guide_src - mean_guide * mean_src

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = ndimage.gaussian_filter(a, sigma)
    mean_b = ndimage.gaussian_filter(b, sigma)

    return mean_a * guide + mean_b


def chroma_denoise(
    image: np.ndarray,
    strength: float = 1.0,
    radius: float = 2.5,
    eps: float = 1e-3,
) -> np.ndarray:
    """Filter chrominance noise using luminance-guided edge-preserving filtering.

    Parameters
    ----------
    image : ndarray
        RGB image array with shape (H, W, 3). Can be a NumPy or CuPy array.
    strength : float, default 1.0
        Denoising strength. 0.0 disables filtering, 1.0 is standard full denoising,
        values up to 2.0 apply stronger suppression by scaling filter extent.
    radius : float, default 2.5
        Spatial filter radius (Gaussian sigma) in pixels.
    eps : float, default 1e-3
        Regularization parameter for the guided filter. Lower values strictly enforce
        luminance edges; higher values allow smoother blending across small variations.

    Returns
    -------
    ndarray
        Denoised RGB image with identical luminance and cleaned chrominance.
    """
    if strength <= 0.0 or radius <= 0.0:
        return image

    if image.ndim != 3 or image.shape[-1] < 3:
        return image

    from spektrafilm.cuda.device import is_cupy_array

    is_gpu = is_cupy_array(image)
    if is_gpu:
        import cupy as xp
        import cupyx.scipy.ndimage as ndimage
    else:
        xp = np
        import scipy.ndimage as ndimage

    orig_dtype = image.dtype
    # Use float32 for fast computation if input is float64 or uint
    calc_dtype = xp.float32 if orig_dtype != xp.float64 else xp.float64
    img_calc = image[..., :3].astype(calc_dtype, copy=False)

    r = img_calc[..., 0]
    g = img_calc[..., 1]
    b = img_calc[..., 2]

    # Luminance channel
    luma = 0.299 * r + 0.587 * g + 0.114 * b

    # Chrominance difference channels
    cb = b - luma
    cr = r - luma

    # When strength > 1.0, scale the filter radius and regularization to increase smoothing
    s = float(strength)
    eff_radius = float(radius) * (1.0 + max(s - 1.0, 0.0) * 0.8)
    eff_eps = float(eps) * (1.0 + max(s - 1.0, 0.0) * 0.5)

    # Pass 1: Fine-scale guided filter for high-frequency sensor noise speckles
    cb_filtered = _guided_filter_channel(luma, cb, eff_radius, eff_eps, xp, ndimage)
    cr_filtered = _guided_filter_channel(luma, cr, eff_radius, eff_eps, xp, ndimage)

    # Pass 2: Coarse-scale guided filter for low-frequency chroma mottling/blotches
    # Scales smoothly with strength so higher strength cleans wider color blotches
    if s >= 0.3:
        coarse_weight = min(max((s - 0.3) / 0.7, 0.0), 1.0)
        coarse_radius = eff_radius * (2.5 + max(s - 1.0, 0.0) * 1.0)
        coarse_eps = eff_eps * 2.0
        cb_coarse = _guided_filter_channel(luma, cb_filtered, coarse_radius, coarse_eps, xp, ndimage)
        cr_coarse = _guided_filter_channel(luma, cr_filtered, coarse_radius, coarse_eps, xp, ndimage)
        cb_filtered = (1.0 - coarse_weight) * cb_filtered + coarse_weight * cb_coarse
        cr_filtered = (1.0 - coarse_weight) * cr_filtered + coarse_weight * cr_coarse

    # Blend with original chrominance (blend must never exceed 1.0 to prevent color inversion)
    blend = min(max(s, 0.0), 1.0)
    cb_out = (1.0 - blend) * cb + blend * cb_filtered
    cr_out = (1.0 - blend) * cr + blend * cr_filtered

    # Reconstruct RGB preserving exact luminance
    r_out = luma + cr_out
    b_out = luma + cb_out
    g_out = (luma - 0.299 * r_out - 0.114 * b_out) / 0.587

    out = xp.stack([r_out, g_out, b_out], axis=-1)

    # If original image had alpha/extra channels, preserve them
    if image.shape[-1] > 3:
        out = xp.concatenate([out, image[..., 3:]], axis=-1)

    return out.astype(orig_dtype)


__all__ = ["chroma_denoise"]
