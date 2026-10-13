"""CUDA-accelerated color conversions and transfer functions."""
from __future__ import annotations

import functools
from typing import Any
import numpy as np
import colour

from spektrafilm.cuda.device import is_cupy_array, to_device, to_cpu


# --- Cached Conversion Matrices ---

@functools.lru_cache(maxsize=64)
def get_xyz_to_rgb_matrix(
    colourspace: str,
    illuminant_xy: tuple[float, float] | None = None,
    cat: str = "CAT02",
) -> np.ndarray:
    """Precompute the 3x3 XYZ -> RGB conversion matrix with chromatic adaptation."""
    cs = colour.RGB_COLOURSPACES[colourspace]
    if illuminant_xy is None:
        return np.asarray(cs.matrix_XYZ_to_RGB, dtype=np.float64)

    wp_test = np.asarray(illuminant_xy, dtype=np.float64)
    wp_ref = np.asarray(cs.whitepoint, dtype=np.float64)
    if np.allclose(wp_test, wp_ref):
        return np.asarray(cs.matrix_XYZ_to_RGB, dtype=np.float64)

    m_cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        colour.xyY_to_XYZ(colour.xy_to_xyY(wp_test)),
        colour.xyY_to_XYZ(colour.xy_to_xyY(wp_ref)),
        transform=cat,
    )
    return np.asarray(np.dot(cs.matrix_XYZ_to_RGB, m_cat), dtype=np.float64)


@functools.lru_cache(maxsize=64)
def get_rgb_to_xyz_matrix(
    colourspace: str,
    illuminant_xy: tuple[float, float] | None = None,
    cat: str = "CAT16",
) -> np.ndarray:
    """Precompute the 3x3 RGB -> XYZ conversion matrix with chromatic adaptation."""
    cs = colour.RGB_COLOURSPACES[colourspace]
    if illuminant_xy is None:
        return np.asarray(cs.matrix_RGB_to_XYZ, dtype=np.float64)

    wp_test = np.asarray(illuminant_xy, dtype=np.float64)
    wp_ref = np.asarray(cs.whitepoint, dtype=np.float64)
    if np.allclose(wp_test, wp_ref):
        return np.asarray(cs.matrix_RGB_to_XYZ, dtype=np.float64)

    m_cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        colour.xyY_to_XYZ(colour.xy_to_xyY(wp_ref)),
        colour.xyY_to_XYZ(colour.xy_to_xyY(wp_test)),
        transform=cat,
    )
    return np.asarray(np.dot(m_cat, cs.matrix_RGB_to_XYZ), dtype=np.float64)


@functools.lru_cache(maxsize=64)
def get_rgb_to_rgb_matrix(
    input_colourspace: str,
    output_colourspace: str,
    cat: str = "CAT02",
) -> np.ndarray:
    """Precompute the 3x3 RGB -> RGB conversion matrix with chromatic adaptation."""
    cs_in = colour.RGB_COLOURSPACES[input_colourspace]
    cs_out = colour.RGB_COLOURSPACES[output_colourspace]
    return np.asarray(colour.matrix_RGB_to_RGB(cs_in, cs_out, cat), dtype=np.float64)


# --- CuPy Matrix Multiplication ---

def cuda_apply_matrix(image: Any, matrix: Any) -> Any:
    """Apply a 3x3 conversion matrix to the last dimension of a GPU or CPU array.

    Matches colour.algebra.vecmul(matrix, image) to machine precision.
    """
    import cupy as cp

    is_f64 = hasattr(image, "dtype") and (image.dtype == np.float64 or image.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32

    d_img = to_device(image, dtype=target_dtype)
    d_mat = to_device(matrix, dtype=target_dtype)
    return cp.tensordot(d_img, d_mat, axes=([-1], [1]))


# --- CuPy Transfer Function Kernels ---

_SRGB_ENCODE_KERNEL = None
_SRGB_DECODE_KERNEL = None
_BT2020_ENCODE_KERNEL = None
_BT2020_DECODE_KERNEL = None
_ROMM_ENCODE_KERNEL = None
_ROMM_DECODE_KERNEL = None
_GAMMA_ENCODE_KERNEL = None
_GAMMA_DECODE_KERNEL = None
_TRI2QUAD_KERNEL = None


def _init_kernels():
    global _SRGB_ENCODE_KERNEL, _SRGB_DECODE_KERNEL
    global _BT2020_ENCODE_KERNEL, _BT2020_DECODE_KERNEL
    global _ROMM_ENCODE_KERNEL, _ROMM_DECODE_KERNEL
    global _GAMMA_ENCODE_KERNEL, _GAMMA_DECODE_KERNEL
    global _TRI2QUAD_KERNEL

    if _SRGB_ENCODE_KERNEL is not None:
        return

    import cupy as cp

    _SRGB_ENCODE_KERNEL = cp.ElementwiseKernel(
        "float64 L", "float64 V",
        """
        double a = L;
        if (a <= 0.0031308) {
            V = a * 12.92;
        } else {
            double s = (a >= 0.0) ? 1.0 : -1.0;
            V = 1.055 * (s * pow(fabs(a), 1.0 / 2.4)) - 0.055;
        }
        """,
        "srgb_encode_kernel",
    )

    _SRGB_DECODE_KERNEL = cp.ElementwiseKernel(
        "float64 V", "float64 L",
        """
        double v = V;
        if (v <= 0.040449936) {
            L = v / 12.92;
        } else {
            double term = (v + 0.055) / 1.055;
            double s = (term >= 0.0) ? 1.0 : -1.0;
            L = s * pow(fabs(term), 2.4);
        }
        """,
        "srgb_decode_kernel",
    )

    _BT2020_ENCODE_KERNEL = cp.ElementwiseKernel(
        "float64 E", "float64 Ep",
        """
        double a = 1.099;
        double b = 0.018;
        double e = E;
        if (b > e) {
            Ep = e * 4.5;
        } else {
            double s = (e >= 0.0) ? 1.0 : -1.0;
            Ep = a * (s * pow(fabs(e), 0.45)) - (a - 1.0);
        }
        """,
        "bt2020_encode_kernel",
    )

    _BT2020_DECODE_KERNEL = cp.ElementwiseKernel(
        "float64 Ep", "float64 E",
        """
        double a = 1.099;
        double b = 0.018;
        double ep = Ep;
        if (ep < b * 4.5) {
            E = ep / 4.5;
        } else {
            double term = (ep + (a - 1.0)) / a;
            double s = (term >= 0.0) ? 1.0 : -1.0;
            E = s * pow(fabs(term), 1.0 / 0.45);
        }
        """,
        "bt2020_decode_kernel",
    )

    _ROMM_ENCODE_KERNEL = cp.ElementwiseKernel(
        "float64 X", "float64 Xp",
        """
        double et = 0.001953125;
        double x = X;
        if (x < et) {
            Xp = x * 16.0;
        } else {
            double s = (x >= 0.0) ? 1.0 : -1.0;
            Xp = s * pow(fabs(x), 1.0 / 1.8);
        }
        """,
        "romm_encode_kernel",
    )

    _ROMM_DECODE_KERNEL = cp.ElementwiseKernel(
        "float64 Xp", "float64 X",
        """
        double xp = Xp;
        if (xp < 0.03125) {
            X = xp / 16.0;
        } else {
            double s = (xp >= 0.0) ? 1.0 : -1.0;
            X = s * pow(fabs(xp), 1.8);
        }
        """,
        "romm_decode_kernel",
    )

    _GAMMA_ENCODE_KERNEL = cp.ElementwiseKernel(
        "float64 x, float64 inv_gamma", "float64 y",
        """
        double s = (x >= 0.0) ? 1.0 : -1.0;
        y = s * pow(fabs(x), inv_gamma);
        """,
        "gamma_encode_kernel",
    )

    _GAMMA_DECODE_KERNEL = cp.ElementwiseKernel(
        "float64 y, float64 gamma", "float64 x",
        """
        double s = (y >= 0.0) ? 1.0 : -1.0;
        x = s * pow(fabs(y), gamma);
        """,
        "gamma_decode_kernel",
    )

    _TRI2QUAD_KERNEL = cp.ElementwiseKernel(
        "T tx, T ty",
        "T x, T y",
        """
        T one_minus_tx = (T)1.0 - tx;
        T denom = (one_minus_tx > (T)1e-10) ? one_minus_tx : (T)1e-10;
        T y_val = ty / denom;
        T x_val = one_minus_tx * one_minus_tx;
        x = (x_val < (T)0.0) ? (T)0.0 : ((x_val > (T)1.0) ? (T)1.0 : x_val);
        y = (y_val < (T)0.0) ? (T)0.0 : ((y_val > (T)1.0) ? (T)1.0 : y_val);
        """,
        "tri2quad_kernel",
    )


def cuda_cctf_encoding(rgb: Any, colourspace: str = "sRGB") -> Any:
    """Evaluate CCTF encoding on GPU for the given color space."""
    _init_kernels()
    d_rgb = to_device(rgb, dtype=np.float64)

    if colourspace in ("sRGB", "Display P3"):
        return _SRGB_ENCODE_KERNEL(d_rgb)
    elif colourspace == "ITU-R BT.2020":
        return _BT2020_ENCODE_KERNEL(d_rgb)
    elif colourspace == "ProPhoto RGB":
        return _ROMM_ENCODE_KERNEL(d_rgb)
    elif colourspace == "DCI-P3":
        return _GAMMA_ENCODE_KERNEL(d_rgb, 1.0 / 2.6)
    elif colourspace == "Adobe RGB (1998)":
        return _GAMMA_ENCODE_KERNEL(d_rgb, 1.0 / (563.0 / 256.0))
    elif colourspace in ("ACES2065-1", "ACEScc", "ACEScct"):
        return d_rgb
    else:
        # Fallback to colour-science on CPU for unsupported or exotic spaces
        cpu_rgb = to_cpu(d_rgb)
        cs = colour.RGB_COLOURSPACES[colourspace]
        if cs.cctf_encoding is not None:
            res = cs.cctf_encoding(cpu_rgb)
            return to_device(res)
        return d_rgb


def cuda_cctf_decoding(rgb: Any, colourspace: str = "sRGB") -> Any:
    """Evaluate CCTF decoding on GPU for the given color space."""
    _init_kernels()
    d_rgb = to_device(rgb, dtype=np.float64)

    if colourspace in ("sRGB", "Display P3"):
        return _SRGB_DECODE_KERNEL(d_rgb)
    elif colourspace == "ITU-R BT.2020":
        return _BT2020_DECODE_KERNEL(d_rgb)
    elif colourspace == "ProPhoto RGB":
        return _ROMM_DECODE_KERNEL(d_rgb)
    elif colourspace == "DCI-P3":
        return _GAMMA_DECODE_KERNEL(d_rgb, 2.6)
    elif colourspace == "Adobe RGB (1998)":
        return _GAMMA_DECODE_KERNEL(d_rgb, 563.0 / 256.0)
    elif colourspace in ("ACES2065-1", "ACEScc", "ACEScct"):
        return d_rgb
    else:
        # Fallback to colour-science on CPU for unsupported or exotic spaces
        cpu_rgb = to_cpu(d_rgb)
        cs = colour.RGB_COLOURSPACES[colourspace]
        if cs.cctf_decoding is not None:
            res = cs.cctf_decoding(cpu_rgb)
            return to_device(res)
        return d_rgb


def cuda_tri2quad(xy: Any) -> Any:
    """Convert triangular/chromaticity coordinates into square/quad coordinates on GPU.

    Matches spektrafilm.utils.spectral_upsampling._tri2quad to exact machine precision.
    """
    _init_kernels()
    import cupy as cp
    is_f64 = hasattr(xy, "dtype") and (xy.dtype == np.float64 or xy.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32
    d_xy = to_device(xy, dtype=target_dtype)
    x_out, y_out = _TRI2QUAD_KERNEL(d_xy[..., 0], d_xy[..., 1])
    return cp.stack((x_out, y_out), axis=-1)


def cuda_rgb_to_tc_b(
    rgb: Any,
    colourspace: str = "ITU-R BT.2020",
    apply_cctf_decoding: bool = False,
    reference_illuminant_xy: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Convert linear/encoded RGB to triangular chromaticity coordinates and scale b on GPU.

    Replaces spectral_upsampling._rgb_to_tc_b while keeping all operations on GPU.
    """
    import cupy as cp

    is_f64 = hasattr(rgb, "dtype") and (rgb.dtype == np.float64 or rgb.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32
    d_rgb = to_device(rgb, dtype=target_dtype)
    if apply_cctf_decoding:
        d_rgb = cuda_cctf_decoding(d_rgb, colourspace)

    m = get_rgb_to_xyz_matrix(colourspace, reference_illuminant_xy, cat="CAT16")
    xyz = cuda_apply_matrix(d_rgb, m)

    b = cp.sum(xyz, axis=-1)
    xy = xyz[..., 0:2] / cp.fmax(b[..., None], 1e-10)
    tc = cuda_tri2quad(xy)
    b = cp.nan_to_num(b)

    return tc, b
