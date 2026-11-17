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
) -> Any:
    """Dispatch output gamut compression on GPU (same result as the CPU path)."""
    if not spec.active or spec.algorithm == "off":
        return rgb

    import cupy as cp
    is_f64 = hasattr(rgb, "dtype") and (rgb.dtype == np.float64 or rgb.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32

    d_rgb = to_device(rgb, dtype=target_dtype)
    t, l, p = spec.knee

    if spec.algorithm == "aces_rgc":
        return cuda_compress_rgb_aces_rgc(d_rgb, threshold=t, limit=l, power=p)

    perceptual = {
        "cam16ucs": cuda_compress_rgb_cam16ucs_chroma,
        "oklch": cuda_compress_rgb_oklch_chroma,
        "oklrab": cuda_compress_rgb_oklrab_chroma,
        "jzazbz": cuda_compress_rgb_jzazbz_chroma,
    }
    if spec.algorithm in perceptual:
        return perceptual[spec.algorithm](
            d_rgb, output_color_space, threshold=t, limit=l, power=p,
            lightness_compression=spec.lightness_compression,
        )

    # Unknown algorithm: let the CPU path raise its error.
    from spektrafilm.utils.gamut_compression import compress_rgb
    return compress_rgb(to_cpu(d_rgb), spec, output_color_space=output_color_space)


# ---------------------------------------------------------------------------
# CAM16-UCS chroma compression on GPU — a CuPy port of
# utils.gamut_compression.compress_rgb_cam16ucs_chroma, which runs
# colour-science's XYZ_to_CAM16 / CAM16_to_XYZ (Li 2017 UCS) per pixel.
# Viewing-condition scalars come from colour itself, so only the per-pixel
# formulas are re-stated here.
# ---------------------------------------------------------------------------

_CAM16UCS_C1 = 0.007
_CAM16UCS_C2 = 0.0228
_CAM16_CONSTANTS: dict = {}
_DEVICE_TABLES: dict = {}


def _cam16ucs_constants(output_color_space: str) -> dict:
    """Per-output-space scalars and fused 3x3 matrices (cached)."""
    if output_color_space in _CAM16_CONSTANTS:
        return _CAM16_CONSTANTS[output_color_space]
    import colour
    from colour.appearance.cam16 import MATRIX_16, MATRIX_INVERSE_16, VIEWING_CONDITIONS_CAM16
    from colour.appearance.ciecam02 import (
        achromatic_response_forward,
        degree_of_adaptation,
        post_adaptation_non_linear_response_compression_forward,
        viewing_conditions_dependent_parameters,
    )
    from spektrafilm.utils import gamut_compression as gc

    cs = colour.RGB_COLOURSPACES[output_color_space]
    white = np.asarray(cs.whitepoint, dtype=float)
    surround = VIEWING_CONDITIONS_CAM16["Average"]
    XYZ_w = gc._output_cs_whitepoint_xyz(output_color_space) * 100
    L_A, Y_b = np.asarray(gc._CAM16UCS_L_A), np.asarray(gc._CAM16UCS_Y_B)
    Y_w = XYZ_w[1]

    RGB_w = MATRIX_16 @ XYZ_w
    D = np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
    n, F_L, N_bb, N_cb, z = (
        float(v) for v in viewing_conditions_dependent_parameters(Y_b, np.asarray(Y_w), L_A)
    )
    D_RGB = D * Y_w / RGB_w + 1 - D
    RGB_aw = post_adaptation_non_linear_response_compression_forward(D_RGB * RGB_w, F_L)
    A_w = float(achromatic_response_forward(RGB_aw, N_bb))

    # Linear maps are exact as matrices: feed the identity through colour.
    rgb_to_xyz = np.asarray(colour.RGB_to_XYZ(
        np.eye(3), colourspace=output_color_space, illuminant=white, apply_cctf_decoding=False))
    xyz_to_rgb = np.asarray(colour.XYZ_to_RGB(
        np.eye(3), colourspace=output_color_space, illuminant=white, apply_cctf_encoding=False))

    consts = dict(
        fwd=100.0 * rgb_to_xyz @ MATRIX_16.T * D_RGB[None, :],       # rgb -> RGB_c
        inv=(MATRIX_INVERSE_16.T / D_RGB[:, None]) @ xyz_to_rgb / 100.0,  # RGB_c -> rgb
        F_L=F_L, n=n, N_bb=N_bb, N_cb=N_cb, cz=surround.c * z, N_c=float(surround.N_c),
        A_w=A_w, K=(1.64 - 0.29 ** n) ** 0.73,
        Jp_white=gc._cam16ucs_white_Jp(output_color_space),
    )
    _CAM16_CONSTANTS[output_color_space] = consts
    return consts


def _spow(cp, a, p):
    return cp.sign(a) * cp.abs(a) ** p


def _sdiv(cp, a, b):
    # colour's default "Ignore Zero Conversion" safe division.
    return cp.nan_to_num(a / b, nan=0.0, posinf=0.0, neginf=0.0)


def cuda_compress_rgb_cam16ucs_chroma(
    rgb: Any,
    output_color_space: str,
    *,
    threshold: float,
    limit: float,
    power: float,
    lightness_compression: tuple[float, float, float] | None = None,
) -> Any:
    """GPU twin of ``compress_rgb_cam16ucs_chroma``; keeps the input dtype."""
    import cupy as cp
    from spektrafilm.utils.gamut_compression import _c_max_lookup

    k = _cam16ucs_constants(output_color_space)
    dt = rgb.dtype
    F_L, N_bb, A_w, cz, K = k["F_L"], k["N_bb"], k["A_w"], k["cz"], k["K"]
    tm = 50000.0 / 13.0 * k["N_c"] * k["N_cb"]

    # --- forward: rgb -> RGB_c -> RGB_a -> J, M, h -> J'a'b'
    RGB_c = rgb @ cp.asarray(k["fwd"], dtype=dt)
    F = (F_L * cp.abs(RGB_c) / 100) ** 0.42
    RGB_a = 400 * cp.sign(RGB_c) * F / (27.13 + F) + 0.1
    del RGB_c, F
    Ra, Ga, Ba = RGB_a[..., 0], RGB_a[..., 1], RGB_a[..., 2]
    a = Ra - 12 * Ga / 11 + Ba / 11
    b = (Ra + Ga - 2 * Ba) / 9
    h = cp.degrees(cp.arctan2(b, a)) % 360
    e_t = 0.25 * (cp.cos(2 + h * (np.pi / 180)) + 3.8)
    A = (2 * Ra + Ga + Ba / 20 - 0.305) * N_bb
    J = 100 * _spow(cp, _sdiv(cp, A, A_w), cz)
    t = tm * _sdiv(cp, e_t * cp.sqrt(a * a + b * b), Ra + Ga + 21 * Ba / 20)
    del RGB_a, Ra, Ga, Ba, a, b, A, e_t
    M = _spow(cp, t, 0.9) * _spow(cp, J / 100, 0.5) * K * F_L ** 0.25
    del t
    Jp = (1 + 100 * _CAM16UCS_C1) * J / (1 + _CAM16UCS_C1 * J)
    Mp = cp.log1p(_CAM16UCS_C2 * M) / _CAM16UCS_C2
    hr = cp.radians(h)
    ap, bp = Mp * cp.cos(hr), Mp * cp.sin(hr)
    del J, M, Mp, h, hr

    if lightness_compression is not None:
        lt, ll, lp = lightness_compression
        Jp = cuda_reinhard_knee(Jp / k["Jp_white"], lt, ll, lp) * k["Jp_white"]

    # --- chroma compression against Cp_max(Jp, hp)
    Cp = cp.hypot(ap, bp)
    hp = cp.arctan2(bp, ap)
    del ap, bp
    table = _device_c_max_table("cam16ucs", output_color_space, dt)
    safe_Cp_max = cp.fmax(_c_max_lookup(Jp, hp, *table), 1e-9)
    Cp_new = cuda_reinhard_knee(Cp / safe_Cp_max, threshold, limit, power) * safe_Cp_max
    del Cp, safe_Cp_max

    # --- inverse: J'a'b' -> J, M, h -> RGB_a -> RGB_c -> rgb
    # Cp_new >= 0, so hypot/arctan2 of (Cp_new cos hp, Cp_new sin hp) is (Cp_new, hp).
    J = -Jp / (_CAM16UCS_C1 * Jp - 1 - 100 * _CAM16UCS_C1)
    M = cp.expm1(_CAM16UCS_C2 * Cp_new) / _CAM16UCS_C2
    del Jp, Cp_new
    C = M / F_L ** 0.25
    t = _spow(cp, C / (cp.sqrt(cp.maximum(J, np.finfo(np.double).eps) / 100) * K), 1 / 0.9)
    del C, M
    h = cp.degrees(hp) % 360
    hr = cp.radians(h)
    e_t = 0.25 * (cp.cos(2 + h * (np.pi / 180)) + 3.8)
    A = A_w * _spow(cp, J / 100, 1 / cz)
    del J, h, hp
    P_1 = _sdiv(cp, tm * e_t, t)
    P_2 = A / N_bb + 0.305
    P_3 = 21 / 20
    del A, e_t
    sin_hr, cos_hr = cp.sin(hr), cp.cos(hr)
    del hr
    cos_sin = _sdiv(cp, cos_hr, sin_hr)
    sin_cos = _sdiv(cp, sin_hr, cos_hr)
    nn = P_2 * (2 + P_3) * (460 / 1403)
    use_sin = cp.abs(sin_hr) >= cp.abs(cos_hr)
    use_cos = cp.abs(sin_hr) < cp.abs(cos_hr)  # both False for NaN hue -> a = b = 0, as in colour
    b_s = nn / (_sdiv(cp, P_1, sin_hr) + (2 + P_3) * (220 / 1403) * cos_sin
                - (27 / 1403) + P_3 * (6300 / 1403))
    a_c = nn / (_sdiv(cp, P_1, cos_hr) + (2 + P_3) * (220 / 1403)
                - ((27 / 1403) - P_3 * (6300 / 1403)) * sin_cos)
    a = cp.where(use_sin, b_s * cos_sin, cp.where(use_cos, a_c, 0))
    b = cp.where(use_sin, b_s, cp.where(use_cos, a_c * sin_cos, 0))
    del b_s, a_c, cos_sin, sin_cos, sin_hr, cos_hr, nn, P_1, use_sin, use_cos
    chromatic = t != 0
    a, b = a * chromatic, b * chromatic
    del t, chromatic

    RGB_a = cp.stack([
        (460 * P_2 + 451 * a + 288 * b) / 1403,
        (460 * P_2 - 891 * a - 261 * b) / 1403,
        (460 * P_2 - 220 * a - 6300 * b) / 1403,
    ], axis=-1)
    del P_2, a, b
    x = RGB_a - 0.1
    ax = cp.abs(x)
    RGB_c = cp.sign(x) * 100 / F_L * _spow(cp, 27.13 * ax / (400 - ax), 1 / 0.42)
    del RGB_a, x, ax
    return (RGB_c @ cp.asarray(k["inv"], dtype=dt)).astype(dt, copy=False)


# ---------------------------------------------------------------------------
# OkLch / OkLrab / JzAzBz chroma compression on GPU — CuPy ports of the
# matching utils.gamut_compression functions (colour's XYZ_to_Oklab /
# XYZ_to_Jzazbz "Safdar 2017" and their inverses).
# ---------------------------------------------------------------------------

_PERCEPTUAL_CONSTANTS: dict = {}


def _device_c_max_table(space: str, output_color_space: str, dt) -> tuple:
    import cupy as cp
    from spektrafilm.utils.gamut_compression import _get_output_c_max_table
    key = (space, output_color_space, np.dtype(dt).str)
    if key not in _DEVICE_TABLES:
        L_grid, h_grid, table = _get_output_c_max_table(space, output_color_space)
        _DEVICE_TABLES[key] = (L_grid, h_grid, cp.asarray(table, dtype=dt))
    return _DEVICE_TABLES[key]


def _rgb_xyz_matrices(output_color_space: str) -> tuple[np.ndarray, np.ndarray]:
    """(rgb->xyz, xyz->rgb) as right-multiply matrices, exact w.r.t. colour."""
    import colour
    white = np.asarray(colour.RGB_COLOURSPACES[output_color_space].whitepoint, dtype=float)
    to_xyz = np.asarray(colour.RGB_to_XYZ(
        np.eye(3), colourspace=output_color_space, illuminant=white, apply_cctf_decoding=False))
    to_rgb = np.asarray(colour.XYZ_to_RGB(
        np.eye(3), colourspace=output_color_space, illuminant=white, apply_cctf_encoding=False))
    return to_xyz, to_rgb


def _oklab_constants(output_color_space: str) -> dict:
    key = ("oklab", output_color_space)
    if key not in _PERCEPTUAL_CONSTANTS:
        from colour.models.oklab import (
            MATRIX_1_LMS_TO_XYZ, MATRIX_1_XYZ_TO_LMS, MATRIX_2_LAB_TO_LMS, MATRIX_2_LMS_TO_LAB,
        )
        to_xyz, to_rgb = _rgb_xyz_matrices(output_color_space)
        _PERCEPTUAL_CONSTANTS[key] = dict(
            fwd1=to_xyz @ MATRIX_1_XYZ_TO_LMS.T, fwd2=MATRIX_2_LMS_TO_LAB.T,
            inv1=MATRIX_2_LAB_TO_LMS.T, inv2=MATRIX_1_LMS_TO_XYZ.T @ to_rgb,
        )
    return _PERCEPTUAL_CONSTANTS[key]


def _jzazbz_constants(output_color_space: str) -> dict:
    key = ("jzazbz", output_color_space)
    if key not in _PERCEPTUAL_CONSTANTS:
        from colour.models.jzazbz import (
            CONSTANTS_JZAZBZ_SAFDAR2017 as C,
            MATRIX_JZAZBZ_IZAZBZ_TO_LMS_P_SAFDAR2017,
            MATRIX_JZAZBZ_LMS_P_TO_IZAZBZ_SAFDAR2017,
            MATRIX_JZAZBZ_XYZ_TO_LMS,
        )
        from spektrafilm.utils import gamut_compression as gc
        to_xyz, to_rgb = _rgb_xyz_matrices(output_color_space)
        # X' = bX - (b-1)Z, Y' = gY - (g-1)X, Z' = Z is linear, so fold it into the matrices.
        P = np.array([[C.b, 0, -(C.b - 1)], [-(C.g - 1), C.g, 0], [0, 0, 1]])
        Y_w = gc._JZAZBZ_Y_W_CDM2
        _PERCEPTUAL_CONSTANTS[key] = dict(
            fwd1=Y_w * to_xyz @ P.T @ MATRIX_JZAZBZ_XYZ_TO_LMS.T,
            fwd2=MATRIX_JZAZBZ_LMS_P_TO_IZAZBZ_SAFDAR2017.T,
            inv1=MATRIX_JZAZBZ_IZAZBZ_TO_LMS_P_SAFDAR2017.T,
            inv2=np.linalg.inv(MATRIX_JZAZBZ_XYZ_TO_LMS).T @ np.linalg.inv(P).T @ to_rgb / Y_w,
            c1=C.c_1, c2=C.c_2, c3=C.c_3, m1=C.m_1, m2=C.m_2, d=C.d, d0=C.d_0,
            Jz_white=gc._jzazbz_white_Jz(output_color_space),
        )
    return _PERCEPTUAL_CONSTANTS[key]


def _mat(cp, x, m):
    return x @ cp.asarray(m, dtype=x.dtype)


def _compress_polar_chroma(cp, L_lookup, a, b, space, output_color_space, threshold, limit, power):
    """Shared step: soft-compress chroma against the C_max(L, h) table; returns (a, b)."""
    from spektrafilm.utils.gamut_compression import _c_max_lookup
    C = cp.hypot(a, b)
    h = cp.arctan2(b, a)
    table = _device_c_max_table(space, output_color_space, a.dtype)
    safe_C_max = cp.fmax(_c_max_lookup(L_lookup, h, *table), 1e-9)
    C_new = cuda_reinhard_knee(C / safe_C_max, threshold, limit, power) * safe_C_max
    return C_new * cp.cos(h), C_new * cp.sin(h)


def _oklab_chroma(rgb, output_color_space, threshold, limit, power, lightness_compression, space):
    import cupy as cp
    k = _oklab_constants(output_color_space)
    lab = _mat(cp, cp.cbrt(_mat(cp, rgb, k["fwd1"])), k["fwd2"])  # cbrt == colour's spow(x, 1/3)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    if lightness_compression is not None:
        L = cuda_reinhard_knee(L, *lightness_compression)  # L_white = 1
    if space == "oklrab":
        from spektrafilm.utils.gamut_compression import _oklab_L_to_oklrab_Lr
        L_lookup = _oklab_L_to_oklrab_Lr(L)
    else:
        L_lookup = L
    a, b = _compress_polar_chroma(cp, L_lookup, a, b, space, output_color_space, threshold, limit, power)
    lms = _mat(cp, cp.stack([L, a, b], axis=-1), k["inv1"])
    return _mat(cp, lms * lms * lms, k["inv2"])


def cuda_compress_rgb_oklch_chroma(rgb, output_color_space, *, threshold, limit, power,
                                   lightness_compression=None):
    """GPU twin of ``compress_rgb_oklch_chroma``; keeps the input dtype."""
    return _oklab_chroma(rgb, output_color_space, threshold, limit, power, lightness_compression, "oklch")


def cuda_compress_rgb_oklrab_chroma(rgb, output_color_space, *, threshold, limit, power,
                                    lightness_compression=None):
    """GPU twin of ``compress_rgb_oklrab_chroma``; keeps the input dtype."""
    return _oklab_chroma(rgb, output_color_space, threshold, limit, power, lightness_compression, "oklrab")


def cuda_compress_rgb_jzazbz_chroma(rgb, output_color_space, *, threshold, limit, power,
                                    lightness_compression=None):
    """GPU twin of ``compress_rgb_jzazbz_chroma``; keeps the input dtype."""
    import cupy as cp
    out_dtype = rgb.dtype
    rgb = rgb.astype(cp.float64, copy=False)  # PQ in float32 drifts ~1e-3 near black; f64 costs ~70 ms at 3.4 MP
    k = _jzazbz_constants(output_color_space)
    c1, c2, c3, m1, m2, d, d0 = (k[n] for n in ("c1", "c2", "c3", "m1", "m2", "d", "d0"))

    Y_p = _spow(cp, _mat(cp, rgb, k["fwd1"]) / 10000, m1)  # PQ encode (eotf_inverse_ST2084)
    lms_p = _spow(cp, (c1 + c2 * Y_p) / (c3 * Y_p + 1), m2)
    del Y_p
    iab = _mat(cp, lms_p, k["fwd2"])
    I_z, a, b = iab[..., 0], iab[..., 1], iab[..., 2]
    Jz = (1 + d) * I_z / (1 + d * I_z) - d0

    if lightness_compression is not None:
        Jz = cuda_reinhard_knee(Jz / k["Jz_white"], *lightness_compression) * k["Jz_white"]
    a, b = _compress_polar_chroma(cp, Jz, a, b, "jzazbz", output_color_space, threshold, limit, power)

    I_z = (Jz + d0) / (1 + d - d * (Jz + d0))
    V_p = _spow(cp, _mat(cp, cp.stack([I_z, a, b], axis=-1), k["inv1"]), 1 / m2)
    lms = 10000 * _spow(cp, cp.maximum(0, V_p - c1) / (c2 - c3 * V_p), 1 / m1)  # eotf_ST2084
    return _mat(cp, lms, k["inv2"]).astype(out_dtype, copy=False)
