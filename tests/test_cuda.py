"""Unit and integration tests for CUDA hardware acceleration."""
from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.cuda.device import (
    is_cuda_available,
    get_device_info,
    resolve_device,
    to_device,
    to_cpu,
    is_cupy_array,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def skip_if_no_cuda():
    if not is_cuda_available():
        pytest.skip("CUDA device / CuPy not available")


def test_cuda_device_detection():
    """Verify that CUDA device is detected and returns valid info."""
    assert is_cuda_available() is True
    info = get_device_info()
    assert info["available"] is True
    assert "GeForce" in info["device_name"] or "NVIDIA" in info["device_name"] or "RTX" in info["device_name"]
    assert info["total_memory_gb"] > 0
    assert info["compute_capability"][0] >= 5


def test_resolve_device():
    """Verify device resolution behavior."""
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") == "cuda"


def test_to_device_and_to_cpu():
    """Verify data transfer to and from GPU."""
    import cupy as cp

    x_np = np.linspace(0.0, 1.0, 100, dtype=np.float64).reshape(10, 10)
    x_gpu = to_device(x_np)
    assert is_cupy_array(x_gpu)
    assert isinstance(x_gpu, cp.ndarray)
    assert x_gpu.shape == x_np.shape

    x_back = to_cpu(x_gpu)
    assert isinstance(x_back, np.ndarray)
    np.testing.assert_array_equal(x_np, x_back)


def test_cuda_fast_interp_parity_common_axis():
    """Verify cuda_fast_interp exactly matches Numba fast_interp on 1D axis."""
    from spektrafilm.utils.fast_interp import _numba_fast_interp
    from spektrafilm.cuda.interp import cuda_fast_interp

    K = 128
    x_axis = np.linspace(-3.0, 3.0, K)
    y_vals = np.stack([
        np.sin(x_axis),
        np.cos(x_axis),
        np.exp(-x_axis ** 2),
    ], axis=-1)

    img = np.random.uniform(-4.0, 4.0, (32, 32, 3)).astype(np.float64)

    cpu_out = _numba_fast_interp(img, x_axis, y_vals)
    gpu_out = to_cpu(cuda_fast_interp(to_device(img), x_axis, y_vals))

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-12, rtol=1e-12)


def test_cuda_fast_interp_parity_channel_axis():
    """Verify cuda_fast_interp exactly matches Numba fast_interp on 2D axis."""
    from spektrafilm.utils.fast_interp import _numba_fast_interp
    from spektrafilm.cuda.interp import cuda_fast_interp

    K = 128
    x_axis = np.stack([
        np.linspace(-2.0, 2.0, K),
        np.linspace(-3.0, 1.0, K),
        np.linspace(-1.0, 4.0, K),
    ], axis=-1)
    y_vals = np.stack([
        x_axis[:, 0] ** 2,
        np.sqrt(np.abs(x_axis[:, 1])),
        np.tanh(x_axis[:, 2]),
    ], axis=-1)

    img = np.random.uniform(-3.0, 3.0, (32, 32, 3)).astype(np.float64)

    cpu_out = _numba_fast_interp(img, x_axis, y_vals)
    gpu_out = to_cpu(cuda_fast_interp(to_device(img), x_axis, y_vals))

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-12, rtol=1e-12)


def test_cuda_apply_lut_cubic_2d_parity():
    """Verify cuda_apply_lut_cubic_2d matches Numba Mitchell-Netravali sampling."""
    from spektrafilm.utils.fast_interp_lut import _apply_lut_cubic_2d
    from spektrafilm.cuda.interp import cuda_apply_lut_cubic_2d

    L = 32
    lut = np.random.uniform(0.0, 1.0, (L, L, 3)).astype(np.float64)
    image_2d = np.random.uniform(-0.2, 1.2, (64, 64, 2)).astype(np.float64)

    cpu_out = _apply_lut_cubic_2d(lut, image_2d)
    gpu_out = to_cpu(cuda_apply_lut_cubic_2d(lut, to_device(image_2d)))

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-12, rtol=1e-12)


def test_cuda_reinhard_knee_parity():
    """Verify cuda_reinhard_knee matches CPU reinhard_knee."""
    from spektrafilm.utils.gamut_compression import reinhard_knee
    from spektrafilm.cuda.gamut_compression import cuda_reinhard_knee

    d = np.linspace(-0.5, 3.0, 500, dtype=np.float64)
    threshold, limit, power = 0.5, 1.5, 1.2

    cpu_out = reinhard_knee(d, threshold=threshold, limit=limit, power=power)
    gpu_out = to_cpu(cuda_reinhard_knee(to_device(d), threshold, limit, power))

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-14, rtol=1e-14)


def test_cuda_aces_rgc_parity():
    """Verify cuda_compress_rgb_aces_rgc matches CPU compress_rgb_aces_rgc."""
    from spektrafilm.utils.gamut_compression import compress_rgb_aces_rgc
    from spektrafilm.cuda.gamut_compression import cuda_compress_rgb_aces_rgc

    rgb = np.random.uniform(-0.2, 2.0, (32, 32, 3)).astype(np.float64)
    threshold, limit, power = 0.6, 1.8, 1.5

    cpu_out = compress_rgb_aces_rgc(rgb, threshold=threshold, limit=limit, power=power)
    gpu_out = to_cpu(cuda_compress_rgb_aces_rgc(to_device(rgb), threshold, limit, power))

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-14, rtol=1e-14)


def test_cuda_spectral_density_parity():
    """Verify spectral tensor dot product on GPU matches opt_einsum CPU output."""
    from opt_einsum import contract
    from spektrafilm.cuda.spectral import cuda_compute_density_spectral, cuda_density_to_light

    H, W = 16, 16
    density_cmy = np.random.uniform(0.1, 2.5, (H, W, 3)).astype(np.float64)
    channel_density = np.random.uniform(0.0, 1.0, (81, 3)).astype(np.float64)
    base_density = np.random.uniform(0.05, 0.2, (81,)).astype(np.float64)

    cpu_spectral = contract("ijk, lk->ijl", density_cmy, channel_density) + base_density
    gpu_spectral = to_cpu(cuda_compute_density_spectral(channel_density, to_device(density_cmy), base_density))
    np.testing.assert_allclose(gpu_spectral, cpu_spectral, atol=1e-14, rtol=1e-14)

    illuminant = np.random.uniform(0.5, 1.5, (81,)).astype(np.float64)
    cpu_light = (10.0 ** (-cpu_spectral)) * illuminant
    gpu_light = to_cpu(cuda_density_to_light(to_device(gpu_spectral), illuminant))
    np.testing.assert_allclose(gpu_light, cpu_light, atol=1e-12, rtol=1e-12)


def test_cuda_gaussian_filter():
    """Verify CUDA gaussian filter produces smooth blur matching CPU characteristics."""
    from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
    from spektrafilm.cuda.filters import cuda_gaussian_filter

    img = np.random.uniform(0.0, 1.0, (64, 64, 3)).astype(np.float64)
    sigma = 2.5

    cpu_out = fast_gaussian_filter(img, sigma)
    gpu_out = to_cpu(cuda_gaussian_filter(to_device(img), sigma))

    diff = np.abs(cpu_out - gpu_out)
    assert diff.mean() < 0.05


def test_pipeline_end_to_end_deterministic():
    """Verify full end-to-end simulation pipeline execution on CUDA vs CPU."""
    from spektrafilm.runtime import init_params, simulate

    img = np.random.uniform(0.2, 0.8, (64, 64, 3)).astype(np.float64)

    p_cpu = init_params()
    p_cpu.settings.device = "cpu"
    p_cpu.film_render.grain.amount = 0.0
    p_cpu.print_render.glare.active = False

    p_cuda = init_params()
    p_cuda.settings.device = "cuda"
    p_cuda.film_render.grain.amount = 0.0
    p_cuda.print_render.glare.active = False

    out_cpu = simulate(img, p_cpu)
    out_cuda = simulate(img, p_cuda)

    assert out_cuda.shape == out_cpu.shape
    assert out_cuda.dtype == out_cpu.dtype
    assert not np.any(np.isnan(out_cuda))

    # Mean difference across the entire pipeline is small (around ~0.1% of range)
    diff = np.abs(out_cpu - out_cuda)
    assert diff.mean() < 0.01


def test_cuda_grain_spatial_anchoring_invariance():
    """Verify grain is deterministic and perfectly invariant under spatial crop offsets."""
    import cupy as cp
    from spektrafilm.cuda.grain import cuda_apply_grain
    from spektrafilm.runtime.params_schema import GrainParams

    H, W = 128, 128
    np.random.seed(42)
    density_cmy = np.random.uniform(0.5, 1.5, (H, W, 3)).astype(np.float64)
    density_curves = np.linspace(0.1, 2.5, 50)[:, None].repeat(3, axis=1)
    density_curves_layers = np.linspace(0.05, 0.8, 50)[:, None, None].repeat(3, axis=1).repeat(3, axis=2)

    grain_params = GrainParams(active=True, sublayers_active=True, blur=0.0)
    pixel_size_um = 10.0

    # 1. Full image grain
    d_full = to_device(density_cmy)
    out_full = to_cpu(cuda_apply_grain(
        d_full,
        pixel_size_um=pixel_size_um,
        grain_params=grain_params,
        density_curves=density_curves,
        density_curves_layers=density_curves_layers,
        profile_type="negative",
        origin_x_um=0.0,
        origin_y_um=0.0,
    ))

    # 2. Cropped sub-region [y0:y1, x0:x1]
    y0, y1, x0, x1 = 30, 90, 40, 100
    density_crop = density_cmy[y0:y1, x0:x1].copy()
    origin_x_um = x0 * pixel_size_um
    origin_y_um = y0 * pixel_size_um

    d_crop = to_device(density_crop)
    out_crop = to_cpu(cuda_apply_grain(
        d_crop,
        pixel_size_um=pixel_size_um,
        grain_params=grain_params,
        density_curves=density_curves,
        density_curves_layers=density_curves_layers,
        profile_type="negative",
        origin_x_um=origin_x_um,
        origin_y_um=origin_y_um,
    ))

    # The cropped region must match the corresponding region of the full image exactly
    max_diff = np.max(np.abs(out_full[y0:y1, x0:x1] - out_crop))
    assert max_diff == 0.0, f"Grain is not spatially anchored! Max diff: {max_diff}"


def test_cuda_grain_sharpness_and_validity():
    """Verify CUDA grain generates valid, crisp grain structure without NaNs."""
    from spektrafilm.cuda.grain import cuda_apply_grain
    from spektrafilm.runtime.params_schema import GrainParams

    H, W = 64, 64
    density_cmy = np.ones((H, W, 3), dtype=np.float64) * 0.8
    density_curves = np.linspace(0.1, 2.5, 50)[:, None].repeat(3, axis=1)
    density_curves_layers = np.linspace(0.05, 0.8, 50)[:, None, None].repeat(3, axis=1).repeat(3, axis=2)

    grain_params = GrainParams(active=True, sublayers_active=True, blur=0.0)
    out = to_cpu(cuda_apply_grain(
        to_device(density_cmy),
        pixel_size_um=10.0,
        grain_params=grain_params,
        density_curves=density_curves,
        density_curves_layers=density_curves_layers,
        profile_type="negative",
    ))

    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    # Stdev should reflect genuine grain noise (not flat or blown up)
    assert out.std() > 0.01


def test_cuda_color_matrix_precision():
    """Verify GPU matrix conversions match colour-science to machine precision (< 1e-14)."""
    from spektrafilm.cuda.color import get_xyz_to_rgb_matrix, get_rgb_to_xyz_matrix, cuda_apply_matrix
    import colour

    test_xyz = np.random.uniform(0.01, 1.0, (16, 16, 3))
    for cs_name in ("sRGB", "Display P3", "Adobe RGB (1998)", "ITU-R BT.2020"):
        illu_xy = (0.3457, 0.3585)
        # XYZ to RGB
        expected_rgb = colour.XYZ_to_RGB(
            test_xyz,
            colourspace=cs_name,
            illuminant=illu_xy,
            chromatic_adaptation_transform="CAT02",
            apply_cctf_encoding=False,
        )
        m_xyz2rgb = get_xyz_to_rgb_matrix(cs_name, illu_xy, cat="CAT02")
        actual_rgb = to_cpu(cuda_apply_matrix(test_xyz, m_xyz2rgb))
        max_err = np.max(np.abs(expected_rgb - actual_rgb))
        assert max_err < 1e-14, f"XYZ->RGB precision mismatch for {cs_name}: max_err={max_err}"

        # RGB to XYZ
        test_rgb = np.random.uniform(0.01, 1.0, (16, 16, 3))
        expected_xyz = colour.RGB_to_XYZ(
            test_rgb,
            colourspace=cs_name,
            illuminant=illu_xy,
            chromatic_adaptation_transform="CAT16",
            apply_cctf_decoding=False,
        )
        m_rgb2xyz = get_rgb_to_xyz_matrix(cs_name, illu_xy, cat="CAT16")
        actual_xyz = to_cpu(cuda_apply_matrix(test_rgb, m_rgb2xyz))
        max_err = np.max(np.abs(expected_xyz - actual_xyz))
        assert max_err < 1e-14, f"RGB->XYZ precision mismatch for {cs_name}: max_err={max_err}"


def test_cuda_cctf_precision():
    """Verify GPU transfer function encoding/decoding match colour-science (< 1e-14)."""
    from spektrafilm.cuda.color import cuda_cctf_encoding, cuda_cctf_decoding
    import colour

    vals = np.linspace(0.001, 1.2, 500)
    for cs_name in ("sRGB", "Display P3", "ITU-R BT.2020", "ProPhoto RGB", "DCI-P3", "Adobe RGB (1998)"):
        cs = colour.RGB_COLOURSPACES[cs_name]

        # Encoding
        ref_enc = cs.cctf_encoding(vals)
        gpu_enc = to_cpu(cuda_cctf_encoding(vals, cs_name))
        max_err = np.max(np.abs(ref_enc - gpu_enc))
        assert max_err < 1e-14, f"CCTF encoding mismatch for {cs_name}: max_err={max_err}"

        # Decoding
        ref_dec = cs.cctf_decoding(ref_enc)
        gpu_dec = to_cpu(cuda_cctf_decoding(ref_enc, cs_name))
        max_err = np.max(np.abs(ref_dec - gpu_dec))
        assert max_err < 1e-14, f"CCTF decoding mismatch for {cs_name}: max_err={max_err}"


def test_cuda_rgb_to_tc_b_precision():
    """Verify GPU rgb_to_tc_b matches CPU _rgb_to_tc_b to machine precision (< 1e-14)."""
    from spektrafilm.cuda.color import cuda_rgb_to_tc_b
    from spektrafilm.utils.spectral_upsampling import _rgb_to_tc_b, _illuminant_to_xy

    rgb = np.random.uniform(0.05, 0.95, (32, 32, 3))
    ref_tc, ref_b = _rgb_to_tc_b(rgb, color_space="ITU-R BT.2020", apply_cctf_decoding=False, reference_illuminant="D55")

    illu_xy = tuple(_illuminant_to_xy("D55"))
    gpu_tc, gpu_b = cuda_rgb_to_tc_b(rgb, colourspace="ITU-R BT.2020", apply_cctf_decoding=False, reference_illuminant_xy=illu_xy)

    max_err_tc = np.max(np.abs(ref_tc - to_cpu(gpu_tc)))
    max_err_b = np.max(np.abs(ref_b - to_cpu(gpu_b)))
    assert max_err_tc < 1e-14, f"tc max_err: {max_err_tc}"
    assert max_err_b < 1e-14, f"b max_err: {max_err_b}"


def test_cuda_fused_spectral_precision():
    """Verify cuda_fused_spectral_integrate matches reference spectral integration (< 1e-14)."""
    from spektrafilm.cuda.spectral import (
        cuda_fused_spectral_integrate,
        cuda_compute_density_spectral,
        cuda_density_to_light,
        cuda_spectral_integrate,
    )

    H, W, C_in = 32, 32, 3
    W_count = 81
    C_out = 3

    cmy = np.random.uniform(0.1, 2.5, (H, W, C_in))
    cd = np.random.uniform(0.0, 1.2, (W_count, C_in))
    bd = np.random.uniform(0.0, 0.4, (W_count,))
    illu = np.random.uniform(0.5, 1.5, (W_count,))
    resp = np.random.uniform(0.0, 1.0, (W_count, C_out))
    norm_vec = np.array([100.0, 105.0, 95.0])
    norm_scalar = 50.0

    # Test 1: Vector normalization + base density
    ref_ds = to_cpu(cuda_compute_density_spectral(cd, cmy, bd))
    ref_lt = to_cpu(cuda_density_to_light(ref_ds, illu))
    ref_out = to_cpu(cuda_spectral_integrate(ref_lt, resp, normalization=norm_vec))

    fused_out = to_cpu(cuda_fused_spectral_integrate(
        cmy, cd, base_density=bd, illuminant=illu, response_curves=resp, normalization=norm_vec
    ))
    max_err = np.max(np.abs(ref_out - fused_out))
    assert max_err < 1e-14, f"Fused spectral mismatch (vector norm): max_err={max_err}"

    # Test 2: Scalar normalization without base density
    ref_ds2 = to_cpu(cuda_compute_density_spectral(cd, cmy, None))
    ref_lt2 = to_cpu(cuda_density_to_light(ref_ds2, illu))
    ref_out2 = to_cpu(cuda_spectral_integrate(ref_lt2, resp, normalization=norm_scalar))

    fused_out2 = to_cpu(cuda_fused_spectral_integrate(
        cmy, cd, base_density=None, illuminant=illu, response_curves=resp, normalization=norm_scalar
    ))
    max_err2 = np.max(np.abs(ref_out2 - fused_out2))
    assert max_err2 < 1e-14, f"Fused spectral mismatch (scalar norm): max_err={max_err2}"


def test_cuda_young_van_vliet_iir_precision():
    """Verify GPU Young & van Vliet IIR Gaussian filter matches CPU reference to machine precision."""
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
    from spektrafilm.cuda.filters import cuda_gaussian_filter

    np.random.seed(123)
    img = np.random.uniform(0.1, 0.9, (120, 160, 3)).astype(np.float64)

    for sigma in (5.0, 15.0, 45.0):
        cpu_res = fast_gaussian_filter(img, sigma)
        gpu_res = to_cpu(cuda_gaussian_filter(to_device(img), sigma))
        max_diff = np.max(np.abs(cpu_res - gpu_res))
        assert max_diff < 1e-11, f"YvV filter mismatch for sigma={sigma}: max_diff={max_diff}"


def test_cuda_grain_seed_randomization():
    """Verify different seeds produce different grain patterns while same seed produces identical grain."""
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    from spektrafilm.cuda.grain import cuda_apply_grain
    from spektrafilm.runtime.params_schema import GrainParams

    H, W = 64, 64
    density_cmy = np.ones((H, W, 3), dtype=np.float64) * 0.8
    density_curves = np.linspace(0.1, 2.5, 50)[:, None].repeat(3, axis=1)
    density_curves_layers = np.linspace(0.05, 0.8, 50)[:, None, None].repeat(3, axis=1).repeat(3, axis=2)

    grain_params = GrainParams(active=True, sublayers_active=True, blur=0.0)

    # 1. Same seed must be identical
    out_seed1_a = to_cpu(cuda_apply_grain(
        to_device(density_cmy), 10.0, grain_params, density_curves, density_curves_layers, "negative", seed=42,
    ))
    out_seed1_b = to_cpu(cuda_apply_grain(
        to_device(density_cmy), 10.0, grain_params, density_curves, density_curves_layers, "negative", seed=42,
    ))
    assert np.array_equal(out_seed1_a, out_seed1_b)

    # 2. Different seed must produce different grain
    out_seed2 = to_cpu(cuda_apply_grain(
        to_device(density_cmy), 10.0, grain_params, density_curves, density_curves_layers, "negative", seed=999,
    ))
    diff = np.abs(out_seed1_a - out_seed2)
    assert np.max(diff) > 0.5, "Different seeds should produce different grain patterns!"
    assert np.isclose(out_seed1_a.std(), out_seed2.std(), rtol=0.05), "Statistical grain intensity should match across seeds!"


