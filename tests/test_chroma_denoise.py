"""Unit and integration tests for Chroma Denoiser in Spektrafilm."""

from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.utils.chroma_denoise import chroma_denoise


def test_chroma_denoise_luminance_preservation():
    """Verify that luminance (Y) is 100% preserved to machine precision."""
    rng = np.random.default_rng(42)
    img = rng.uniform(0.1, 0.9, size=(64, 64, 3)).astype(np.float32)

    denoised = chroma_denoise(img, strength=1.0, radius=2.5)

    y_in = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    y_out = 0.299 * denoised[..., 0] + 0.587 * denoised[..., 1] + 0.114 * denoised[..., 2]

    # Luminance must match to within floating point tolerances (< 1e-6)
    np.testing.assert_allclose(y_out, y_in, atol=1e-6)


def test_chroma_denoise_reduces_chroma_variance():
    """Verify that high-frequency chrominance noise variance is significantly reduced."""
    rng = np.random.default_rng(123)
    # Neutral grey background
    img = np.full((128, 128, 3), 0.5, dtype=np.float32)
    # Add strong zero-sum chromatic noise (R/B fluctuations without changing Y)
    chroma_noise = rng.normal(0.0, 0.05, size=(128, 128)).astype(np.float32)
    img[..., 0] += chroma_noise
    img[..., 2] -= chroma_noise
    # Balance green so Y is strictly constant
    img[..., 1] = (0.5 - 0.299 * img[..., 0] - 0.114 * img[..., 2]) / 0.587

    var_cb_before = np.var(img[..., 2] - 0.5)
    var_cr_before = np.var(img[..., 0] - 0.5)

    denoised = chroma_denoise(img, strength=1.0, radius=3.0)

    var_cb_after = np.var(denoised[..., 2] - 0.5)
    var_cr_after = np.var(denoised[..., 0] - 0.5)

    # Chrominance variance should be reduced by more than 75%
    assert var_cb_after < var_cb_before * 0.25
    assert var_cr_after < var_cr_before * 0.25


def test_chroma_denoise_bypass_when_strength_or_radius_zero():
    """Verify that strength=0.0 or radius=0.0 bypasses filtering with exact identity."""
    img = np.ones((16, 16, 3), dtype=np.float32) * 0.4
    res_s0 = chroma_denoise(img, strength=0.0)
    assert res_s0 is img

    res_r0 = chroma_denoise(img, strength=1.0, radius=0.0)
    assert res_r0 is img


def test_chroma_denoise_alpha_channel_passthrough():
    """Verify that an alpha or 4th channel is preserved unchanged."""
    img = np.ones((16, 16, 4), dtype=np.float32) * 0.5
    img[..., 3] = 0.8  # Alpha channel
    denoised = chroma_denoise(img, strength=1.0)
    assert denoised.shape == (16, 16, 4)
    np.testing.assert_allclose(denoised[..., 3], 0.8)


def test_chroma_denoise_gpu_cpu_parity():
    """Verify CuPy GPU execution gives parity with NumPy CPU execution."""
    try:
        import cupy as cp
        from spektrafilm.cuda.device import is_cuda_available

        if not is_cuda_available():
            pytest.skip("CUDA device unavailable")
    except ImportError:
        pytest.skip("CuPy not installed")

    rng = np.random.default_rng(999)
    img_cpu = rng.uniform(0.1, 0.9, size=(64, 64, 3)).astype(np.float32)
    img_gpu = cp.asarray(img_cpu)

    res_cpu = chroma_denoise(img_cpu, strength=1.0, radius=2.5)
    res_gpu = chroma_denoise(img_gpu, strength=1.0, radius=2.5)

    assert isinstance(res_gpu, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(res_gpu), res_cpu, atol=1e-5)


def test_chroma_denoise_pipeline_preprocess_integration():
    """Verify that SimulationPipeline respects IOParams.chroma_denoise."""
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm.runtime.pipeline import SimulationPipeline

    params = init_params(film_profile="kodak_portra_400", print_profile="kodak_portra_endura")
    params.io.chroma_denoise = True
    params.io.chroma_denoise_strength = 1.0

    digested = digest_params(params)
    pipeline = SimulationPipeline(digested)

    # Process small test image
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    processed = pipeline.process(img)

    assert processed.ndim == 3
    assert processed.shape[2] == 3
    assert np.all(np.isfinite(processed))


def test_chroma_denoise_raw_processor_integration(monkeypatch):
    """Verify that load_and_process_raw_file runs chroma denoising when requested."""
    from types import SimpleNamespace
    from spektrafilm.utils import raw_file_processor

    raw_image = np.array(
        [
            [[8192, 8192, 8192], [12288, 8192, 6144]],
            [[16384, 12288, 8192], [8192, 12288, 16384]],
        ],
        dtype=np.uint16,
    )

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **_kwargs):
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )

    result_raw = raw_file_processor.load_and_process_raw_file(
        'synthetic.nef',
        white_balance='daylight',
        chroma_denoise=True,
        chroma_denoise_strength=1.0,
    )

    assert result_raw.shape == (2, 2, 3)
    assert result_raw.dtype == np.float32
    assert np.all(np.isfinite(result_raw))


def test_chroma_denoise_strength_greater_than_one_reduces_variance_without_inversion():
    """Verify that strength > 1.0 (e.g. 1.5, 2.0) smooths noise monotonically and does not amplify or invert."""
    rng = np.random.default_rng(42)
    img = np.full((64, 64, 3), 0.5, dtype=np.float32)
    noise = rng.normal(0.0, 0.05, size=(64, 64, 3)).astype(np.float32)
    noisy = np.clip(img + noise, 0.0, 1.0)

    var_noisy = np.var(noisy[..., 0] - noisy[..., 1])
    res_s1 = chroma_denoise(noisy, strength=1.0)
    var_s1 = np.var(res_s1[..., 0] - res_s1[..., 1])

    res_s2 = chroma_denoise(noisy, strength=2.0)
    var_s2 = np.var(res_s2[..., 0] - res_s2[..., 1])

    # Strength 1.0 must reduce chroma variance significantly
    assert var_s1 < var_noisy * 0.3
    # Strength 2.0 must reduce variance even further and never amplify noise
    assert var_s2 <= var_s1
    assert var_s2 < var_noisy * 0.3

