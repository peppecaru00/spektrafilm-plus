import numpy as np
import pytest

from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation
from spektrafilm.utils import spectral_upsampling as spectral_upsampling_module


pytestmark = pytest.mark.unit


def test_rgb_to_raw_hanatos2025_computes_tc_lut_when_missing(monkeypatch):
    sensitivity = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    rgb = np.zeros((2, 3, 3), dtype=np.float64)

    def fake_rgb_to_tc_b(data, **_kwargs):
        tc = np.zeros(data.shape[:-1] + (2,), dtype=np.float64)
        if data.shape == (1, 1, 3):
            scale = np.ones((1, 1), dtype=np.float64)
        else:
            scale = np.full(data.shape[:-1], 2.0, dtype=np.float64)
        return tc, scale

    lut_calls = []

    def fake_compute_hanatos2025_tc_lut(arg_sensitivity, _adaptation):
        lut_calls.append(arg_sensitivity.copy())
        return np.zeros((2, 2, 3), dtype=np.float64)

    def fake_apply_lut_cubic_2d(_tc_lut, tc):
        lut_raw = np.empty(tc.shape[:-1] + (3,), dtype=np.float64)
        lut_raw[..., 0] = 2.0
        lut_raw[..., 1] = 4.0
        lut_raw[..., 2] = 6.0
        return lut_raw

    monkeypatch.setattr(spectral_upsampling_module, '_rgb_to_tc_b', fake_rgb_to_tc_b)
    monkeypatch.setattr(spectral_upsampling_module, 'compute_hanatos2025_tc_lut', fake_compute_hanatos2025_tc_lut)
    monkeypatch.setattr(spectral_upsampling_module, 'apply_lut_cubic_2d', fake_apply_lut_cubic_2d)

    raw = spectral_upsampling_module.rgb_to_raw_hanatos2025(
        rgb,
        sensitivity,
        color_space='sRGB',
        apply_cctf_decoding=False,
        reference_illuminant='D65',
    )

    assert len(lut_calls) == 1
    np.testing.assert_allclose(lut_calls[0], sensitivity)
    expected = np.empty_like(raw)
    expected[..., 0] = 4.0
    expected[..., 1] = 8.0
    expected[..., 2] = 12.0
    assert raw.shape == (2, 3, 3)
    np.testing.assert_allclose(raw, expected)


def test_rgb_to_raw_hanatos2025_lut_path_supports_image_rgb(monkeypatch):
    sensitivity = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    rgb = np.zeros((2, 3, 3), dtype=np.float64)

    def fake_rgb_to_tc_b(data, **_kwargs):
        tc = np.zeros(data.shape[:-1] + (2,), dtype=np.float64)
        if data.shape == (1, 1, 3):
            scale = np.ones((1, 1), dtype=np.float64)
        else:
            scale = np.full(data.shape[:-1], 2.0, dtype=np.float64)
        return tc, scale

    def fake_apply_lut_cubic_2d(_tc_lut, tc):
        lut_raw = np.empty(tc.shape[:-1] + (3,), dtype=np.float64)
        lut_raw[..., 0] = 2.0
        lut_raw[..., 1] = 4.0
        lut_raw[..., 2] = 6.0
        return lut_raw

    monkeypatch.setattr(spectral_upsampling_module, '_rgb_to_tc_b', fake_rgb_to_tc_b)
    monkeypatch.setattr(spectral_upsampling_module, 'apply_lut_cubic_2d', fake_apply_lut_cubic_2d)

    raw = spectral_upsampling_module.rgb_to_raw_hanatos2025(
        rgb,
        sensitivity,
        color_space='sRGB',
        apply_cctf_decoding=False,
        reference_illuminant='D65',
        tc_lut=np.zeros((2, 2, 3), dtype=np.float64),
    )

    expected = np.empty_like(raw)
    expected[..., 0] = 4.0
    expected[..., 1] = 8.0
    expected[..., 2] = 12.0
    assert raw.shape == (2, 3, 3)
    np.testing.assert_allclose(raw, expected)


def test_spectral_bandpass_windows_return_wavelength_channel_arrays():
    erf4 = spectral_upsampling_module.eval_erf4_spectral_bandpass(
        np.array([415.0, 12.0, 667.0, 76.0], dtype=np.float64)
    )
    logiflex8 = spectral_upsampling_module.eval_logiflex8_spectral_bandpass(
        np.array([415.0, 12.0, 667.0, 76.0, 430.0, 650.0, 1.0, 1.0], dtype=np.float64)
    )

    assert erf4.shape == (81, 3)
    assert logiflex8.shape == (81, 3)
    np.testing.assert_allclose(erf4[:, 0], erf4[:, 1])
    np.testing.assert_allclose(erf4[:, 1], erf4[:, 2])


def test_compute_hanatos2025_tc_lut_normalizes_window_to_preserve_midgray(monkeypatch):
    lut = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float64,
    )
    sensitivity = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    window = np.array(
        [
            [0.5, 0.25, 0.75],
            [0.8, 0.6, 0.4],
        ],
        dtype=np.float64,
    )
    illuminant = np.array([2.0, 4.0], dtype=np.float64)

    adaptation = Hanatos2025SensitivityAdaptation(
        window_params=np.array([415.0, 12.0, 667.0, 76.0], dtype=np.float64),
        reference_illuminant='D55',
        apply_window=True,
        apply_surface=False,
    )

    monkeypatch.setattr(spectral_upsampling_module, 'HANATOS2025_SPECTRA_LUT', lut)
    monkeypatch.setattr(spectral_upsampling_module, 'eval_spectral_bandpass_window', lambda _params: window)
    monkeypatch.setattr(spectral_upsampling_module, 'standard_illuminant', lambda _label: illuminant)

    raw_lut = spectral_upsampling_module.compute_hanatos2025_tc_lut(sensitivity, adaptation)

    normalization = np.sum(sensitivity * illuminant[:, None] * window, axis=0) / np.sum(sensitivity * illuminant[:, None], axis=0)
    expected = np.einsum('ijl,lm->ijm', lut, sensitivity * (window / normalization))
    np.testing.assert_allclose(raw_lut, expected)


def test_lut_registry_contains_arctic2026beta04_and_hanatos2025():
    available = spectral_upsampling_module.available_lut_identifiers()
    assert 'arctic2026beta04' in available
    assert 'hanatos2025' in available

    desc = spectral_upsampling_module.lut_descriptor('arctic2026beta04')
    assert desc['kind'] == 'reflectance'
    assert desc['identifier'] == 'arctic2026beta04'
    assert desc['reflectance']['scene_illuminant'] == 'D65'

    spectra = spectral_upsampling_module.get_lut_spectra('arctic2026beta04')
    assert spectra.shape == (192, 192, 81)


def test_compute_reflectance_tc_lut_arctic2026beta04():
    sensitivity = np.ones((81, 3), dtype=np.float64)
    tc_lut = spectral_upsampling_module.compute_tc_lut('arctic2026beta04', sensitivity, 'D55')
    assert tc_lut.shape == (192, 192, 3)
    assert np.all(np.isfinite(tc_lut))
    assert np.all(tc_lut >= 0)


def test_rgb_to_raw_arctic2026beta04_preserves_neutral_midgray():
    sensitivity = np.ones((81, 3), dtype=np.float64)
    tc_lut = spectral_upsampling_module.compute_tc_lut('arctic2026beta04', sensitivity, 'D55')
    midgray_rgb = np.full((1, 1, 3), 0.184, dtype=np.float64)

    raw = spectral_upsampling_module.rgb_to_raw(
        'arctic2026beta04',
        midgray_rgb,
        sensitivity,
        color_space='sRGB',
        apply_cctf_decoding=False,
        reference_illuminant='D55',
        tc_lut=tc_lut,
    )
    assert raw.shape == (1, 1, 3)
    # Neutral balance: all 3 channels should be identical
    np.testing.assert_allclose(raw[0, 0, 0], raw[0, 0, 1], rtol=1e-5)
    np.testing.assert_allclose(raw[0, 0, 1], raw[0, 0, 2], rtol=1e-5)


def test_simulation_pipeline_with_arctic2026beta04():
    from spektrafilm.runtime.params_schema import RuntimePhotoParams
    from spektrafilm.profiles.io import load_profile
    from spektrafilm.runtime.pipeline import SimulationPipeline

    film = load_profile('kodak_portra_400')
    paper = load_profile('kodak_endura_premier')
    params = RuntimePhotoParams(film=film, print=paper)
    params.settings.preview_mode = True
    params.settings.rgb_to_raw_method = 'arctic2026beta04'

    pipeline = SimulationPipeline(params)
    image = np.full((16, 16, 3), 0.184, dtype=np.float32)
    result = pipeline.process(image)
    assert result.shape == (16, 16, 3)
    assert np.all(np.isfinite(result))
