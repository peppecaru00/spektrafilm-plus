from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import spektrafilm.runtime.stages.filming as filming_module


def test_rgb_to_film_raw_passes_linear_sensitivity_to_hanatos2025(monkeypatch) -> None:
    """`_rgb_to_film_raw` should pass `10**log_sensitivity` (linear sensitivity)
    to `rgb_to_raw_hanatos2025`. With no UV/IR filter active, no bandpass
    correction is applied. Hanatos2025 adaptation is handled separately via
    the LUT service's `tc_lut`, not by mutating sensitivity in this method.
    """
    captured: dict[str, np.ndarray] = {}
    log_sensitivity = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], dtype=float)

    def fake_rgb_to_raw_hanatos2025(
        rgb,
        sensitivity,
        *,
        color_space=None,
        apply_cctf_decoding=None,
        reference_illuminant=None,
        tc_lut=None,
    ):
        del color_space, apply_cctf_decoding, reference_illuminant, tc_lut
        captured['sensitivity'] = np.asarray(sensitivity, dtype=float)
        return np.ones(rgb.shape, dtype=float)

    monkeypatch.setattr(filming_module, 'rgb_to_raw_hanatos2025', fake_rgb_to_raw_hanatos2025)

    stage = object.__new__(filming_module.FilmingStage)
    setattr(stage, '_film', SimpleNamespace(
        info=SimpleNamespace(reference_illuminant='D55'),
        data=SimpleNamespace(log_sensitivity=log_sensitivity),
    ))
    setattr(stage, '_camera', SimpleNamespace(filter_uv=(0.0, 0.0, 0.0), filter_ir=(0.0, 0.0, 0.0)))
    setattr(stage, '_settings', SimpleNamespace(rgb_to_raw_method='hanatos2025'))
    setattr(stage, '_lut_service', SimpleNamespace(get_filming_tc_lut=lambda sensitivity: None))

    rgb = np.ones((1, 1, 3), dtype=float)

    getattr(stage, '_rgb_to_film_raw')(rgb)

    np.testing.assert_allclose(captured['sensitivity'], 10.0 ** log_sensitivity)


def test_positive_film_scan_produces_non_black_image() -> None:
    from spektrafilm.profiles.io import load_profile
    from spektrafilm.runtime.params_schema import RuntimePhotoParams
    from spektrafilm.runtime.pipeline import SimulationPipeline

    film = load_profile('kodak_ektachrome_100')
    paper = load_profile('kodak_endura_premier')
    params = RuntimePhotoParams(film=film, print=paper)
    params.io.scan_film = True
    params.settings.preview_mode = True

    pipeline = SimulationPipeline(params)
    img = np.full((32, 32, 3), 0.184, dtype=np.float64)
    out = pipeline.process(img)

    assert out.shape == (32, 32, 3)
    assert np.mean(out) > 0.05, f"Expected non-black image for Ektachrome 100 scan, got mean {np.mean(out)}"


def test_scan_film_responds_to_chemistry_gamma_morph() -> None:
    from dataclasses import replace
    from spektrafilm.profiles.io import load_profile
    from spektrafilm.runtime.params_schema import RuntimePhotoParams
    from spektrafilm.runtime.pipeline import SimulationPipeline

    film = load_profile('kodak_ektachrome_100')
    paper = load_profile('kodak_endura_premier')
    params = RuntimePhotoParams(film=film, print=paper)
    params.io.scan_film = True
    params.settings.preview_mode = True

    img = np.full((32, 32, 3), 0.184, dtype=np.float64)

    params.print_render.density_curves_morph = replace(
        params.print_render.density_curves_morph, active=True, gamma_factor=0.7
    )
    p_low = SimulationPipeline(params)
    out_low = p_low.process(img)

    params.print_render.density_curves_morph = replace(
        params.print_render.density_curves_morph, active=True, gamma_factor=1.3
    )
    p_high = SimulationPipeline(params)
    out_high = p_high.process(img)

    assert np.max(np.abs(out_low - out_high)) > 0.1, "Chemistry gamma should significantly alter scanned positive film"
