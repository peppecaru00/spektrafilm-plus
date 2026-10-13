"""Unit tests for ColorReferenceService and scanner white/black correction."""
from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.runtime.params_builder import init_params, digest_params
from spektrafilm.runtime.api import Simulator


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "white_correction,black_correction",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_scanner_white_black_correction_print_paper(device: str, white_correction: bool, black_correction: bool):
    """Verify that scanner white/black correction executes cleanly on print paper across devices."""
    if device == "cuda":
        try:
            from spektrafilm.cuda.device import is_cuda_available
            if not is_cuda_available():
                pytest.skip("CUDA device not available")
        except ImportError:
            pytest.skip("CuPy not installed")

    params = init_params(film_profile="kodak_portra_400", print_profile="kodak_portra_endura")
    params.scanner.white_correction = white_correction
    params.scanner.black_correction = black_correction
    params.settings.device = device

    sim = Simulator(digest_params(params))
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    out = sim.process(img)

    assert out.shape == (32, 32, 3)
    assert np.all(np.isfinite(out))
    assert 0.0 <= np.mean(out) <= 1.0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "white_correction,black_correction",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_scanner_white_black_correction_slide_film(device: str, white_correction: bool, black_correction: bool):
    """Verify that scanner white/black correction executes cleanly on positive slide film across devices."""
    if device == "cuda":
        try:
            from spektrafilm.cuda.device import is_cuda_available
            if not is_cuda_available():
                pytest.skip("CUDA device not available")
        except ImportError:
            pytest.skip("CuPy not installed")

    params = init_params(film_profile="fujifilm_velvia_100")
    params.io.scan_film = True
    params.scanner.white_correction = white_correction
    params.scanner.black_correction = black_correction
    params.settings.device = device

    sim = Simulator(digest_params(params))
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    out = sim.process(img)

    assert out.shape == (32, 32, 3)
    assert np.all(np.isfinite(out))
    assert 0.0 <= np.mean(out) <= 1.0
