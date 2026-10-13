import numpy as np
import pytest

from spektrafilm.runtime.process import simulate
from tests.conftest import make_fast_test_params


def test_precision_pipeline_float32():
    image = np.full((16, 16, 3), 0.18, dtype=np.float32)
    params = make_fast_test_params()
    params.settings.precision = "float32"
    result = simulate(image, params)
    assert result.dtype == np.float32
    assert result.shape == (16, 16, 3)
    assert np.all(np.isfinite(result))


def test_precision_pipeline_float64():
    image = np.full((16, 16, 3), 0.18, dtype=np.float64)
    params = make_fast_test_params()
    params.settings.precision = "float64"
    result = simulate(image, params)
    assert result.dtype == np.float64
    assert result.shape == (16, 16, 3)
    assert np.all(np.isfinite(result))


def test_precision_numerical_consistency():
    image = np.linspace(0.01, 1.0, 16 * 16 * 3, dtype=np.float64).reshape((16, 16, 3))
    
    params_f32 = make_fast_test_params()
    params_f32.settings.precision = "float32"
    res_f32 = simulate(image, params_f32)
    
    params_f64 = make_fast_test_params()
    params_f64.settings.precision = "float64"
    res_f64 = simulate(image, params_f64)
    
    np.testing.assert_allclose(res_f32, res_f64, rtol=1e-3, atol=1e-4)


def test_precision_cuda():
    from spektrafilm.cuda.device import is_cuda_available
    if not is_cuda_available():
        pytest.skip("CUDA device not available")

    image = np.full((16, 16, 3), 0.18, dtype=np.float32)
    params = make_fast_test_params()
    params.settings.device = "cuda"
    params.settings.precision = "float32"
    result_f32 = simulate(image, params)
    assert result_f32.dtype == np.float32

    params.settings.precision = "float64"
    image_f64 = np.full((16, 16, 3), 0.18, dtype=np.float64)
    result_f64 = simulate(image_f64, params)
    assert result_f64.dtype == np.float64


def test_precision_gui_mapping():
    from spektrafilm_gui.params_mapper import build_params_from_state
    from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state

    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.input_image.settings.precision = "float64"
    params = build_params_from_state(state)
    assert params.settings.precision == "float64"

    state.input_image.settings.precision = "float32"
    params = build_params_from_state(state)
    assert params.settings.precision == "float32"

