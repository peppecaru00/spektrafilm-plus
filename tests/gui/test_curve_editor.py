from __future__ import annotations

import os
import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from spektrafilm_gui.curve_editor import CurveCanvas, YRGBCurveEditor, pchip_interpolate
from spektrafilm_gui.param_manifest import CHEMISTRY_MANIFEST
from spektrafilm_gui.widget_sections import ChemistrySection
from spektrafilm.utils.morph_curves import PrintCurvesMorphParams


@pytest.fixture(autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_pchip_interpolate_monotonicity():
    x_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_pts = np.array([0.0, 0.15, 0.5, 0.85, 1.0])
    x_eval = np.linspace(0.0, 1.0, 100)
    y_eval = pchip_interpolate(x_pts, y_pts, x_eval)

    # Values at knots match exactly
    for x, y in zip(x_pts, y_pts):
        idx = np.argmin(np.abs(x_eval - x))
        assert np.isclose(y_eval[idx], y, atol=0.01)

    # Monotonically non-decreasing
    assert np.all(np.diff(y_eval) >= -1e-6)


def test_curve_canvas_point_manipulation():
    canvas = CurveCanvas()
    canvas.resize(200, 200)

    # Default points are (0,0) and (1,1)
    pts_y = canvas.get_points("Y")
    assert len(pts_y) == 2
    assert pts_y[0] == (0.0, 0.0)
    assert pts_y[1] == (1.0, 1.0)

    # Add point via setting points
    canvas.set_points("Y", [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)])
    assert len(canvas.get_points("Y")) == 3
    assert canvas.get_points("Y")[1] == (0.5, 0.6)

    # Evaluate curve
    y_eval = canvas.evaluate_curve("Y", np.array([0.5]))
    assert np.isclose(y_eval[0], 0.6, atol=1e-3)

    # Channel switching
    assert canvas.active_channel == "Y"
    canvas.active_channel = "R"
    assert canvas.active_channel == "R"

    # Reset channel
    canvas.set_points("R", [(0.0, 0.0), (0.3, 0.7), (1.0, 1.0)])
    canvas.reset_channel("R")
    assert len(canvas.get_points("R")) == 2

    # Reset all
    canvas.set_points("G", [(0.0, 0.0), (0.4, 0.8), (1.0, 1.0)])
    canvas.reset_all()
    assert len(canvas.get_points("G")) == 2


def test_yrgb_curve_editor_chemistry_bidirectional_sync():
    editor = YRGBCurveEditor()
    editor.resize(240, 220)

    # 1. Update from chemistry parameters
    params = PrintCurvesMorphParams(
        gamma_factor=1.3,
        gamma_factor_red=1.2,
        gamma_factor_green=0.9,
        gamma_factor_blue=1.0,
        gamma_factor_fast=1.1,
        gamma_factor_slow=0.95,
    )
    editor.set_chemistry_params(params)

    # Extract updates from curves
    updates = editor.get_chemistry_updates()
    assert np.isclose(updates["gamma_factor"], 1.3, atol=0.05)
    assert np.isclose(updates["gamma_factor_red"], 1.2, atol=0.05)
    assert np.isclose(updates["gamma_factor_green"], 0.9, atol=0.05)
    assert np.isclose(updates["gamma_factor_blue"], 1.0, atol=0.05)

    # 2. Reset restores defaults
    editor.canvas.reset_all()
    res = editor.get_chemistry_updates()
    assert np.isclose(res["gamma_factor"], 1.0, atol=0.05)
    assert np.isclose(res["gamma_factor_red"], 1.0, atol=0.05)
    assert np.isclose(res["gamma_factor_green"], 1.0, atol=0.05)
    assert np.isclose(res["gamma_factor_blue"], 1.0, atol=0.05)


def test_chemistry_section_integration():
    section = ChemistrySection(CHEMISTRY_MANIFEST)
    assert section.curve_editor is not None
    assert hasattr(section, "gamma_factor")
    assert hasattr(section, "gamma_factor_red")

    # Modifying slider updates curve editor
    section.gamma_factor.setValue(1.4)
    updates = section.curve_editor.get_chemistry_updates()
    assert np.isclose(updates["gamma_factor"], 1.4, atol=0.05)

    # Modifying curve editor updates slider
    section.curve_editor.canvas.set_points("Y", [(0.0, 0.0), (0.5, 0.35), (1.0, 1.0)])
    section.curve_editor._on_canvas_curves_updated()
    assert section.gamma_factor.value > 1.2

    # set_state updates curve editor
    new_params = PrintCurvesMorphParams(
        gamma_factor=0.8,
        gamma_factor_red=1.1,
        gamma_factor_green=1.0,
        gamma_factor_blue=0.9,
    )
    section.set_state(new_params)
    assert np.isclose(section.gamma_factor.value, 0.8, atol=0.01)
    updates = section.curve_editor.get_chemistry_updates()
    assert np.isclose(updates["gamma_factor"], 0.8, atol=0.05)


def test_yrgb_curve_editor_per_channel_fast_slow():
    editor = YRGBCurveEditor()
    editor.resize(240, 220)

    # 1. Update from chemistry params with RGB fast/slow
    params = PrintCurvesMorphParams(
        gamma_factor=1.0,
        gamma_factor_red=1.1,
        gamma_factor_red_fast=1.2,
        gamma_factor_red_slow=0.85,
        gamma_factor_green=1.0,
        gamma_factor_green_fast=0.9,
        gamma_factor_green_slow=1.1,
        gamma_factor_blue=1.0,
    )
    editor.set_chemistry_params(params)

    # Red should now have 5 points because fast/slow != 1.0
    pts_r = editor.canvas.get_points("R")
    assert len(pts_r) == 5

    # Green also has 5 points
    pts_g = editor.canvas.get_points("G")
    assert len(pts_g) == 5

    # Blue has default (flat) fast/slow, so 2 points
    pts_b = editor.canvas.get_points("B")
    assert len(pts_b) == 2

    # Extract updates from curves
    updates = editor.get_chemistry_updates()
    assert np.isclose(updates["gamma_factor_red"], 1.1, atol=0.05)
    assert np.isclose(updates["gamma_factor_red_fast"], 1.2, atol=0.05)
    assert np.isclose(updates["gamma_factor_red_slow"], 0.85, atol=0.05)
    assert np.isclose(updates["gamma_factor_green_fast"], 0.9, atol=0.05)
    assert np.isclose(updates["gamma_factor_green_slow"], 1.1, atol=0.05)
    assert np.isclose(updates["gamma_factor_blue_fast"], 1.0, atol=0.05)
    assert np.isclose(updates["gamma_factor_blue_slow"], 1.0, atol=0.05)


def test_chemistry_section_per_channel_fast_slow_integration():
    section = ChemistrySection(CHEMISTRY_MANIFEST)
    assert hasattr(section, "gamma_factor_red_fast")
    assert hasattr(section, "gamma_factor_red_slow")
    assert hasattr(section, "gamma_factor_green_fast")
    assert hasattr(section, "gamma_factor_green_slow")
    assert hasattr(section, "gamma_factor_blue_fast")
    assert hasattr(section, "gamma_factor_blue_slow")

    # Modifying slider updates curve editor
    section.gamma_factor_red_fast.setValue(1.3)
    updates = section.curve_editor.get_chemistry_updates()
    assert np.isclose(updates["gamma_factor_red_fast"], 1.3, atol=0.05)


def test_curve_canvas_live_background_histogram():
    from qtpy import QtGui

    canvas = CurveCanvas()
    canvas.resize(240, 200)

    assert canvas.show_histogram is True
    assert canvas._has_histogram is False

    # 1. Provide an RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # Red
    img[:, :, 1] = 120  # Green
    img[:, :, 2] = 200  # Blue

    canvas.set_histogram_image(img)
    assert canvas._has_histogram is True
    assert canvas._norm_r is not None
    assert canvas._norm_g is not None
    assert canvas._norm_b is not None
    assert canvas._norm_y is not None
    assert len(canvas._norm_r) == 256
    assert np.isclose(canvas._norm_r[50], 1.0, atol=1e-3)
    assert np.isclose(canvas._norm_g[120], 1.0, atol=1e-3)
    assert np.isclose(canvas._norm_b[200], 1.0, atol=1e-3)

    # 2. Render canvas to pixmap across all channels to verify painter paths
    pix = QtGui.QPixmap(canvas.size())
    for ch in ("Y", "R", "G", "B"):
        canvas.active_channel = ch
        canvas.render(pix)
        assert not pix.isNull()

    # 3. Toggle histogram visibility
    canvas.show_histogram = False
    assert canvas.show_histogram is False
    canvas.render(pix)

    canvas.show_histogram = True
    assert canvas.show_histogram is True

    # 4. Clear histogram
    canvas.clear_histogram()
    assert canvas._has_histogram is False
    assert canvas._norm_r is None


def test_curve_editor_histogram_integration():
    editor = YRGBCurveEditor()
    editor.resize(250, 220)

    assert hasattr(editor, "hist_btn")
    assert editor.hist_btn.isChecked() is True
    assert editor.canvas.show_histogram is True

    # Toggle HIST button
    editor.hist_btn.click()
    assert editor.hist_btn.isChecked() is False
    assert editor.canvas.show_histogram is False

    editor.hist_btn.click()
    assert editor.hist_btn.isChecked() is True
    assert editor.canvas.show_histogram is True

    # Forward image via editor
    test_img = np.ones((60, 60, 3), dtype=np.float32) * 0.5
    editor.set_histogram_image(test_img)
    assert editor.canvas._has_histogram is True
    assert editor.canvas._norm_y is not None

    # Forward image via ChemistrySection
    section = ChemistrySection(CHEMISTRY_MANIFEST)
    section.set_histogram_image(test_img)
    assert section.curve_editor.canvas._has_histogram is True
