from __future__ import annotations

import numpy as np
import pytest
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.gallery_filmstrip import GalleryFilmstripWidget
from spektrafilm_gui.gallery_model import GalleryModel
from spektrafilm_gui.histogram_widget import (
    ALL_MODES,
    ALL_SCOPES,
    MODE_LUMA,
    MODE_RGB,
    SCOPE_HISTOGRAM,
    SCOPE_PARADE,
    HistogramCanvas,
    HistogramWidget,
    compute_channel_histogram,
    compute_parade_image,
)

_q_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_compute_channel_histogram():
    data = np.array([0, 0, 10, 50, 100, 255, 255], dtype=np.uint8)
    counts, shadow_clip, highlight_clip = compute_channel_histogram(data, num_bins=256)
    assert counts[0] == 2.0
    assert counts[255] == 2.0
    assert counts[10] == 1.0
    assert shadow_clip == pytest.approx(2.0 / 7.0)
    assert highlight_clip == pytest.approx(2.0 / 7.0)

    # Empty array
    empty_counts, s_c, h_c = compute_channel_histogram(np.array([], dtype=np.uint8))
    assert np.all(empty_counts == 0.0)
    assert s_c == 0.0 and h_c == 0.0


def test_compute_parade_image():
    # 1. Float image in [0.0, 1.0]
    img_f = np.random.rand(80, 120, 3).astype(np.float32)
    qimg = compute_parade_image(img_f, target_w=240, target_h=90, gap=3)
    assert qimg is not None
    assert isinstance(qimg, QtGui.QImage)
    assert qimg.height() == 90
    assert qimg.width() > 100
    assert qimg.format() == QtGui.QImage.Format_RGBA8888

    # 2. Uint8 image in [0, 255]
    img_u8 = (img_f * 255).astype(np.uint8)
    qimg_u8 = compute_parade_image(img_u8, target_w=200, target_h=80)
    assert qimg_u8 is not None
    assert qimg_u8.height() == 80

    # 3. Grayscale (2D) image
    gray = np.random.rand(60, 60).astype(np.float32)
    qimg_gray = compute_parade_image(gray, target_w=180, target_h=70)
    assert qimg_gray is not None

    # 4. None / empty image
    assert compute_parade_image(None, 200, 80) is None
    assert compute_parade_image(np.array([]), 200, 80) is None
    assert compute_parade_image(img_f, 10, 5) is None


def test_histogram_canvas_modes_and_scopes():
    canvas = HistogramCanvas()

    assert canvas.scope_type == SCOPE_HISTOGRAM
    assert canvas.mode == MODE_RGB

    # Toggle scope
    next_scope = canvas.toggle_scope_type()
    assert next_scope == SCOPE_PARADE
    assert canvas.scope_type == SCOPE_PARADE

    # Toggle back
    canvas.toggle_scope_type()
    assert canvas.scope_type == SCOPE_HISTOGRAM

    # Cycle histogram channel mode
    assert canvas.cycle_mode() == MODE_LUMA
    assert canvas.mode == MODE_LUMA


def test_histogram_canvas_painting_and_rendering():
    canvas = HistogramCanvas()
    canvas.resize(250, 104)

    # 1. Paint empty canvas (no image)
    pix = QtGui.QPixmap(canvas.size())
    canvas.render(pix)
    assert not pix.isNull()

    # 2. Paint with image in Histogram mode
    test_img = np.zeros((100, 100, 3), dtype=np.float32)
    test_img[:50, :, 0] = 0.8  # Red
    test_img[:, :50, 1] = 0.5  # Green
    test_img[:, :, 2] = 0.3    # Blue
    canvas.set_image(test_img)

    assert canvas._has_data is True
    assert canvas._norm_r is not None
    assert canvas._norm_g is not None
    assert canvas._norm_b is not None

    canvas.render(pix)
    assert not pix.isNull()

    # 3. Switch to Parade mode and paint
    canvas.scope_type = SCOPE_PARADE
    canvas.render(pix)
    assert not pix.isNull()
    assert canvas._parade_image is not None

    # 4. Resize canvas and verify parade updates
    canvas.resize(350, 104)
    canvas.render(pix)
    assert not pix.isNull()

    # 5. Clear image
    canvas.set_image(None)
    assert canvas._has_data is False


def test_histogram_widget_ui_integration():
    widget = HistogramWidget()

    assert widget.scope_type == SCOPE_HISTOGRAM
    assert not widget.channels_container.isHidden()
    assert widget.parade_legend_label.isHidden()

    # Click PARADE button
    widget.scope_parade_btn.click()
    assert widget.scope_type == SCOPE_PARADE
    assert widget.channels_container.isHidden()
    assert not widget.parade_legend_label.isHidden()

    # Click HIST button
    widget.scope_hist_btn.click()
    assert widget.scope_type == SCOPE_HISTOGRAM
    assert not widget.channels_container.isHidden()
    assert widget.parade_legend_label.isHidden()

    # Set image and verify it propagates to canvas
    img = np.ones((40, 40, 3), dtype=np.uint8) * 100
    widget.set_image(img)
    assert widget.canvas._has_data is True

    # Switch to Parade with image present
    widget.set_scope_type(SCOPE_PARADE)
    assert widget.canvas.scope_type == SCOPE_PARADE

    # Clear
    widget.clear()
    assert widget.canvas._has_data is False


def test_left_panel_resizability():
    model = GalleryModel()
    strip = GalleryFilmstripWidget(model)

    # Verify panel is resizable (not locked with setFixedWidth)
    assert strip.minimumWidth() <= 200
    assert strip.maximumWidth() >= 600
    assert strip.minimumWidth() != strip.maximumWidth()

    # Resize panel to wider dimension and verify layout adapts
    strip.resize(380, 700)
    assert strip.width() == 380
    assert strip.histogram_widget.width() > 0


def test_scope_section_height_resizability():
    model = GalleryModel()
    strip = GalleryFilmstripWidget(model)
    strip.resize(250, 800)
    strip.show()

    # 1. Verify existence and vertical orientation of v_splitter
    assert hasattr(strip, "v_splitter")
    assert isinstance(strip.v_splitter, QtWidgets.QSplitter)
    assert strip.v_splitter.orientation() == QtCore.Qt.Vertical

    # 2. Verify widgets in v_splitter
    assert strip.v_splitter.count() == 2
    assert strip.v_splitter.widget(0) is strip.histogram_widget
    assert strip.v_splitter.widget(1) is strip.filmstrip_content

    # 3. Test programmatic resizing of scope section by height
    strip.v_splitter.setSizes([320, 480])
    QtWidgets.QApplication.processEvents()

    assert strip.histogram_widget.height() >= 280
    assert strip.histogram_widget.canvas.height() > 200

    # 4. Check Parade mode rendering at expanded height
    strip.histogram_widget.set_scope_type(SCOPE_PARADE)
    test_img = np.ones((50, 50, 3), dtype=np.uint8) * 150
    strip.set_histogram_image(test_img)

    pix = QtGui.QPixmap(strip.histogram_widget.canvas.size())
    strip.histogram_widget.canvas.render(pix)
    assert strip.histogram_widget.canvas._parade_image is not None
    assert strip.histogram_widget.canvas._parade_image.height() > 200

    # 5. Shrink scope height to compact mode
    strip.v_splitter.setSizes([80, 720])
    QtWidgets.QApplication.processEvents()

    assert strip.histogram_widget.height() <= 120
    pix_small = QtGui.QPixmap(strip.histogram_widget.canvas.size())
    strip.histogram_widget.canvas.render(pix_small)
    assert strip.histogram_widget.canvas._parade_image is not None
    assert strip.histogram_widget.canvas._parade_image.height() < 120
