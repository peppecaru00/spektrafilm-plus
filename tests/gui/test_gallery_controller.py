from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.app import connect_controller_signals, initialize_controller
from spektrafilm_gui.controller import GuiController
from spektrafilm_gui.gallery_filmstrip import GalleryFilmstripWidget
from spektrafilm_gui.gallery_model import GalleryItem, GalleryModel
from spektrafilm_gui.gallery_sidecar import has_sidecar, load_sidecar, save_sidecar, sidecar_path
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state
from spektrafilm_gui.state_bridge import apply_gui_state
from spektrafilm_gui.widgets import create_widget_bundle

from .helpers import FakeViewer

_q_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gallery_filmstrip_widget_creation():
    model = GalleryModel()
    strip = GalleryFilmstripWidget(model)

    assert strip.counter_label.text() == "No photos"
    assert not strip.prev_btn.isEnabled()
    assert not strip.next_btn.isEnabled()
    assert not strip.scan_all_btn.isEnabled()


def test_controller_gallery_switch_photo_auto_saves_and_inherits(tmp_path: Path, monkeypatch):
    p1 = tmp_path / "img_01.jpg"
    p2 = tmp_path / "img_02.jpg"
    qimg = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(100, 100, 100))
    qimg.save(str(p1))
    qimg.save(str(p2))

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    dummy_img = np.full((10, 10, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", lambda path: dummy_img)

    model = controller._gallery_model
    assert model is not None
    model.load_folder(tmp_path)
    assert model.count == 2
    assert model.current_index == 0

    # Ensure p1 was loaded and an initial sidecar created
    assert has_sidecar(p1)

    # Change a setting on photo 1 (e.g. exposure compensation)
    widgets.camera.exposure_compensation_ev.setValue(1.75)
    controller.request_auto_preview()

    # Advance to photo 2 (which currently has NO sidecar)
    assert not has_sidecar(p2)
    model.next_photo()
    assert model.current_index == 1

    # Photo 1's pending auto-save should have flushed before leaving!
    p1_state = load_sidecar(p1)
    assert p1_state is not None
    assert p1_state.camera.exposure_compensation_ev == 1.75

    # Photo 2 should have inherited the active look and saved its own sidecar!
    assert has_sidecar(p2)
    p2_state = load_sidecar(p2)
    assert p2_state is not None
    assert p2_state.camera.exposure_compensation_ev == 1.75


def test_controller_gallery_copy_paste_look(tmp_path: Path, monkeypatch):
    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    widgets.camera.exposure_compensation_ev.setValue(2.25)
    controller.gallery_copy_look()

    # Reset widget value
    widgets.camera.exposure_compensation_ev.setValue(0.0)
    assert widgets.camera.exposure_compensation_ev.value == 0.0

    # Paste look
    controller.gallery_paste_look()
    assert widgets.camera.exposure_compensation_ev.value == 2.25


def test_controller_flush_auto_save_sidecar(tmp_path: Path, monkeypatch):
    p = tmp_path / "sample.jpg"
    qimg = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(100, 100, 100))
    qimg.save(str(p))

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    dummy_img = np.full((10, 10, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", lambda path: dummy_img)

    controller.load_input_image(str(p))
    widgets.camera.exposure_compensation_ev.setValue(-1.5)

    controller._flush_auto_save_sidecar()

    saved = load_sidecar(p)
    assert saved is not None
    assert saved.camera.exposure_compensation_ev == -1.5


def test_controller_gallery_remove_photo(tmp_path: Path, monkeypatch):
    p1 = tmp_path / "img_01.jpg"
    p2 = tmp_path / "img_02.jpg"
    qimg = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(100, 100, 100))
    qimg.save(str(p1))
    qimg.save(str(p2))

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    dummy_img = np.full((10, 10, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", lambda path: dummy_img)

    model = controller._gallery_model
    model.load_folder(tmp_path)
    assert model.count == 2
    assert model.current_index == 0
    assert controller._current_input_path == str(p1)

    # Remove photo 1 via controller
    controller.gallery_remove_current_photo()

    # Photo 1 is removed from strip, but still exists on disk!
    assert model.count == 1
    assert model.current_index == 0
    assert model.current_item.filename == "img_02.jpg"
    assert p1.exists()
    assert p2.exists()

    # Photo 2 is now loaded in controller
    assert controller._current_input_path == str(p2)

    # Remove photo 2 -> strip is empty
    controller.gallery_remove_current_photo()
    assert model.count == 0
    assert model.current_index == -1
    assert controller._current_input_path is None
    assert p2.exists()


def test_horizontal_scroll_area_wheel_event():
    from spektrafilm_gui.gallery_filmstrip import HorizontalScrollArea
    area = HorizontalScrollArea()
    content = QtWidgets.QWidget()
    content.setFixedSize(1000, 100)
    area.setWidget(content)
    area.resize(200, 100)
    area.show()

    h_bar = area.horizontalScrollBar()
    initial_val = h_bar.value()

    # Simulate vertical wheel scroll down (delta = -120)
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(50, 50), QtCore.QPointF(50, 50),
        QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.ScrollUpdate, False
    )
    area.wheelEvent(wheel)

    # Should have scrolled horizontally!
    assert h_bar.value() > initial_val


def test_controller_gallery_remove_multiple_selected(tmp_path: Path, monkeypatch):
    paths = [tmp_path / f"img_{i:02d}.jpg" for i in range(4)]
    for p in paths:
        qimg = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
        qimg.fill(QtGui.QColor(100, 100, 100))
        qimg.save(str(p))

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    dummy_img = np.full((10, 10, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", lambda path: dummy_img)

    model = controller._gallery_model
    model.load_folder(tmp_path)
    assert model.count == 4

    # Select photos at indices 1 and 2
    model.set_selected(1, True)
    model.set_selected(2, True)
    assert model.selected_indices == {1, 2}

    # Remove via controller (which represents clicking 🗑️ or pressing Del)
    controller.gallery_remove_current_photo()

    # Photos 1 and 2 should be removed from strip; photos 0 and 3 should remain
    assert model.count == 2
    assert [it.filename for it in model.items] == ["img_00.jpg", "img_03.jpg"]
    assert model.selected_count == 0

    # Ensure all files still exist on disk
    for p in paths:
        assert p.exists()


def test_controller_gallery_paste_look_to_multiple_selected(tmp_path: Path, monkeypatch):
    paths = [tmp_path / f"img_{i:02d}.jpg" for i in range(3)]
    for p in paths:
        qimg = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
        qimg.fill(QtGui.QColor(100, 100, 100))
        qimg.save(str(p))

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    dummy_img = np.full((10, 10, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", lambda path: dummy_img)

    model = controller._gallery_model
    model.load_folder(tmp_path)
    assert model.count == 3
    assert model.current_index == 0

    # Set exposure on photo 0 and copy look
    widgets.camera.exposure_compensation_ev.setValue(3.5)
    controller.gallery_copy_look()

    # Reset widget exposure back to 0.0
    widgets.camera.exposure_compensation_ev.setValue(0.0)

    # Multi-select photos 1 and 2
    model.set_selected(1, True)
    model.set_selected(2, True)

    # Paste look
    controller.gallery_paste_look()

    # Sidecars for photo 1 and photo 2 should be created with exposure 3.5
    s1 = load_sidecar(paths[1])
    s2 = load_sidecar(paths[2])
    assert s1 is not None
    assert s1.camera.exposure_compensation_ev == 3.5
    assert s2 is not None
    assert s2.camera.exposure_compensation_ev == 3.5


