from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.app import initialize_controller
from spektrafilm_gui.controller import GuiController
from spektrafilm_gui.gallery_filmstrip import GalleryFilmstripWidget
from spektrafilm_gui.gallery_model import GalleryModel
from spektrafilm_gui.gallery_params import all_param_defs, get_param_def
from spektrafilm_gui.gallery_sidecar import has_sidecar, load_sidecar, save_sidecar
from spektrafilm_gui.gallery_sync_dialog import SyncParameterDialog
from spektrafilm_gui.persistence import PROJECT_DEFAULT_GUI_STATE
from spektrafilm_gui.state import clone_gui_state
from spektrafilm_gui.state_bridge import apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import create_widget_bundle

from .helpers import FakeViewer

_q_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gallery_sync_dialog_creation_and_filtering():
    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.simulation.selection.film_stock = "Kodak Tri-X 400"
    state.camera.exposure_compensation_ev = 1.67

    dialog = SyncParameterDialog(
        None,
        active_state=state,
        active_filename="test_photo.cr3",
        selected_count=2,
        total_count=5,
    )

    # Check initial values and tree population
    assert dialog.tree.topLevelItemCount() >= 5
    assert dialog.radio_selected.isChecked()
    assert "2 photos" in dialog.radio_selected.text()
    assert "5 total" in dialog.radio_all.text()

    # Select film_stock in tree
    stock_item = None
    for i in range(dialog.tree.topLevelItemCount()):
        cat = dialog.tree.topLevelItem(i)
        for j in range(cat.childCount()):
            child = cat.child(j)
            if child.data(0, QtCore.Qt.UserRole) == "film_stock":
                stock_item = child
                break
        if stock_item is not None:
            break

    assert stock_item is not None
    assert "Tri-X 400" in stock_item.text(1)
    dialog.tree.setCurrentItem(stock_item)
    assert dialog.get_selected_param_id() == "film_stock"
    assert dialog.apply_btn.isEnabled()
    assert dialog.copy_btn.isEnabled()

    # Test search filter
    dialog.search_edit.setText("Grain")
    # Tree should filter items
    assert not stock_item.isHidden() or stock_item.parent().isHidden()

    # Copy button action
    dialog._on_copy_clicked()
    assert dialog.was_copy_only() is True


def test_controller_gallery_paste_single_param_across_selected_photos(tmp_path: Path, monkeypatch):
    p0 = tmp_path / "img_00.jpg"
    p1 = tmp_path / "img_01.jpg"
    p2 = tmp_path / "img_02.jpg"
    for p in (p0, p1, p2):
        qimg = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
        qimg.fill(QtGui.QColor(100, 100, 100))
        qimg.save(str(p))

    # Give photos initial individual settings
    s0 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    s0.simulation.selection.film_stock = "Kodak Portra 400"
    s0.camera.exposure_compensation_ev = 1.5
    save_sidecar(p0, s0)

    s1 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    s1.simulation.selection.film_stock = "Fuji Pro 400H"
    s1.camera.exposure_compensation_ev = -0.75
    save_sidecar(p1, s1)

    s2 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    s2.simulation.selection.film_stock = "Ilford HP5"
    s2.camera.exposure_compensation_ev = 0.0
    save_sidecar(p2, s2)

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    dummy_img = np.full((10, 10, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", lambda path: dummy_img)

    model = controller._gallery_model
    assert model is not None
    model.load_folder(tmp_path)
    assert model.count == 3
    assert model.current_index == 0

    # Multi-select photos 1 and 2
    model.set_selected(1, True)
    model.set_selected(2, True)
    assert model.selected_count == 2
    assert model.selected_indices == {1, 2}

    # Now paste ONLY "film_stock" = "Kodak Portra 400" to selected photos
    updated_count = controller.gallery_paste_single_param(
        param_id="film_stock",
        value="Kodak Portra 400",
        target_indices=[1, 2],
    )
    assert updated_count == 2

    # Verify sidecar for photo 1: film_stock changed, but exposure_compensation_ev preserved (-0.75)!
    reloaded_p1 = load_sidecar(p1)
    assert reloaded_p1 is not None
    assert reloaded_p1.simulation.selection.film_stock == "Kodak Portra 400"
    assert reloaded_p1.camera.exposure_compensation_ev == pytest.approx(-0.75)

    # Verify sidecar for photo 2: film_stock changed, but exposure_compensation_ev preserved (0.0)!
    reloaded_p2 = load_sidecar(p2)
    assert reloaded_p2 is not None
    assert reloaded_p2.simulation.selection.film_stock == "Kodak Portra 400"
    assert reloaded_p2.camera.exposure_compensation_ev == pytest.approx(0.0)

    # Verify photo 0 was not touched
    reloaded_p0 = load_sidecar(p0)
    assert reloaded_p0 is not None
    assert reloaded_p0.simulation.selection.film_stock == "Kodak Portra 400"
    assert reloaded_p0.camera.exposure_compensation_ev == pytest.approx(1.5)


def test_controller_gallery_copy_and_paste_single_param_via_clipboard(tmp_path: Path, monkeypatch):
    p0 = tmp_path / "img_00.jpg"
    p1 = tmp_path / "img_01.jpg"
    for p in (p0, p1):
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
    assert model.count == 2

    # Set exposure EV on photo 0 to +2.5 EV
    widgets.camera.exposure_compensation_ev.setValue(2.5)

    # Copy single param "exposure_compensation"
    controller.gallery_copy_single_param("exposure_compensation")
    assert model.has_param_clipboard
    assert model.param_clipboard_id == "exposure_compensation"
    assert model.param_clipboard_value == pytest.approx(2.5)

    # Select photo 1
    model.set_selected(1, True)

    # Paste copied single param to selected photo 1
    controller.gallery_paste_single_param()

    # Verify photo 1 sidecar has +2.5 EV
    reloaded_p1 = load_sidecar(p1)
    assert reloaded_p1 is not None
    assert reloaded_p1.camera.exposure_compensation_ev == pytest.approx(2.5)


def test_filmstrip_widget_keyboard_shortcuts(tmp_path: Path, monkeypatch):
    # Mock thumbnail service to avoid background file reader locks on Windows
    from spektrafilm_gui.gallery_thumbnails import ThumbnailService
    monkeypatch.setattr(ThumbnailService.instance(), "get_thumbnail", lambda *args, **kwargs: None)

    model = GalleryModel()
    strip = GalleryFilmstripWidget(model)

    # Add 3 dummy items to model in tmp_path
    paths = [tmp_path / f"test_{i}.jpg" for i in range(3)]
    for p in paths:
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    from spektrafilm_gui.gallery_model import GalleryItem
    model._items = [GalleryItem(path=p) for p in paths]
    model._current_index = 0
    model.items_changed.emit()

    assert model.selected_count == 0

    # Simulate Ctrl+A key event on filmstrip
    ctrl_a = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_A,
        QtCore.Qt.ControlModifier,
    )
    strip.keyPressEvent(ctrl_a)
    assert model.selected_count == 3
    assert strip.selection_label.text() == "Sel: 3"

    # Simulate Escape key event on filmstrip
    escape = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_Escape,
        QtCore.Qt.NoModifier,
    )
    strip.keyPressEvent(escape)
    assert model.selected_count == 0
    assert strip.selection_label.text() == "Sel: 0"


def test_sync_dialog_check_all_none_and_category_toggling():
    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.simulation.selection.film_stock = "Kodak Portra 160"
    state.camera.exposure_compensation_ev = 1.25

    dialog = SyncParameterDialog(
        None,
        active_state=state,
        active_filename="test.jpg",
        selected_count=1,
        total_count=3,
    )

    total_params = len(all_param_defs())
    assert len(dialog.get_checked_params_dict()) == total_params
    assert dialog.apply_btn.isEnabled()

    # Uncheck all
    dialog.check_none_btn.click()
    assert len(dialog.get_checked_params_dict()) == 0
    assert not dialog.apply_btn.isEnabled()
    assert not dialog.copy_btn.isEnabled()

    # Find the "Camera & Exposure" category item
    cat_cam = None
    for i in range(dialog.tree.topLevelItemCount()):
        item = dialog.tree.topLevelItem(i)
        if "Camera" in item.text(0):
            cat_cam = item
            break
    assert cat_cam is not None

    # Check the category -> should check all its children
    cat_cam.setCheckState(0, QtCore.Qt.Checked)
    checked_dict = dialog.get_checked_params_dict()
    assert len(checked_dict) == cat_cam.childCount()
    assert "exposure_compensation" in checked_dict
    assert checked_dict["exposure_compensation"] == pytest.approx(1.25)
    assert dialog.apply_btn.isEnabled()

    # Check all
    dialog.check_all_btn.click()
    assert len(dialog.get_checked_params_dict()) == total_params


def test_controller_gallery_paste_multi_params_dict(tmp_path: Path, monkeypatch):
    p0 = tmp_path / "img_00.jpg"
    p1 = tmp_path / "img_01.jpg"
    p2 = tmp_path / "img_02.jpg"
    for p in (p0, p1, p2):
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

    # Multi-select 1 and 2
    model.set_selected(1, True)
    model.set_selected(2, True)

    multi_params = {
        "film_stock": "Kodak Ektar 100",
        "print_exposure": 18.5,
        "temperature": 4500.0,
        "exposure_compensation": -1.5,
    }

    applied = controller.gallery_paste_params_dict(multi_params, target_indices=[1, 2])
    assert applied == 2

    # Check photo 1 sidecar
    s1 = load_sidecar(p1)
    assert s1 is not None
    assert s1.simulation.selection.film_stock == "Kodak Ektar 100"
    assert s1.simulation.enlarger.print_exposure == pytest.approx(18.5)
    assert s1.gui_only.load_raw.temperature == pytest.approx(4500.0)
    assert s1.camera.exposure_compensation_ev == pytest.approx(-1.5)

    # Check photo 2 sidecar
    s2 = load_sidecar(p2)
    assert s2 is not None
    assert s2.simulation.selection.film_stock == "Kodak Ektar 100"
    assert s2.simulation.enlarger.print_exposure == pytest.approx(18.5)


def test_controller_gallery_raw_invalidation_on_paste(tmp_path: Path, monkeypatch):
    raw_path = tmp_path / "photo_01.arw"
    raw_path.write_bytes(b"\x00" * 100)

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    raw_reloads: list[str] = []
    auto_previews: list[bool] = []
    monkeypatch.setattr(controller, "load_raw_image", lambda path: raw_reloads.append(path))
    monkeypatch.setattr(controller, "request_auto_preview", lambda: auto_previews.append(True))

    model = controller._gallery_model
    model.add_files([raw_path])
    model.select_index(0)
    controller._current_input_path = str(raw_path)
    raw_reloads.clear()
    auto_previews.clear()

    # 1. Pasting non-RAW parameter should trigger request_auto_preview, not load_raw_image
    controller.gallery_paste_params_dict({"exposure_compensation": 1.5}, target_indices=[0])
    assert len(raw_reloads) == 0
    assert len(auto_previews) >= 1

    # 2. Pasting RAW parameter (temperature) should trigger load_raw_image
    auto_previews.clear()
    controller.gallery_paste_params_dict({"temperature": 6500.0}, target_indices=[0])
    assert len(raw_reloads) == 1
    assert raw_reloads[0] == str(raw_path)

    # 3. Pasting entire look should trigger load_raw_image on RAW photo
    raw_reloads.clear()
    look_state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    look_state.gui_only.load_raw.tint = 12.0
    model.copy_look(look_state)
    controller.gallery_paste_look()
    assert len(raw_reloads) == 1
    assert raw_reloads[0] == str(raw_path)
