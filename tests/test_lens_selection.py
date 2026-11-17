from __future__ import annotations

import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from spektrafilm.utils import raw_file_processor
from spektrafilm_gui.state import clone_gui_state, PROJECT_DEFAULT_GUI_STATE
from spektrafilm_gui.widget_editors import LensSelectorWidget
from spektrafilm_gui.widget_sections import LoadRawSection
from spektrafilm_gui.gallery_params import get_param_def, RAW_AFFECTING_PARAM_IDS


def _ensure_qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_get_available_lenses():
    lenses = raw_file_processor.get_available_lenses()
    assert isinstance(lenses, list)
    assert len(lenses) > 500
    # Verify formatting
    assert any("Canon" in l for l in lenses)
    assert any("Nikon" in l for l in lenses)
    assert any("Sony" in l for l in lenses)
    for l in lenses[:20]:
        assert " - " in l
    # Verify sorting
    assert lenses == sorted(lenses)


def test_detect_raw_lens_mocked(monkeypatch):
    exif = raw_file_processor.ExifData(
        make="Canon",
        model="Canon EOS 5D Mark IV",
        lens_make="Canon",
        lens_model="Canon EF 50mm f/1.8 STM",
        focal_length=50.0,
        f_number=2.8,
    )
    monkeypatch.setattr(raw_file_processor, "_read_exif_metadata", lambda p: exif)

    canonical, summary = raw_file_processor.detect_raw_lens("fake_path.cr2")
    assert "Canon" in canonical
    assert "50mm" in canonical
    assert "50.0mm" in summary or "50mm" in summary


def test_detect_raw_lens_empty_when_no_lens(monkeypatch):
    exif = raw_file_processor.ExifData(
        make="",
        model="",
        lens_make="",
        lens_model="",
        focal_length=0.0,
        f_number=0.0,
    )
    monkeypatch.setattr(raw_file_processor, "_read_exif_metadata", lambda p: exif)

    canonical, summary = raw_file_processor.detect_raw_lens("fake_path.cr2")
    assert canonical == ""
    assert summary == ""


def test_apply_lens_correction_manual_selection():
    # Image with no lens in EXIF (e.g. vintage lens with dumb adapter)
    exif = raw_file_processor.ExifData(
        make="Sony",
        model="ILCE-7M3",
        lens_make="",
        lens_model="",
        focal_length=0.0,
        f_number=0.0,
    )
    rgb = np.ones((64, 64, 3), dtype=np.float32)

    # When Auto and no lens in EXIF, no correction is applied
    res_rgb, info_auto = raw_file_processor._apply_lens_correction(rgb, exif, selected_lens="Auto")
    assert info_auto == ""

    # When manually selecting a lens from the database
    res_manual, info_manual = raw_file_processor._apply_lens_correction(
        rgb, exif, selected_lens="Canon - Canon FDn 50mm 1:1.4"
    )
    assert "(Manual)" in info_manual
    assert "Canon FD" in info_manual or "50mm" in info_manual
    assert res_manual.shape == rgb.shape


def test_lens_selector_widget():
    _ensure_qapp()
    widget = LensSelectorWidget()

    assert widget.value == "Auto"
    assert widget.reset_button.isEnabled() is False
    assert "Auto" in widget.status_label.text()

    # Test detected lens update
    widget.set_detected_lens("Canon EF 50mm f/1.8 STM @ 50mm f/2.8")
    assert "Canon EF 50mm" in widget.status_label.text()
    assert "Canon EF 50mm" in widget.list_widget.item(0).text()

    # Search filter
    widget.search_edit.setText("nikkor 50")
    # Non-matching items should be hidden
    visible_items = [
        widget.list_widget.item(i).text()
        for i in range(widget.list_widget.count())
        if not widget.list_widget.item(i).isHidden()
    ]
    assert all("nikkor" in t.lower() or "auto" in t.lower() for t in visible_items)

    # Select a manual lens
    matching_item = None
    for i in range(1, widget.list_widget.count()):
        item = widget.list_widget.item(i)
        if "nikkor" in item.text().lower():
            matching_item = item
            break
    assert matching_item is not None

    widget.list_widget.setCurrentItem(matching_item)
    assert widget.value == matching_item.data(QtCore.Qt.UserRole)
    assert widget.reset_button.isEnabled() is True
    assert "Manual:" in widget.status_label.text()

    # Reset button
    widget.reset_button.click()
    assert widget.value == "Auto"
    assert widget.reset_button.isEnabled() is False
    assert "Auto:" in widget.status_label.text()


def test_load_raw_section_lens_selection(monkeypatch, tmp_path):
    _ensure_qapp()
    section = LoadRawSection()

    assert hasattr(section, "selected_lens")
    assert hasattr(section, "lens_correction")
    assert section.selected_lens.value == "Auto"

    # Initially lens_correction is unchecked, so selected_lens is disabled
    section.lens_correction.value = False
    assert section.selected_lens.isEnabled() is False

    # Checking lens_correction enables selected_lens
    section.lens_correction.value = True
    assert section.selected_lens.isEnabled() is True

    # State round-trip
    state = section.get_state()
    assert state.selected_lens == "Auto"

    state.selected_lens = "Zeiss - Zeiss Milvus 1.4/50"
    state.lens_correction = True
    section.set_state(state)
    assert section.selected_lens.value == "Zeiss - Zeiss Milvus 1.4/50"
    assert section.selected_lens.isEnabled() is True

    # set_detected_lens
    section.set_detected_lens("Detected Lens 50mm f/1.4")
    assert "Detected Lens 50mm" in section.selected_lens.list_widget.item(0).text()


def test_gallery_params_selected_lens():
    assert "selected_lens" in RAW_AFFECTING_PARAM_IDS
    param_def = get_param_def("selected_lens")
    assert param_def is not None

    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    assert param_def.getter(state) == "Auto"
    assert param_def.display_getter(state) == "Auto"

    param_def.setter(state, "Sony - FE 50mm F1.8")
    assert state.gui_only.load_raw.selected_lens == "Sony - FE 50mm F1.8"
    assert param_def.getter(state) == "Sony - FE 50mm F1.8"
    assert param_def.display_getter(state) == "Sony - FE 50mm F1.8"


def test_lens_selector_widget_collapsible():
    _ensure_qapp()
    widget = LensSelectorWidget()

    # By default it should be collapsed
    assert widget.is_picker_expanded is False
    assert widget.picker_container.isHidden() is True
    assert "Browse Lenses" in widget.toggle_button.text()

    # Expand via toggle button click
    widget.toggle_button.click()
    assert widget.is_picker_expanded is True
    assert widget.picker_container.isHidden() is False
    assert "Close" in widget.toggle_button.text()

    # Collapse via toggle button click
    widget.toggle_button.click()
    assert widget.is_picker_expanded is False
    assert widget.picker_container.isHidden() is True
    assert "Browse Lenses" in widget.toggle_button.text()

    # Direct setter
    widget.set_picker_expanded(True)
    assert widget.is_picker_expanded is True
    assert widget.picker_container.isHidden() is False


def test_load_raw_section_lens_position():
    _ensure_qapp()
    section = LoadRawSection()

    # Find the form layout inside section
    # CollapsibleSection contains _frame which contains form
    collapsible = section.findChild(QtWidgets.QWidget, "sectionCard")
    assert collapsible is not None
    frame = collapsible.findChild(QtWidgets.QFrame, "sectionFrame")
    assert frame is not None

    form = None
    for child in frame.children():
        if isinstance(child, QtWidgets.QFormLayout):
            form = child
            break
    if form is None and frame.layout() is not None:
        form = frame.layout().itemAt(0).widget().layout()

    # Find row of lens_correction and row of selected_lens
    lens_corr_row = -1
    selected_lens_row = -1
    for r in range(form.rowCount()):
        item_field = form.itemAt(r, QtWidgets.QFormLayout.FieldRole)
        item_label = form.itemAt(r, QtWidgets.QFormLayout.LabelRole)
        item_spanning = form.itemAt(r, QtWidgets.QFormLayout.SpanningRole)

        widget = None
        if item_field is not None and item_field.widget() is not None:
            widget = item_field.widget()
        elif item_spanning is not None and item_spanning.widget() is not None:
            widget = item_spanning.widget()

        if widget is section.lens_correction:
            lens_corr_row = r
        elif widget is section.selected_lens:
            selected_lens_row = r

    assert lens_corr_row >= 0
    assert selected_lens_row == lens_corr_row + 1, f"selected_lens (row {selected_lens_row}) must be directly under lens_correction (row {lens_corr_row})"


def test_accent_color_selector():
    _ensure_qapp()
    from spektrafilm_gui.widget_editors import AccentColorSelectorWidget
    from spektrafilm_gui.theme import get_accent_color, set_accent_color
    from spektrafilm_gui.theme_palette import ACCENT_PRESETS

    orig_color = get_accent_color()
    try:
        widget = AccentColorSelectorWidget()
        assert widget.current_color == orig_color

        received = []
        widget.colorChanged.connect(received.append)

        # Select Emerald (#10b981)
        emerald_hex = dict(ACCENT_PRESETS)["Emerald"]
        widget.select_color(emerald_hex)

        assert widget.current_color == emerald_hex
        assert get_accent_color() == emerald_hex
        assert received == [emerald_hex]
    finally:
        set_accent_color(orig_color)
