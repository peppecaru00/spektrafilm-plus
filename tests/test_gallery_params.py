from __future__ import annotations

import json
from pathlib import Path

import pytest

from spektrafilm_gui.gallery_params import (
    all_param_defs,
    apply_single_param_to_file,
    get_param_def,
)
from spektrafilm_gui.gallery_sidecar import load_sidecar, save_sidecar, sidecar_path
from spektrafilm_gui.persistence import PROJECT_DEFAULT_GUI_STATE
from spektrafilm_gui.state import clone_gui_state


def test_all_param_defs_registered():
    defs = all_param_defs()
    assert len(defs) >= 15
    ids = [d.id for d in defs]
    assert "film_stock" in ids
    assert "print_paper" in ids
    assert "white_balance" in ids
    assert "exposure_compensation" in ids
    assert "lens_correction" in ids
    assert "chroma_denoise" in ids
    assert "grain" in ids
    assert "halation" in ids
    assert "couplers" in ids
    assert "film_density_gamma" in ids
    assert "enlarger" in ids
    assert "chemistry" in ids
    assert "glare" in ids
    assert "camera_diffusion" in ids
    assert "enlarger_diffusion" in ids
    assert "scanner" in ids
    assert "output_color_space" in ids
    assert "print_exposure_compensation" in ids


def test_param_def_extract_and_apply():
    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)

    # 1. Film stock
    stock_def = get_param_def("film_stock")
    assert stock_def is not None
    target_state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    stock_def.apply(target_state, "Kodak Portra 400")
    assert target_state.simulation.selection.film_stock == "Kodak Portra 400"
    assert stock_def.extract(target_state) == "Kodak Portra 400"
    assert "Portra 400" in stock_def.get_display(target_state)

    # 2. Exposure EV
    ev_def = get_param_def("exposure_compensation")
    assert ev_def is not None
    ev_def.apply(target_state, 1.33)
    assert target_state.camera.exposure_compensation_ev == pytest.approx(1.33)
    assert ev_def.extract(target_state) == pytest.approx(1.33)
    assert "+1.33 EV" in ev_def.get_display(target_state)

    # 3. White balance
    wb_def = get_param_def("white_balance")
    assert wb_def is not None
    wb_def.apply(target_state, ("Daylight", 5600.0, 12.5))
    assert target_state.gui_only.load_raw.white_balance == "Daylight"
    assert target_state.gui_only.load_raw.temperature == pytest.approx(5600.0)
    assert target_state.gui_only.load_raw.tint == pytest.approx(12.5)
    assert "5600 K" in wb_def.get_display(target_state)

    # 4. Lens correction
    lens_def = get_param_def("lens_correction")
    assert lens_def is not None
    lens_def.apply(target_state, True)
    assert target_state.gui_only.load_raw.lens_correction is True
    assert lens_def.get_display(target_state) == "Enabled"
    lens_def.apply(target_state, False)
    assert target_state.gui_only.load_raw.lens_correction is False
    assert lens_def.get_display(target_state) == "Disabled"

    # 5. Film grain
    grain_def = get_param_def("grain")
    assert grain_def is not None
    grain_copy = grain_def.extract(state)
    grain_copy.particle_area_um2 = 1.80
    grain_copy.blur = 0.90
    grain_def.apply(target_state, grain_copy)
    assert target_state.grain.particle_area_um2 == pytest.approx(1.80)
    assert target_state.grain.blur == pytest.approx(0.90)
    assert "1.80" in grain_def.get_display(target_state)

    # 6. Print auto compensation
    comp_def = get_param_def("print_exposure_compensation")
    assert comp_def is not None
    comp_def.apply(target_state, False)
    assert target_state.simulation.enlarger.print_exposure_compensation is False
    assert comp_def.extract(target_state) is False
    assert comp_def.get_display(target_state) == "Disabled"
    comp_def.apply(target_state, True)
    assert target_state.simulation.enlarger.print_exposure_compensation is True
    assert comp_def.extract(target_state) is True
    assert comp_def.get_display(target_state) == "Enabled"


def test_apply_single_param_to_file(tmp_path: Path):
    img_path = tmp_path / "test_photo.jpg"
    img_path.write_text("fake image content")

    # Initial sidecar with custom exposure EV = -1.5 and custom film stock = "Kodak Ektar 100"
    base_state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    base_state.camera.exposure_compensation_ev = -1.5
    base_state.simulation.selection.film_stock = "Kodak Ektar 100"
    save_sidecar(img_path, base_state)

    # Now apply ONLY grain parameter to this file
    new_grain = clone_gui_state(PROJECT_DEFAULT_GUI_STATE).grain
    new_grain.particle_area_um2 = 3.50
    ok = apply_single_param_to_file(img_path, "grain", new_grain)
    assert ok is True

    # Verify that the sidecar on disk has updated grain BUT preserved exposure EV and film stock!
    reloaded = load_sidecar(img_path)
    assert reloaded is not None
    assert reloaded.grain.particle_area_um2 == pytest.approx(3.50)
    # Check that previous edits were NOT lost:
    assert reloaded.camera.exposure_compensation_ev == pytest.approx(-1.5)
    assert reloaded.simulation.selection.film_stock == "Kodak Ektar 100"


def test_apply_single_param_to_file_without_existing_sidecar(tmp_path: Path):
    img_path = tmp_path / "fresh_photo.cr3"
    img_path.write_text("raw bytes")

    # Photo has no existing sidecar
    assert not sidecar_path(img_path).exists()

    # Apply White Balance
    ok = apply_single_param_to_file(
        img_path,
        "white_balance",
        ("Tungsten", 3200.0, -5.0),
        fallback_state=PROJECT_DEFAULT_GUI_STATE,
    )
    assert ok is True
    assert sidecar_path(img_path).exists()

    reloaded = load_sidecar(img_path)
    assert reloaded is not None
    assert reloaded.gui_only.load_raw.white_balance == "Tungsten"
    assert reloaded.gui_only.load_raw.temperature == pytest.approx(3200.0)
    assert reloaded.gui_only.load_raw.tint == pytest.approx(-5.0)
