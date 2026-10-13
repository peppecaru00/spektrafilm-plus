from __future__ import annotations

from pathlib import Path

from spektrafilm_gui.gallery_sidecar import (
    SIDECAR_EXTENSION,
    find_existing_sidecar,
    has_sidecar,
    load_sidecar,
    save_sidecar,
    sidecar_path,
)
from spektrafilm_gui.persistence import save_gui_state_to_path
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state


def test_sidecar_path():
    raw = Path("/photos/DSC0001.ARW")
    assert sidecar_path(raw) == Path("/photos/DSC0001.spktrflm")

    rgb = Path("/photos/summer.jpeg")
    assert sidecar_path(rgb) == Path("/photos/summer.spktrflm")


def test_save_and_load_sidecar(tmp_path: Path):
    photo = tmp_path / "test_photo.RW2"
    photo.write_bytes(b"dummy")

    assert not has_sidecar(photo)
    assert load_sidecar(photo) is None

    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.camera.exposure_compensation_ev = 1.25

    saved_path = save_sidecar(photo, state)
    assert saved_path == tmp_path / "test_photo.spktrflm"
    assert saved_path.exists()
    assert has_sidecar(photo)

    loaded = load_sidecar(photo)
    assert loaded is not None
    assert loaded.camera.exposure_compensation_ev == 1.25


def test_legacy_sidecar_fallback(tmp_path: Path):
    photo = tmp_path / "legacy_photo.CR2"
    photo.write_bytes(b"dummy")

    # Create older legacy sidecar
    legacy_sidecar = tmp_path / "legacy_photo.spektrafilm.json"
    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.camera.exposure_compensation_ev = -0.75
    save_gui_state_to_path(state, legacy_sidecar)

    # Should find and load legacy sidecar
    assert has_sidecar(photo)
    found = find_existing_sidecar(photo)
    assert found == legacy_sidecar

    loaded = load_sidecar(photo)
    assert loaded is not None
    assert loaded.camera.exposure_compensation_ev == -0.75

    # If canonical .spktrflm is subsequently saved, it takes priority
    state.camera.exposure_compensation_ev = 2.0
    canonical = save_sidecar(photo, state)
    assert canonical.exists()
    assert find_existing_sidecar(photo) == canonical
    reloaded = load_sidecar(photo)
    assert reloaded.camera.exposure_compensation_ev == 2.0
