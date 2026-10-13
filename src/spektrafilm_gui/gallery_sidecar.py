from __future__ import annotations

import logging
from pathlib import Path

from spektrafilm_gui.persistence import (
    load_gui_state_from_path,
    save_gui_state_to_path,
)
from spektrafilm_gui.state import GuiState

logger = logging.getLogger(__name__)

SIDECAR_EXTENSION = ".spktrflm"
LEGACY_EXTENSIONS = (".spektrafilm.json", ".json")


def sidecar_path(image_path: str | Path) -> Path:
    """Return the canonical .spktrflm sidecar path for a given image file."""
    p = Path(image_path)
    return p.with_suffix(SIDECAR_EXTENSION)


def find_existing_sidecar(image_path: str | Path) -> Path | None:
    """Return the existing sidecar file path, prioritizing .spktrflm, then legacy names."""
    p = Path(image_path)
    canonical = p.with_suffix(SIDECAR_EXTENSION)
    if canonical.exists():
        return canonical

    # Check potential legacy extensions in the same directory
    for legacy_ext in LEGACY_EXTENSIONS:
        legacy = p.with_suffix(legacy_ext)
        if legacy.exists():
            return legacy

    return None


def has_sidecar(image_path: str | Path) -> bool:
    """Return True if a .spktrflm or legacy sidecar exists for the image."""
    return find_existing_sidecar(image_path) is not None


def save_sidecar(image_path: str | Path, state: GuiState) -> Path:
    """Save the current GUI state non-destructively to the image's .spktrflm sidecar file."""
    dest = sidecar_path(image_path)
    save_gui_state_to_path(state, dest)
    try:
        from spektrafilm_gui.gallery_cache import get_global_gallery_cache
        get_global_gallery_cache().put_sidecar(image_path, state)
    except Exception:
        pass
    logger.debug("Saved sidecar settings to %s", dest)
    return dest


def load_sidecar(image_path: str | Path) -> GuiState | None:
    """Load the GUI state from the sidecar file if present, else return None."""
    try:
        from spektrafilm_gui.gallery_cache import get_global_gallery_cache
        cached = get_global_gallery_cache().get_sidecar(image_path)
        if cached is not None:
            return cached
    except Exception:
        pass

    sidecar_file = find_existing_sidecar(image_path)
    if sidecar_file is None:
        return None

    try:
        state = load_gui_state_from_path(sidecar_file)
        try:
            from spektrafilm_gui.gallery_cache import get_global_gallery_cache
            get_global_gallery_cache().put_sidecar(image_path, state)
        except Exception:
            pass
        return state
    except Exception as exc:
        logger.warning("Failed to load sidecar settings from %s: %s", sidecar_file, exc)
        return None
