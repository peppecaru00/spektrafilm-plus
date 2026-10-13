from __future__ import annotations

import os
from pathlib import Path
import pytest
from qtpy import QtCore

from spektrafilm_gui.gallery_model import GalleryModel


@pytest.fixture
def sample_gallery(tmp_path: Path) -> GalleryModel:
    model = GalleryModel()
    paths = []
    for i in range(5):
        p = tmp_path / f"photo_{i:02d}.jpg"
        p.write_text("fake image content")
        paths.append(p)
    model.add_files(paths)
    return model


def test_initial_selection_state(sample_gallery: GalleryModel):
    assert sample_gallery.count == 5
    assert sample_gallery.selected_count == 0
    assert sample_gallery.selected_indices == frozenset()
    assert not sample_gallery.is_selected(0)


def test_toggle_and_set_selection(sample_gallery: GalleryModel):
    signals_received = []
    sample_gallery.selection_changed.connect(lambda: signals_received.append(1))

    # Toggle photo 1
    res = sample_gallery.toggle_selection(1)
    assert res is True
    assert sample_gallery.is_selected(1)
    assert sample_gallery.selected_indices == {1}
    assert sample_gallery.selected_count == 1
    assert len(signals_received) == 1

    # Toggle photo 3
    sample_gallery.toggle_selection(3)
    assert sample_gallery.selected_indices == {1, 3}
    assert sample_gallery.selected_count == 2
    assert len(signals_received) == 2

    # Toggle photo 1 off
    res = sample_gallery.toggle_selection(1)
    assert res is False
    assert not sample_gallery.is_selected(1)
    assert sample_gallery.selected_indices == {3}
    assert len(signals_received) == 3

    # set_selected explicitly
    sample_gallery.set_selected(4, True)
    assert sample_gallery.is_selected(4)
    assert sample_gallery.selected_indices == {3, 4}

    sample_gallery.set_selected(4, True)  # No change
    assert len(signals_received) == 4

    sample_gallery.set_selected(4, False)
    assert not sample_gallery.is_selected(4)
    assert len(signals_received) == 5


def test_select_all_and_deselect_all(sample_gallery: GalleryModel):
    assert sample_gallery.selected_count == 0

    sample_gallery.select_all()
    assert sample_gallery.selected_count == 5
    assert sample_gallery.selected_indices == {0, 1, 2, 3, 4}

    sample_gallery.deselect_all()
    assert sample_gallery.selected_count == 0
    assert sample_gallery.selected_indices == set()


def test_select_range(sample_gallery: GalleryModel):
    # Select range 1 to 3
    sample_gallery.select_range(1, 3)
    assert sample_gallery.selected_indices == {1, 2, 3}

    # Range with reversed endpoints: 4 down to 2
    sample_gallery.select_range(4, 2)
    assert sample_gallery.selected_indices == {2, 3, 4}

    # Extend existing range
    sample_gallery.select_range(0, 1, extend=True)
    assert sample_gallery.selected_indices == {0, 1, 2, 3, 4}


def test_get_target_indices(sample_gallery: GalleryModel):
    # Case 1: No selection -> falls back to current index
    assert sample_gallery.current_index == 0
    assert sample_gallery.selected_count == 0
    assert sample_gallery.get_target_indices(fallback_to_current=True) == [0]

    # Without fallback
    assert sample_gallery.get_target_indices(fallback_to_current=False) == []

    # Case 2: Multi-selection active
    sample_gallery.toggle_selection(2)
    sample_gallery.toggle_selection(4)
    assert sample_gallery.get_target_indices() == [2, 4]


def test_selection_adjusted_on_item_removal(sample_gallery: GalleryModel):
    # Select items 1, 3, 4
    sample_gallery.set_selected(1, True)
    sample_gallery.set_selected(3, True)
    sample_gallery.set_selected(4, True)
    assert sample_gallery.selected_indices == {1, 3, 4}

    # Remove item 2 (between 1 and 3)
    sample_gallery.remove_item(2)
    # Item 1 stays 1. Items 3 and 4 shift down by 1 -> 2 and 3.
    assert sample_gallery.selected_indices == {1, 2, 3}
    assert sample_gallery.count == 4

    # Remove item 1 (which was selected)
    sample_gallery.remove_item(1)
    # Item 1 was removed. Old items 2, 3 shift to 1, 2.
    assert sample_gallery.selected_indices == {1, 2}
    assert sample_gallery.count == 3


def test_selection_cleared_on_clear_and_load(sample_gallery: GalleryModel, tmp_path: Path):
    sample_gallery.select_all()
    assert sample_gallery.selected_count == 5

    sample_gallery.clear()
    assert sample_gallery.selected_count == 0
    assert sample_gallery.count == 0


def test_single_parameter_clipboard(sample_gallery: GalleryModel):
    assert not sample_gallery.has_param_clipboard
    assert sample_gallery.get_single_param_clipboard() is None

    sample_gallery.copy_single_param("film_stock", "Fujifilm Pro 400H")
    assert sample_gallery.has_param_clipboard
    assert sample_gallery.param_clipboard_id == "film_stock"
    assert sample_gallery.param_clipboard_value == "Fujifilm Pro 400H"
    assert sample_gallery.get_single_param_clipboard() == ("film_stock", "Fujifilm Pro 400H")

    sample_gallery.clear_param_clipboard()
    assert not sample_gallery.has_param_clipboard
    assert sample_gallery.get_single_param_clipboard() is None


def test_gallery_cache_eviction_and_full_image_reload(tmp_path: Path):
    from spektrafilm_gui.gallery_cache import GalleryCache, CachedInputImage
    import numpy as np

    cache = GalleryCache(max_full_images=2, max_previews=10)
    for i in range(4):
        p = tmp_path / f"img_{i}.jpg"
        p.write_text("dummy")
        mtime = os.path.getmtime(str(p.resolve()))
        entry = CachedInputImage(
            path=str(p.resolve()),
            mtime=mtime,
            signature=("sig", i),
            full_image=np.zeros((10, 10, 3), dtype=np.float32),
            lens_info={},
            preview_image=np.zeros((5, 5, 3), dtype=np.float32),
            preview_display_image=np.zeros((5, 5, 3), dtype=np.uint8),
            preview_max_size=640,
        )
        cache.put_input(entry)

    # First two should have had full_image evicted to None to preserve RAM
    e0 = cache.get_input(str(tmp_path / "img_0.jpg"), ("sig", 0), 640)
    assert e0 is not None
    assert e0.full_image is None
    assert e0.preview_image is not None

    e1 = cache.get_input(str(tmp_path / "img_1.jpg"), ("sig", 1), 640)
    assert e1 is not None
    assert e1.full_image is None

    # Latest two retain full_image
    e2 = cache.get_input(str(tmp_path / "img_2.jpg"), ("sig", 2), 640)
    assert e2 is not None
    assert e2.full_image is not None

    e3 = cache.get_input(str(tmp_path / "img_3.jpg"), ("sig", 3), 640)
    assert e3 is not None
    assert e3.full_image is not None
