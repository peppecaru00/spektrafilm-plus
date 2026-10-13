from __future__ import annotations

from pathlib import Path

from qtpy import QtCore

from spektrafilm_gui.gallery_model import GalleryItem, GalleryModel
from spektrafilm_gui.gallery_sidecar import save_sidecar
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state


def test_gallery_item_properties(tmp_path: Path):
    raw_path = tmp_path / "DSC_0001.NEF"
    raw_path.write_bytes(b"dummy")

    item = GalleryItem(path=raw_path)
    assert item.filename == "DSC_0001.NEF"
    assert item.stem == "DSC_0001"
    assert item.extension == ".nef"
    assert item.is_raw is True
    assert item.has_settings is False

    # Save sidecar
    save_sidecar(raw_path, PROJECT_DEFAULT_GUI_STATE)
    assert item.has_settings is True


def test_gallery_model_load_folder(tmp_path: Path):
    # Populate folder with mixed supported and unsupported files
    (tmp_path / "img_02.rw2").write_bytes(b"")
    (tmp_path / "img_01.arw").write_bytes(b"")
    (tmp_path / "img_03.jpg").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore me")

    model = GalleryModel()

    items_changed_fired = []
    current_changed_fired = []

    model.items_changed.connect(lambda: items_changed_fired.append(True))
    model.current_changed.connect(lambda idx, it: current_changed_fired.append((idx, it)))

    count = model.load_folder(tmp_path)
    assert count == 3
    assert model.count == 3
    assert len(items_changed_fired) == 1
    assert len(current_changed_fired) == 1
    assert current_changed_fired[0][0] == 0

    # Natural sort order
    names = [it.filename for it in model.items]
    assert names == ["img_01.arw", "img_02.rw2", "img_03.jpg"]
    assert model.current_index == 0
    assert model.current_item.filename == "img_01.arw"


def test_gallery_model_add_files(tmp_path: Path):
    f1 = tmp_path / "b.png"
    f2 = tmp_path / "a.dng"
    f1.write_bytes(b"")
    f2.write_bytes(b"")

    model = GalleryModel()
    added = model.add_files([f1, f2, f1])  # Duplicate f1 should be ignored
    assert added == 2
    assert model.count == 2
    names = [it.filename for it in model.items]
    assert names == ["a.dng", "b.png"]


def test_gallery_model_navigation(tmp_path: Path):
    for i in range(3):
        (tmp_path / f"photo_{i:02d}.jpg").write_bytes(b"")

    model = GalleryModel()
    model.load_folder(tmp_path)
    assert model.current_index == 0

    # Next
    assert model.next_photo() is True
    assert model.current_index == 1
    assert model.next_photo() is True
    assert model.current_index == 2
    assert model.next_photo() is False  # At end
    assert model.current_index == 2

    # Previous
    assert model.previous_photo() is True
    assert model.current_index == 1
    assert model.previous_photo() is True
    assert model.current_index == 0
    assert model.previous_photo() is False  # At start
    assert model.current_index == 0

    # Select path
    target = tmp_path / "photo_02.jpg"
    assert model.select_path(target) is True
    assert model.current_index == 2


def test_gallery_model_copy_paste_look():
    model = GalleryModel()
    assert model.clipboard_state is None
    assert model.paste_look() is None

    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.camera.exposure_compensation_ev = 2.5
    model.copy_look(state)

    assert model.clipboard_state is not None
    pasted = model.paste_look()
    assert pasted is not None
    assert pasted.camera.exposure_compensation_ev == 2.5

    # Modifying pasted does not affect clipboard
    pasted.camera.exposure_compensation_ev = -1.0
    assert model.paste_look().camera.exposure_compensation_ev == 2.5


def test_gallery_model_notify_item_saved(tmp_path: Path):
    f1 = tmp_path / "photo.arw"
    f1.write_bytes(b"")

    model = GalleryModel()
    model.add_files([f1])

    updates = []
    model.item_status_updated.connect(lambda idx: updates.append(idx))

    model.notify_item_saved(f1)
    assert updates == [0]

    model.notify_item_saved(0)
    assert updates == [0, 0]


def test_gallery_model_remove_does_not_delete_file(tmp_path: Path):
    f1 = tmp_path / "img_01.jpg"
    f2 = tmp_path / "img_02.jpg"
    f3 = tmp_path / "img_03.jpg"
    f1.write_bytes(b"1")
    f2.write_bytes(b"2")
    f3.write_bytes(b"3")

    model = GalleryModel()
    model.load_folder(tmp_path)
    assert model.count == 3
    assert model.current_index == 0

    # Remove photo at index 1 (img_02)
    assert model.remove_item(1) is True
    assert model.count == 2
    assert [it.filename for it in model.items] == ["img_01.jpg", "img_03.jpg"]
    assert model.current_index == 0

    # Ensure file still exists on disk!
    assert f1.exists()
    assert f2.exists()
    assert f3.exists()

    # Select last photo and remove it
    model.select_index(1)
    assert model.current_item.filename == "img_03.jpg"
    model.remove_current_photo()
    assert model.count == 1
    assert model.current_index == 0
    assert model.current_item.filename == "img_01.jpg"

    # Remove last remaining photo -> queue empty
    model.remove_current_photo()
    assert model.count == 0
    assert model.current_index == -1
    assert model.current_item is None
    assert f1.exists()


def test_gallery_model_dng_deduplication(tmp_path: Path):
    f_nef = tmp_path / "DSC_0001.NEF"
    f_dng = tmp_path / "DSC_0001.dng"
    f_other = tmp_path / "DSC_0002.NEF"
    f_nef.write_bytes(b"nef_data")
    f_dng.write_bytes(b"dng_data")
    f_other.write_bytes(b"other_nef")

    model = GalleryModel()
    count = model.load_folder(tmp_path)
    assert count == 2
    filenames = [it.filename for it in model.items]
    assert "DSC_0001.dng" in filenames
    assert "DSC_0001.NEF" not in filenames
    assert "DSC_0002.NEF" in filenames

    # Test add_files deduplication
    f_other_dng = tmp_path / "DSC_0002.dng"
    f_other_dng.write_bytes(b"other_dng")
    model.add_files([f_other_dng])
    filenames = [it.filename for it in model.items]
    assert "DSC_0002.dng" in filenames
    assert "DSC_0002.NEF" not in filenames


def test_gallery_model_replace_item_path(tmp_path: Path):
    f_nef = tmp_path / "photo.nef"
    f_dng = tmp_path / "photo.dng"
    f_nef.write_bytes(b"nef")
    f_dng.write_bytes(b"dng")

    model = GalleryModel()
    model.add_files([f_nef])
    assert model.items[0].filename == "photo.nef"

    status_updates = []
    model.item_status_updated.connect(lambda idx: status_updates.append(idx))

    assert model.replace_item_path(f_nef, f_dng) is True
    assert model.items[0].filename == "photo.dng"
    assert status_updates == [0]


def test_gallery_model_remove_items(tmp_path: Path):
    paths = [tmp_path / f"img_{i:02d}.jpg" for i in range(5)]
    for i, p in enumerate(paths):
        p.write_bytes(f"{i}".encode())

    model = GalleryModel()
    model.load_folder(tmp_path)
    assert model.count == 5

    # Multi-select items 1 and 3
    model.set_selected(1, True)
    model.set_selected(3, True)
    assert model.selected_indices == {1, 3}

    # Set active to item 2
    model.select_index(2)
    assert model.current_item.filename == "img_02.jpg"

    # Batch remove items 1 and 3
    removed = model.remove_items([1, 3])
    assert len(removed) == 2
    assert model.count == 3
    assert [it.filename for it in model.items] == ["img_00.jpg", "img_02.jpg", "img_04.jpg"]

    # Active photo was img_02.jpg, which shifted from index 2 to index 1
    assert model.current_index == 1
    assert model.current_item.filename == "img_02.jpg"
    assert model.selected_count == 0

    # Ensure disk files remain
    for p in paths:
        assert p.exists()

    # Now select all and remove all
    model.select_all()
    assert model.selected_count == 3
    model.remove_items(list(model.selected_indices))
    assert model.count == 0
    assert model.current_index == -1
    assert model.current_item is None


