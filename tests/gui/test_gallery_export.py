from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy import QtWidgets

from spektrafilm_gui.gallery_export_dialog import (
    BatchExportWorker,
    ExportOptions,
    format_duration,
)
from spektrafilm_gui.gallery_model import GalleryItem
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE

_q_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_format_duration():
    assert format_duration(0) == "00m 00s"
    assert format_duration(65) == "01m 05s"
    assert format_duration(3665) == "1h 01m 05s"


def test_batch_export_worker_runs_and_emits(tmp_path: Path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    out_dir = tmp_path / "output"

    p1 = photos_dir / "p1.jpg"
    p2 = photos_dir / "p2.jpg"
    p1.write_bytes(b"dummy")
    p2.write_bytes(b"dummy")

    items = [GalleryItem(path=p1), GalleryItem(path=p2)]
    options = ExportOptions(
        output_dir=out_dir,
        output_format="jpg",
        scale_factor=1.0,
        overwrite=True,
    )

    # Mock simulator and image IO
    dummy_img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr("spektrafilm.utils.io.load_image_oiio", lambda path: dummy_img)
    monkeypatch.setattr("spektrafilm.utils.io.read_image_metadata", lambda path: {})
    monkeypatch.setattr("spektrafilm_gui.gallery_export_dialog.write_image_metadata", lambda *args, **kwargs: None)

    saved_files = []
    def fake_save_image_oiio(filepath, img, **kwargs):
        Path(filepath).write_bytes(b"saved_dummy")
        saved_files.append(filepath)

    monkeypatch.setattr("spektrafilm_gui.gallery_export_dialog.save_image_oiio", fake_save_image_oiio)

    class FakeSimulator:
        def __init__(self, *args, **kwargs):
            pass
        def process(self, img):
            return img

    monkeypatch.setattr("spektrafilm_gui.gallery_export_dialog.Simulator", FakeSimulator)

    worker = BatchExportWorker(
        items=items,
        options=options,
        fallback_state=PROJECT_DEFAULT_GUI_STATE,
    )

    progress_events = []
    finished_events = []

    worker.signals.item_progress.connect(lambda cur, tot, name, dur, eta: progress_events.append((cur, tot, name)))
    worker.signals.finished.connect(lambda succ, fail: finished_events.append((succ, fail)))

    worker.run()

    assert len(progress_events) == 2
    assert progress_events[0] == (1, 2, "p1.jpg")
    assert progress_events[1] == (2, 2, "p2.jpg")
    assert finished_events == [(2, 0)]
    assert (out_dir / "p1.jpg").exists()
    assert (out_dir / "p2.jpg").exists()


def test_export_options_dialog_scope(tmp_path: Path):
    from spektrafilm_gui.gallery_export_dialog import ExportOptionsDialog

    out_dir = tmp_path / "output"

    # 1. Dialog with selected photos available and default_scope="selected"
    dialog = ExportOptionsDialog(
        None,
        default_output_dir=out_dir,
        total_photos=10,
        selected_photos=3,
        default_scope="selected",
    )
    assert dialog.scope_selected_radio.isChecked() is True
    assert dialog.scope_selected_radio.isEnabled() is True
    assert "3 selected" in dialog.header.text()

    # Simulate clicking Start
    dialog._on_start()
    opts = dialog.get_options()
    assert opts is not None
    assert opts.scope == "selected"

    # 2. Switch to all photos
    dialog.scope_all_radio.setChecked(True)
    assert "10 gallery" in dialog.header.text()
    dialog._on_start()
    opts_all = dialog.get_options()
    assert opts_all.scope == "all"

    # 3. Dialog with 0 selected photos disables the selected radio
    dialog_zero = ExportOptionsDialog(
        None,
        default_output_dir=out_dir,
        total_photos=5,
        selected_photos=0,
        default_scope="all",
    )
    assert dialog_zero.scope_selected_radio.isEnabled() is False
    assert dialog_zero.scope_all_radio.isChecked() is True
