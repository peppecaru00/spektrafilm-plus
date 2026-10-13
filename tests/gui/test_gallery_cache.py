from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
from qtpy import QtGui, QtWidgets

from spektrafilm_gui.app import initialize_controller
from spektrafilm_gui.controller_runtime import SimulationResult
from spektrafilm_gui.gallery_cache import (
    CachedInputImage,
    GalleryCache,
    GalleryPrefetchWorker,
    compute_input_signature,
    compute_io_signature,
    compute_raw_signature,
    compute_simulation_signature,
    get_global_gallery_cache,
)
from spektrafilm_gui.gallery_model import GalleryModel
from spektrafilm_gui.gallery_sidecar import load_sidecar, save_sidecar
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state
from spektrafilm_gui.state_bridge import apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import create_widget_bundle

from .helpers import FakeViewer

_q_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_signatures():
    state1 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state2 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)

    sig1 = compute_raw_signature(state1)
    sig2 = compute_raw_signature(state2)
    assert sig1 == sig2

    state2.gui_only.load_raw.temperature = 6500.0
    assert compute_raw_signature(state1) != compute_raw_signature(state2)

    io_sig1 = compute_io_signature(state1)
    state2.input_image.io.input_color_space = "sRGB"
    assert io_sig1 != compute_io_signature(state2)

    # Input signature helper
    assert compute_input_signature(state1, is_raw=True) == sig1
    assert compute_input_signature(state1, is_raw=False) == io_sig1

    # Simulation signatures
    sim_sig1 = compute_simulation_signature(state1, preview_max_size=640, is_raw=True)
    sim_sig2 = compute_simulation_signature(state1, preview_max_size=640, is_raw=True)
    assert sim_sig1 == sim_sig2

    # Different preview size gives different signature
    assert compute_simulation_signature(state1, preview_max_size=320, is_raw=True) != sim_sig1

    # Raw setting changes alter sim signature when is_raw=True
    state_raw_mod = clone_gui_state(state1)
    state_raw_mod.gui_only.load_raw.temperature = 7200.0
    assert compute_simulation_signature(state_raw_mod, preview_max_size=640, is_raw=True) != sim_sig1

    # Simulation setting changes alter sim signature
    state_sim_mod = clone_gui_state(state1)
    state_sim_mod.simulation.selection.film_stock = "Fujicolor Pro 400H"
    assert compute_simulation_signature(state_sim_mod, preview_max_size=640, is_raw=False) != compute_simulation_signature(state1, preview_max_size=640, is_raw=False)


def test_gallery_cache_input_storage_and_lru(tmp_path: Path):
    cache = GalleryCache(max_full_images=2, max_previews=4)
    file1 = tmp_path / "img1.jpg"
    file2 = tmp_path / "img2.jpg"
    file3 = tmp_path / "img3.jpg"
    file1.touch()
    file2.touch()
    file3.touch()

    sig = ("sRGB", True)
    full1 = np.zeros((100, 100, 3), dtype=np.float32)
    prev1 = np.zeros((20, 20, 3), dtype=np.float32)
    disp1 = np.zeros((20, 20, 3), dtype=np.uint8)

    entry1 = CachedInputImage(
        path=str(file1.resolve()),
        mtime=os.path.getmtime(str(file1)),
        signature=sig,
        full_image=full1,
        lens_info={},
        preview_image=prev1,
        preview_display_image=disp1,
        preview_max_size=640,
    )
    cache.put_input(entry1)

    hit = cache.get_input(str(file1), sig, 640)
    assert hit is not None
    assert hit.full_image is not None
    assert hit.preview_image is prev1

    # Wrong signature or preview size is a miss
    assert cache.get_input(str(file1), ("AdobeRGB", True), 640) is None
    assert cache.get_input(str(file1), sig, 320) is None

    # Add entry2 and entry3: exceeding max_full_images=2 should drop full_image from entry1
    entry2 = CachedInputImage(
        path=str(file2.resolve()),
        mtime=os.path.getmtime(str(file2)),
        signature=sig,
        full_image=np.zeros((100, 100, 3), dtype=np.float32),
        lens_info={},
        preview_image=prev1,
        preview_display_image=disp1,
        preview_max_size=640,
    )
    cache.put_input(entry2)

    entry3 = CachedInputImage(
        path=str(file3.resolve()),
        mtime=os.path.getmtime(str(file3)),
        signature=sig,
        full_image=np.zeros((100, 100, 3), dtype=np.float32),
        lens_info={},
        preview_image=prev1,
        preview_display_image=disp1,
        preview_max_size=640,
    )
    cache.put_input(entry3)

    # entry1's full_image is evicted, but preview_image and preview_display_image are preserved!
    hit1 = cache.get_input(str(file1), sig, 640)
    assert hit1 is not None
    assert hit1.full_image is None
    assert hit1.preview_image is not None


def test_gallery_cache_simulation_result(tmp_path: Path):
    cache = GalleryCache(max_sim_results=2)
    file1 = tmp_path / "img1.jpg"
    file1.touch()

    sim_res = SimulationResult(
        mode_label="Preview",
        display_image=np.zeros((20, 20, 3), dtype=np.uint8),
        float_image=np.zeros((20, 20, 3), dtype=np.float32),
        output_color_space="sRGB",
        use_display_transform=True,
        status_message="OK",
    )
    cache.put_simulation(str(file1), "sig_abc", sim_res)

    assert cache.get_simulation(str(file1), "sig_abc") is sim_res
    assert cache.get_simulation(str(file1), "different_sig") is None

    # Invalidate path
    cache.invalidate_path(str(file1))
    assert cache.get_simulation(str(file1), "sig_abc") is None


def test_gallery_cache_sidecar(tmp_path: Path):
    cache = GalleryCache()
    file1 = tmp_path / "img1.jpg"
    file1.touch()

    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.camera.exposure_compensation_ev = 2.5
    cache.put_sidecar(str(file1), state)

    cached_state = cache.get_sidecar(str(file1))
    assert cached_state is not None
    assert cached_state.camera.exposure_compensation_ev == 2.5


def test_gallery_prefetch_worker(tmp_path: Path):
    cache = GalleryCache()
    p1 = tmp_path / "test1.jpg"
    p2 = tmp_path / "test2.jpg"
    p1.touch()
    p2.touch()

    dummy_rgb = np.ones((50, 50, 3), dtype=np.float32)
    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)

    worker = GalleryPrefetchWorker(
        targets=[(str(p1), False, state), (str(p2), False, state)],
        cache=cache,
        preview_max_size=640,
        load_raw_fn=lambda *a, **k: dummy_rgb,
        load_rgb_fn=lambda p: dummy_rgb,
        resize_fn=lambda img, max_size=640: img,
        prepare_preview_fn=lambda img, **k: (img * 255).astype(np.uint8),
        generation=1,
        get_current_generation_fn=lambda: 1,
    )

    worker.run()

    sig = compute_input_signature(state, is_raw=False)
    hit1 = cache.get_input(str(p1), sig, 640)
    hit2 = cache.get_input(str(p2), sig, 640)

    assert hit1 is not None
    assert hit2 is not None
    assert hit1.preview_image is not None


def test_controller_cache_hit_on_filmstrip_switch(tmp_path: Path, monkeypatch):
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

    loaded_paths = []
    def mock_load(path):
        loaded_paths.append(str(Path(path).name))
        return np.full((10, 10, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", mock_load)

    model = controller._gallery_model
    assert model is not None
    model.load_folder(tmp_path)
    controller._thread_pool.waitForDone(2000)

    assert model.count == 2
    assert model.current_index == 0
    # Both photos are loaded into cache (p1 loaded on select, p2 prefetched)
    assert set(loaded_paths) == {"img_01.jpg", "img_02.jpg"}
    assert len(loaded_paths) == 2

    # Switch to p2 -> MUST HIT CACHE, no disk read!
    model.next_photo()
    controller._thread_pool.waitForDone(2000)
    assert model.current_index == 1
    assert len(loaded_paths) == 2

    # Switch BACK to p1 -> MUST HIT CACHE, no disk read!
    model.previous_photo()
    controller._thread_pool.waitForDone(2000)
    assert model.current_index == 0
    assert len(loaded_paths) == 2

    # Switch BACK to p2 -> MUST HIT CACHE as well!
    model.next_photo()
    controller._thread_pool.waitForDone(2000)
    assert model.current_index == 1
    assert len(loaded_paths) == 2


def test_controller_raw_cache_and_reprocess(tmp_path: Path, monkeypatch):
    raw1 = tmp_path / "raw_01.dng"
    raw2 = tmp_path / "raw_02.dng"
    raw1.touch()
    raw2.touch()

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    raw_load_count = 0
    def mock_raw_load(path, **kwargs):
        nonlocal raw_load_count
        raw_load_count += 1
        lens_info_out = kwargs.get("lens_info_out")
        if isinstance(lens_info_out, dict):
            lens_info_out["summary"] = "Test Lens"
        return np.full((12, 12, 3), 0.4, dtype=np.float32)

    monkeypatch.setattr("spektrafilm_gui.controller.load_and_process_raw_file", mock_raw_load)

    model = controller._gallery_model
    assert model is not None
    model.load_folder(tmp_path)
    controller._thread_pool.waitForDone(2000)

    assert model.count == 2
    assert model.current_index == 0
    # raw1 decoded on select, raw2 prefetched in background
    assert raw_load_count == 2

    # Switch to raw2 -> MUST HIT CACHE!
    model.next_photo()
    controller._thread_pool.waitForDone(2000)
    assert model.current_index == 1
    assert raw_load_count == 2

    # Switch back to raw1 -> MUST HIT CACHE!
    model.previous_photo()
    controller._thread_pool.waitForDone(2000)
    assert model.current_index == 0
    assert raw_load_count == 2

    # Modifying RAW setting (e.g. temperature) should cause raw re-processing
    widgets.load_raw.temperature.setValue(6800.0)
    controller.reprocess_current_raw()
    controller._thread_pool.waitForDone(2000)
    assert raw_load_count == 3


def test_controller_simulation_result_caching(tmp_path: Path, monkeypatch):
    p1 = tmp_path / "img_01.jpg"
    p1.touch()

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", lambda p: np.full((10, 10, 3), 0.5, dtype=np.float32))

    controller.load_input_image(str(p1))
    controller._thread_pool.waitForDone(2000)

    # Simulate simulation result
    sim_result = SimulationResult(
        mode_label="Preview",
        display_image=np.full((10, 10, 3), 200, dtype=np.uint8),
        float_image=np.full((10, 10, 3), 0.8, dtype=np.float32),
        output_color_space="sRGB",
        use_display_transform=True,
        status_message="OK",
    )
    controller._on_simulation_finished(sim_result)

    # Check that simulation result was put into gallery cache
    state = collect_gui_state(widgets=widgets)
    sim_sig = compute_simulation_signature(state, preview_max_size=640, is_raw=False)
    cached = controller._gallery_cache.get_simulation(str(p1), sim_sig)
    assert cached is not None
    assert cached is sim_result


def test_controller_photo_removal_invalidates_cache(tmp_path: Path, monkeypatch):
    p1 = tmp_path / "img_01.jpg"
    p2 = tmp_path / "img_02.jpg"
    p1.touch()
    p2.touch()

    viewer = FakeViewer()
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)

    monkeypatch.setattr("spektrafilm_gui.controller.load_image_oiio", lambda p: np.full((10, 10, 3), 0.5, dtype=np.float32))

    model = controller._gallery_model
    assert model is not None
    model.load_folder(tmp_path)
    controller._thread_pool.waitForDone(2000)

    state = collect_gui_state(widgets=widgets)
    sig = compute_input_signature(state, is_raw=False)
    assert controller._gallery_cache.get_input(str(p1), sig, 640) is not None

    # Remove photo 1
    controller.gallery_remove_current_photo()
    assert controller._gallery_cache.get_input(str(p1), sig, 640) is None

