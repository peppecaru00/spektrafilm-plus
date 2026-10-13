from __future__ import annotations

import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

import numpy as np
from qtpy import QtCore, QtWidgets

from spektrafilm_gui import controller_persistence as persistence_actions
from spektrafilm_gui import controller_profile_sync as profile_sync
from spektrafilm_gui import controller_runtime as runtime
from spektrafilm_gui.controller_layers import (
    INPUT_LAYER_NAME,
    INPUT_PREVIEW_LAYER_NAME,
    ViewerLayerService,
)
from spektrafilm_gui.persistence import (
    clear_saved_default_gui_state,
    load_dialog_dir,
    load_gui_state_from_path,
    save_default_gui_state,
    save_dialog_dir,
    save_gui_state_to_path,
)
from spektrafilm_gui.gallery_sidecar import load_sidecar, save_sidecar
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
from spektrafilm_gui.gallery_export_dialog import (
    BatchExportWorker,
    BatchProgressDialog,
    ExportOptionsDialog,
)
from spektrafilm_gui.gallery_model import RAW_EXTENSIONS
from spektrafilm_gui.gallery_params import (
    RAW_AFFECTING_PARAM_IDS,
    apply_params_dict_to_file,
    apply_params_dict_to_state,
    apply_single_param_to_file,
    get_param_def,
)
from spektrafilm_gui.gallery_sync_dialog import SyncParameterDialog
from spektrafilm_gui.history_manager import HistoryManager
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state, digest_after_selection, gui_state_from_params
from spektrafilm_gui.napari_layout import dialog_parent, reset_viewer_camera, set_canvas_background, set_status
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.state_bridge import apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import WidgetBundle

OUTPUT_FLOAT_DATA_KEY = 'pipeline_float_output'
OUTPUT_COLOR_SPACE_KEY = 'pipeline_output_color_space'
OUTPUT_CCTF_ENCODING_KEY = 'pipeline_output_cctf_encoding'
OUTPUT_DISPLAY_TRANSFORM_KEY = 'pipeline_use_display_transform'
PROFILE_SYNC_SECTION_NAMES = profile_sync.PROFILE_SYNC_SECTION_NAMES
if TYPE_CHECKING:
    import napari
    from napari.layers import Image as NapariImageLayer


QThreadPool = getattr(QtCore, 'QThreadPool')
QTimer = getattr(QtCore, 'QTimer')
QFileDialog = QtWidgets.QFileDialog
QMessageBox = QtWidgets.QMessageBox
SimulationRequest = runtime.SimulationRequest
SimulationResult = runtime.SimulationResult
STARTUP_PREVIEW_ASPECT_RATIO = (3, 2)


class _DirMemoryDialog:
    """Wraps QFileDialog to open in the last-used directory via QSettings."""

    def __init__(self, key: str) -> None:
        self._key = key

    def get_save_file_name(self, parent, title, filename, file_filter):
        last_dir = load_dialog_dir(self._key)
        initial = str(Path(last_dir) / Path(filename).name) if last_dir else filename
        path, fmt = QFileDialog.getSaveFileName(parent, title, initial, file_filter)
        if path:
            save_dialog_dir(self._key, str(Path(path).parent))
        return path, fmt

    def get_open_file_name(self, parent, title, _initial, file_filter):
        path, fmt = QFileDialog.getOpenFileName(
            parent, title, load_dialog_dir(self._key), file_filter
        )
        if path:
            save_dialog_dir(self._key, str(Path(path).parent))
        return path, fmt


class _LazyModuleProxy:
    def __init__(self, loader):
        self._loader = loader
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = self._loader()
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


def _import_colour_module():
    return import_module('colour')


def _import_pil_image_module():
    return import_module('PIL.Image')


def _import_imagecms_module():
    return import_module('PIL.ImageCms')


def runtime_simulator(*args, **kwargs):
    return import_module('spektrafilm.runtime.api').Simulator(*args, **kwargs)


def digest_params(*args, **kwargs):
    return import_module('spektrafilm.runtime.api').digest_params(*args, **kwargs)


def load_image_oiio(*args, **kwargs):
    return import_module('spektrafilm.utils.io').load_image_oiio(*args, **kwargs)


def save_image_oiio(*args, **kwargs):
    return import_module("spektrafilm.utils.io").save_image_oiio(*args, **kwargs)


def read_image_metadata(*args, **kwargs):
    return import_module("spektrafilm.utils.io").read_image_metadata(*args, **kwargs)


def write_image_metadata(*args, **kwargs):
    return import_module("spektrafilm.utils.io").write_image_metadata(*args, **kwargs)


def load_and_process_raw_file(*args, **kwargs):
    return import_module('spektrafilm.utils.raw_file_processor').load_and_process_raw_file(*args, **kwargs)


def resize_for_preview(*args, **kwargs):
    return import_module('spektrafilm.utils.preview').resize_for_preview(*args, **kwargs)


colour = _LazyModuleProxy(_import_colour_module)
PILImage = _LazyModuleProxy(_import_pil_image_module)
ImageCms = _LazyModuleProxy(_import_imagecms_module)


class GuiController:
    def __init__(self, *, viewer: napari.Viewer, widgets: WidgetBundle):
        self._viewer = viewer
        self._widgets = widgets
        self._layers = ViewerLayerService(
            viewer=viewer,
            output_float_data_key=OUTPUT_FLOAT_DATA_KEY,
            output_color_space_key=OUTPUT_COLOR_SPACE_KEY,
            output_cctf_encoding_key=OUTPUT_CCTF_ENCODING_KEY,
            output_display_transform_key=OUTPUT_DISPLAY_TRANSFORM_KEY,
        )
        self._thread_pool = QThreadPool.globalInstance()
        self._active_simulation_worker: runtime.SimulationWorker | None = None
        self._active_simulation_label: str | None = None
        self._runtime_simulator = None
        self._next_runtime_digest_applies_stock_specifics = True
        self._current_input_image: np.ndarray | None = None
        self._current_input_path: str | None = None
        self._current_preview_image: np.ndarray | None = None
        self._auto_preview_scheduled = False
        self._pending_auto_preview = False
        self._active_simulation_reports_status = True
        self._gallery_model = getattr(widgets, 'gallery_model', None)
        self._gallery_filmstrip = getattr(widgets, 'gallery', None)
        self._suppress_auto_save = False
        self._switching_gallery_item = False
        self._auto_save_timer = QTimer()
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.setInterval(350)
        self._auto_save_timer.timeout.connect(self._flush_auto_save_sidecar)
        self._raw_reprocess_timer = QTimer()
        self._raw_reprocess_timer.setSingleShot(True)
        self._raw_reprocess_timer.setInterval(150)
        self._raw_reprocess_timer.timeout.connect(self.reprocess_current_raw)
        self._gallery_cache = get_global_gallery_cache()
        self._prefetch_generation = 0
        self._history_manager = HistoryManager()
        self._is_undoing_or_redoing = False
        self._history_record_timer = QTimer()
        self._history_record_timer.setSingleShot(True)
        self._history_record_timer.setInterval(350)
        self._history_record_timer.timeout.connect(self._record_debounced_history_state)
        try:
            self._history_manager.reset(self._collect_current_gui_state())
        except Exception:
            pass
        try:
            if hasattr(self._viewer, 'layers') and hasattr(self._viewer.layers, 'events'):
                self._viewer.layers.events.connect(lambda e: self._sync_histogram())
        except Exception:
            pass

    def _collect_current_gui_state(self) -> GuiState:
        try:
            return collect_gui_state(widgets=self._widgets)
        except Exception:
            return clone_gui_state(PROJECT_DEFAULT_GUI_STATE)

    @property
    def history_manager(self) -> HistoryManager:
        return self._history_manager

    @property
    def can_undo(self) -> bool:
        return self._history_manager.can_undo

    @property
    def can_redo(self) -> bool:
        return self._history_manager.can_redo

    def _record_debounced_history_state(self) -> None:
        if self._is_undoing_or_redoing:
            return
        current_state = self._collect_current_gui_state()
        self._history_manager.push(current_state)

    def record_history_state(self, *, immediate: bool = True) -> bool:
        if self._is_undoing_or_redoing:
            return False
        if not immediate:
            self._history_record_timer.start()
            return True
        if self._history_record_timer.isActive():
            self._history_record_timer.stop()
        current_state = self._collect_current_gui_state()
        return self._history_manager.push(current_state)

    def undo(self) -> bool:
        if self._history_record_timer.isActive():
            self._history_record_timer.stop()
            self._record_debounced_history_state()

        if not self._history_manager.can_undo:
            return False

        target_state = self._history_manager.undo()
        if target_state is None:
            return False

        self._apply_restored_state(target_state, action_label="Undo")
        return True

    def redo(self) -> bool:
        if not self._history_manager.can_redo:
            return False

        target_state = self._history_manager.redo()
        if target_state is None:
            return False

        self._apply_restored_state(target_state, action_label="Redo")
        return True

    def _apply_restored_state(self, state: GuiState, *, action_label: str) -> None:
        self._is_undoing_or_redoing = True
        raw_changed = False
        try:
            load_raw_sec = getattr(self._widgets, 'load_raw', None)
            prev_raw = getattr(load_raw_sec, 'get_state', lambda: None)() if load_raw_sec is not None else None
            apply_gui_state(state, widgets=self._widgets)
            self._sync_canvas_background()
            new_raw = getattr(load_raw_sec, 'get_state', lambda: None)() if load_raw_sec is not None else None
            if prev_raw is not None and new_raw is not None:
                raw_changed = (prev_raw != new_raw)
        finally:
            self._is_undoing_or_redoing = False

        if self.is_current_image_raw() and raw_changed and self._current_input_path:
            self.load_raw_image(self._current_input_path)
        else:
            self._request_auto_preview_if_enabled()

        if self._current_input_path is not None and not self._suppress_auto_save:
            self._auto_save_timer.start()

        set_status(self._viewer, f"{action_label}: Settings restored")

    def show_startup_placeholder(self) -> None:
        if self._white_border_layer() is not None:
            return

        state = self._collect_current_gui_state()
        preview_height = max(int(state.gui_only.display.settings.preview_max_size), 1)
        preview_width = max(
            int(round(preview_height * STARTUP_PREVIEW_ASPECT_RATIO[1] / STARTUP_PREVIEW_ASPECT_RATIO[0])),
            1,
        )
        placeholder_preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
        self._layers.set_or_add_input_preview_layer(
            placeholder_preview,
            watermark_source_size=(preview_height, preview_width),
            white_padding=state.gui_only.display.white_padding,
            hide_output=True,
            set_active=True,
        )
        self._home_input_stack()

    def load_input_image(self, path: str) -> None:
        if Path(path).suffix.lower() in RAW_EXTENSIONS:
            self.load_raw_image(path)
            return

        gui_state = self._collect_current_gui_state()
        preview_max_size = int(gui_state.gui_only.display.settings.preview_max_size)
        io_sig = compute_input_signature(gui_state, is_raw=False)
        cached = self._gallery_cache.get_input(path, io_sig, preview_max_size)
        if cached is not None:
            self._current_input_path = path
            self._current_input_image = cached.full_image
            self._current_preview_image = cached.preview_image
            source_size = tuple(int(d) for d in (cached.full_image.shape[:2] if cached.full_image is not None else cached.preview_image.shape[:2]))
            self._layers.set_or_add_input_preview_layer(
                cached.preview_display_image,
                watermark_source_size=source_size,
                white_padding=gui_state.gui_only.display.white_padding,
                hide_output=True,
                set_active=True,
            )
            self._home_input_stack()
            if self._gallery_model is not None and not self._switching_gallery_item:
                self._switching_gallery_item = True
                try:
                    if not self._gallery_model.select_path(path):
                        self._gallery_model.add_files([path])
                        self._gallery_model.select_path(path)
                finally:
                    self._switching_gallery_item = False
            set_status(self._viewer, "Loaded image (cached)")

            sim_sig = compute_simulation_signature(gui_state, preview_max_size=preview_max_size, is_raw=False)
            cached_sim = self._gallery_cache.get_simulation(path, sim_sig)
            if cached_sim is not None:
                self._on_simulation_finished(cached_sim)
            else:
                self._sync_histogram(cached.preview_display_image)
                self._request_auto_preview_if_enabled()
            return

        image = load_image_oiio(path)[..., :3]
        self._current_input_path = path
        self._set_or_add_input_stack(image)

        try:
            canonical = str(Path(path).resolve())
            mtime = os.path.getmtime(canonical)
            preview_img = self._current_preview_image if self._current_preview_image is not None else self._resize_for_preview(image, max_size=preview_max_size)
            preview_disp = self._prepare_input_color_preview_image(
                preview_img,
                input_color_space=gui_state.input_image.io.input_color_space,
                apply_cctf_decoding=gui_state.input_image.io.input_cctf_decoding,
            )
            cached_entry = CachedInputImage(
                path=canonical,
                mtime=mtime,
                signature=io_sig,
                full_image=image,
                lens_info={},
                preview_image=preview_img,
                preview_display_image=preview_disp,
                preview_max_size=preview_max_size,
            )
            self._gallery_cache.put_input(cached_entry)
        except Exception as exc:
            logger.warning("Failed to put input image into cache: %s", exc)

        if self._gallery_model is not None and not self._switching_gallery_item:
            self._switching_gallery_item = True
            try:
                if not self._gallery_model.select_path(path):
                    self._gallery_model.add_files([path])
                    self._gallery_model.select_path(path)
            finally:
                self._switching_gallery_item = False
        self._request_auto_preview_if_enabled()

    def load_raw_image(self, path: str) -> None:
        orig_path = path
        try:
            from spektrafilm.utils.raw_file_processor import resolve_raw_path
            resolved = resolve_raw_path(path)
            path = str(resolved)
        except Exception as exc:
            logger.warning("Could not resolve raw path %s: %s", path, exc)

        if path != orig_path:
            if self._gallery_model is not None:
                self._gallery_model.replace_item_path(orig_path, path)

        load_raw_section = getattr(self._widgets, 'load_raw', None)
        if load_raw_section is not None:
            set_path_fn = getattr(load_raw_section, 'set_path', None)
            if callable(set_path_fn):
                set_path_fn(path)
        gui_state = self._collect_current_gui_state()
        preview_max_size = int(gui_state.gui_only.display.settings.preview_max_size)
        raw_sig = compute_input_signature(gui_state, is_raw=True)
        cached = self._gallery_cache.get_input(path, raw_sig, preview_max_size)
        if cached is not None:
            self._current_input_path = path
            self._current_input_image = cached.full_image
            self._current_preview_image = cached.preview_image
            source_size = tuple(int(d) for d in (cached.full_image.shape[:2] if cached.full_image is not None else cached.preview_image.shape[:2]))
            self._layers.set_or_add_input_preview_layer(
                cached.preview_display_image,
                watermark_source_size=source_size,
                white_padding=gui_state.gui_only.display.white_padding,
                hide_output=True,
                set_active=True,
            )
            self._home_input_stack()

            if self._gallery_model is not None and not self._switching_gallery_item:
                self._switching_gallery_item = True
                try:
                    if not self._gallery_model.select_path(path):
                        self._gallery_model.add_files([path])
                        self._gallery_model.select_path(path)
                finally:
                    self._switching_gallery_item = False

            lens_summary = cached.lens_info.get('summary')
            if lens_summary:
                set_status(self._viewer, f"Loaded raw (cached): {lens_summary}")
            else:
                set_status(self._viewer, "Loaded raw (cached)")

            sim_sig = compute_simulation_signature(gui_state, preview_max_size=preview_max_size, is_raw=True)
            cached_sim = self._gallery_cache.get_simulation(path, sim_sig)
            if cached_sim is not None:
                self._on_simulation_finished(cached_sim)
            else:
                self._sync_histogram(cached.preview_display_image)
                self._request_auto_preview_if_enabled()
            return

        set_status(self._viewer, "Loading raw...", timeout_ms=0)
        lens_info: dict[str, str] = {}
        try:
            image = load_and_process_raw_file(
                path,
                white_balance=gui_state.gui_only.load_raw.white_balance,
                temperature=gui_state.gui_only.load_raw.temperature,
                tint=gui_state.gui_only.load_raw.tint,
                lens_correction=gui_state.gui_only.load_raw.lens_correction,
                chroma_denoise=gui_state.gui_only.load_raw.chroma_denoise,
                chroma_denoise_strength=gui_state.gui_only.load_raw.chroma_denoise_strength,
                raw_exposure_ev=getattr(gui_state.gui_only.load_raw, "raw_exposure_ev", 0.0),
                highlight_recovery=getattr(gui_state.gui_only.load_raw, "highlight_recovery", 1.0),
                highlight_mode=getattr(gui_state.gui_only.load_raw, "highlight_mode", "reconstruct"),
                output_colorspace=gui_state.input_image.io.input_color_space,
                output_cctf_encoding=gui_state.input_image.io.input_cctf_decoding,
                lens_info_out=lens_info,
            )
        except Exception as exc:
            logger.error("Failed to load RAW image %s: %s", path, exc, exc_info=True)
            QMessageBox.critical(dialog_parent(self._viewer), 'Load raw', f'Failed to load RAW image.\n\n{exc}')
            set_status(self._viewer, 'Load raw failed')
            return

        self._current_input_path = path
        self._set_or_add_input_stack(image)

        try:
            canonical = str(Path(path).resolve())
            mtime = os.path.getmtime(canonical)
            preview_img = self._current_preview_image if self._current_preview_image is not None else self._resize_for_preview(image, max_size=preview_max_size)
            preview_disp = self._prepare_input_color_preview_image(
                preview_img,
                input_color_space=gui_state.input_image.io.input_color_space,
                apply_cctf_decoding=gui_state.input_image.io.input_cctf_decoding,
            )
            cached_entry = CachedInputImage(
                path=canonical,
                mtime=mtime,
                signature=raw_sig,
                full_image=image,
                lens_info=lens_info,
                preview_image=preview_img,
                preview_display_image=preview_disp,
                preview_max_size=preview_max_size,
            )
            self._gallery_cache.put_input(cached_entry)
        except Exception:
            pass

        if self._gallery_model is not None and not self._switching_gallery_item:
            self._switching_gallery_item = True
            try:
                if not self._gallery_model.select_path(path):
                    self._gallery_model.add_files([path])
                    self._gallery_model.select_path(path)
            finally:
                self._switching_gallery_item = False

        lens_summary = lens_info.get('summary')
        if lens_summary:
            set_status(
                self._viewer,
                f"Loaded raw and applied lens correction: {lens_summary}",
            )
        elif gui_state.gui_only.load_raw.lens_correction:
            set_status(self._viewer, "Loaded raw, lens correction not applied")
        else:
            set_status(self._viewer, "Loaded raw")
        self._request_auto_preview_if_enabled()

    def is_current_image_raw(self) -> bool:
        if self._gallery_model is not None and self._gallery_model.current_item is not None and self._gallery_model.current_item.is_raw:
            return True
        if self._current_input_path is not None and Path(self._current_input_path).suffix.lower() in RAW_EXTENSIONS:
            return True
        load_raw_section = getattr(self._widgets, 'load_raw', None)
        if load_raw_section is not None:
            raw_text = getattr(getattr(load_raw_section, 'file_path', None), 'text', lambda: '')()
            if raw_text and Path(raw_text).suffix.lower() in RAW_EXTENSIONS:
                return True
        return False

    def reprocess_current_raw(self) -> None:
        path = self._current_input_path
        if not path:
            load_raw_section = getattr(self._widgets, 'load_raw', None)
            if load_raw_section is not None:
                path = getattr(getattr(load_raw_section, 'file_path', None), 'text', lambda: '')().strip()
        if path and self.is_current_image_raw():
            self.load_raw_image(path)

    def on_raw_setting_changed(self, *_args: Any) -> None:
        if self.is_current_image_raw():
            self.reprocess_current_raw()

    def schedule_raw_setting_reprocess(self, *_args: Any) -> None:
        if self.is_current_image_raw():
            self._raw_reprocess_timer.start()

    def refresh_preview_cache(self, *_args) -> None:
        if self._current_input_image is None:
            self._ensure_input_loaded(require_full=True)
        input_image = self._current_input_image
        if input_image is None:
            return
        self._update_preview_cache(
            input_image,
            home_input_stack=False,
            hide_output=False,
        )

    def rotate_input_image_clockwise(self) -> None:
        self._rotate_input_image(quarter_turns=-1)

    def rotate_input_image_counterclockwise(self) -> None:
        self._rotate_input_image(quarter_turns=1)

    def _rotate_input_image(self, *, quarter_turns: int) -> None:
        if self._current_input_image is None:
            self._ensure_input_loaded(require_full=True)
        input_image = self._current_input_image
        if input_image is None:
            return

        rotated_image = np.rot90(np.asarray(input_image), k=int(quarter_turns))
        self._update_preview_cache(
            rotated_image,
            home_input_stack=True,
            hide_output=True,
        )
        self._request_auto_preview_if_enabled()

    def apply_profile_defaults(self, _selected_value: str) -> None:
        state = collect_gui_state(widgets=self._widgets)
        if not state.selection.film_stock or not state.selection.print_paper:
            return

        params = build_params_from_state(state)
        synced_state = gui_state_from_params(
            digest_after_selection(params),
            film_stock=state.selection.film_stock,
            print_paper=state.selection.print_paper,
        )
        self._apply_profile_sync_state(synced_state)
        self._next_runtime_digest_applies_stock_specifics = True
        self.record_history_state(immediate=True)

    def apply_film_profile_defaults(self, film_stock: str) -> None:
        self.apply_profile_defaults(film_stock)

    def _apply_profile_sync_state(self, synced_state) -> None:
        profile_sync.apply_profile_sync_state(
            widgets=self._widgets,
            synced_state=synced_state,
            profile_sync_section_names=PROFILE_SYNC_SECTION_NAMES,
        )

    def run_preview(self) -> None:
        self._run_preview(report_status=True)

    def _run_preview(self, *, report_status: bool) -> None:
        self._start_simulation(
            source_layer_name=INPUT_PREVIEW_LAYER_NAME,
            mode_label='Preview',
            report_status=report_status,
        )

    def run_scan(self) -> None:
        self._start_simulation(source_layer_name=INPUT_LAYER_NAME, mode_label='Scan')

    def request_auto_preview(self, *_args) -> None:
        if self._is_undoing_or_redoing:
            return
        if self._current_input_path is not None and not self._suppress_auto_save:
            self._auto_save_timer.start()
        self._history_record_timer.start()
        if self._auto_preview_scheduled:
            return
        self._auto_preview_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_auto_preview)

    def _request_auto_preview_if_enabled(self) -> None:
        if not self._auto_preview_enabled() or self._current_preview_image is None:
            return
        self.request_auto_preview()

    def report_display_transform_status(self, enabled: bool) -> None:
        if enabled and not self.sync_display_transform_availability(report_status=True):
            return
        set_status(self._viewer, runtime.display_transform_status_message(enabled, imagecms_module=ImageCms))

    def set_gray_18_canvas_enabled(self, enabled: bool) -> None:
        set_canvas_background(self._viewer, gray_18_canvas=enabled)

    def set_output_interpolation_mode(self, mode: str) -> None:
        output_layer = self._output_layer()
        if output_layer is None:
            return
        self._layers.set_output_layer_interpolation(output_layer, mode)

    def sync_display_transform_availability(self, *, report_status: bool) -> bool:
        if runtime.display_profile_available(imagecms_module=ImageCms):
            return True

        self._set_display_transform_checked(False)
        if report_status:
            set_status(self._viewer, 'Display transform unavailable: no display profile detected, disabled')
        return False

    def save_output_layer(self, output_format: str = 'jpg') -> None:
        output_layer = self._output_layer()
        if output_layer is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Save output', 'Run a simulation before saving the output layer.')
            return

        fmt = (output_format or 'jpg').lower().lstrip('.')
        if fmt in {'jpeg', 'jpg'}:
            fmt = 'jpg'
        elif fmt in {'tiff', 'tif'}:
            fmt = 'tif'
        elif fmt == 'png':
            fmt = 'png'
        elif fmt == 'webp':
            fmt = 'webp'

        if self._current_input_path is not None:
            default_name = Path(self._current_input_path).stem + f'.{fmt}'
        else:
            default_name = f'output.{fmt}'

        format_filter_specs = {
            'tif': 'TIFF Image (*.tif *.tiff)',
            'png': 'PNG Image (*.png)',
            'jpg': 'JPEG Image (*.jpg *.jpeg)',
            'webp': 'WebP Image (*.webp)',
        }
        all_supported = 'All Supported Images (*.jpg *.jpeg *.png *.tif *.tiff *.webp *.exr)'
        all_files = 'All Files (*.*)'

        primary_filter = format_filter_specs.get(fmt, f'{fmt.upper()} (*.{fmt})')
        other_filters = [desc for f, desc in format_filter_specs.items() if f != fmt]
        file_filter = ';;'.join([primary_filter, all_supported, *other_filters, all_files])

        filepath, _ = _DirMemoryDialog('save_output').get_save_file_name(
            dialog_parent(self._viewer),
            'Save output image',
            default_name,
            file_filter,
        )
        if not filepath:
            return

        if not Path(filepath).suffix:
            filepath = f"{filepath}.{fmt}"

        gui_state = collect_gui_state(widgets=self._widgets)
        float_image_data = self._output_layer_float_data()
        if float_image_data is None:
            image_data = runtime.normalized_image_data(np.asarray(output_layer.data)[..., :3])
        else:
            image_data = np.asarray(float_image_data)[..., :3]

        source_color_space, source_cctf_encoding = self._output_layer_render_settings(
            default_color_space=gui_state.simulation.io.output_color_space,
            default_cctf_encoding=True,
        )
        saving_color_space = gui_state.simulation.workflow.saving_color_space
        saving_cctf_encoding = gui_state.simulation.workflow.saving_cctf_encoding
        if source_color_space != saving_color_space:
            image_data = colour.RGB_to_RGB(
                image_data,
                source_color_space,
                saving_color_space,
                apply_cctf_decoding=source_cctf_encoding,
                apply_cctf_encoding=saving_cctf_encoding,
            )
        elif source_cctf_encoding != saving_cctf_encoding:
            image_data = colour.RGB_to_RGB(
                image_data,
                source_color_space,
                saving_color_space,
                apply_cctf_decoding=source_cctf_encoding,
                apply_cctf_encoding=saving_cctf_encoding,
            )

        source_metadata = None

        if self._current_input_path is not None:
            source_metadata = read_image_metadata(self._current_input_path)

        try:
            save_image_oiio(
                filepath,
                image_data,
                color_space=saving_color_space,
                cctf_encoding=saving_cctf_encoding,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Save output', f'Failed to save output image.\n\n{exc}')
            return

        metadata_write_error = None
        try:
            write_image_metadata(
                filepath,
                source_metadata,
                saving_color_space=saving_color_space,
                saving_cctf_encoding=saving_cctf_encoding,
            )
        except Exception as exc:
            metadata_write_error = exc

        if metadata_write_error is not None:
            set_status(
                self._viewer,
                f"Saved output image to {filepath}, but failed to copy metadata: {metadata_write_error}",
            )
        else:
            set_status(self._viewer, f"Saved output image to {filepath}")

    def save_current_as_default(self) -> None:
        persistence_actions.save_current_as_default(
            viewer=self._viewer,
            widgets=self._widgets,
            collect_gui_state_fn=collect_gui_state,
            save_default_gui_state_fn=save_default_gui_state,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def save_current_state_to_file(self) -> None:
        persistence_actions.save_current_state_to_file(
            viewer=self._viewer,
            widgets=self._widgets,
            file_dialog=_DirMemoryDialog('gui_state'),
            collect_gui_state_fn=collect_gui_state,
            save_gui_state_to_path_fn=save_gui_state_to_path,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def load_state_from_file(self) -> None:
        persistence_actions.load_state_from_file(
            viewer=self._viewer,
            widgets=self._widgets,
            file_dialog=_DirMemoryDialog('gui_state'),
            load_gui_state_from_path_fn=load_gui_state_from_path,
            apply_gui_state_fn=apply_gui_state,
            sync_canvas_background_fn=self._sync_canvas_background,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )
        self.record_history_state(immediate=True)

    def restore_factory_default(self) -> None:
        persistence_actions.restore_factory_default(
            viewer=self._viewer,
            widgets=self._widgets,
            project_default_gui_state=PROJECT_DEFAULT_GUI_STATE,
            clear_saved_default_gui_state_fn=clear_saved_default_gui_state,
            apply_gui_state_fn=apply_gui_state,
            sync_canvas_background_fn=self._sync_canvas_background,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )
        self.record_history_state(immediate=True)

    def _flush_auto_save_sidecar(self) -> None:
        if self._suppress_auto_save or self._current_input_path is None:
            return
        if self._auto_save_timer.isActive():
            self._auto_save_timer.stop()
        try:
            state = collect_gui_state(widgets=self._widgets)
            save_sidecar(self._current_input_path, state)
            if self._gallery_model is not None:
                self._gallery_model.notify_item_saved(self._current_input_path)
        except Exception:
            pass

    def on_gallery_current_changed(self, current_idx: int, item: Any | None) -> None:
        if self._switching_gallery_item:
            return

        if item is None:
            self._current_input_path = None
            self._current_input_image = None
            self._current_preview_image = None
            self.show_startup_placeholder()
            self._history_record_timer.stop()
            self._history_manager.reset(self._collect_current_gui_state())
            set_status(self._viewer, "Gallery strip is empty.")
            return

        self._flush_auto_save_sidecar()
        self._switching_gallery_item = True
        self._suppress_auto_save = True
        try:
            target_path = Path(item.path)
            photo_state = load_sidecar(target_path)
            if photo_state is not None:
                apply_gui_state(photo_state, widgets=self._widgets)
            else:
                if item.is_raw:
                    load_raw_section = getattr(self._widgets, 'load_raw', None)
                    if load_raw_section is not None:
                        set_path_fn = getattr(load_raw_section, 'set_path', None)
                        if callable(set_path_fn):
                            set_path_fn(str(target_path))
                current_state = collect_gui_state(widgets=self._widgets)
                save_sidecar(target_path, current_state)
                if self._gallery_model is not None:
                    self._gallery_model.notify_item_saved(target_path)

            if item.is_raw:
                self.load_raw_image(str(target_path))
            else:
                self.load_input_image(str(target_path))
        finally:
            self._suppress_auto_save = False
            self._switching_gallery_item = False

        self._history_record_timer.stop()
        self._history_manager.reset(self._collect_current_gui_state())
        self._schedule_gallery_prefetch(current_idx)

    def _schedule_gallery_prefetch(self, current_idx: int) -> None:
        if self._gallery_model is None or self._gallery_model.count <= 1:
            return

        self._prefetch_generation += 1
        current_gen = self._prefetch_generation

        count = self._gallery_model.count
        candidate_indices = [
            current_idx + 1,
            current_idx - 1,
            current_idx + 2,
            current_idx - 2,
        ]
        valid_indices = [i for i in candidate_indices if 0 <= i < count and i != current_idx]

        gui_state = self._collect_current_gui_state()
        preview_max_size = int(gui_state.gui_only.display.settings.preview_max_size)
        targets: list[tuple[str, bool, GuiState]] = []

        for idx in valid_indices:
            item = self._gallery_model.item_at(idx)
            if item is None:
                continue
            path_str = str(item.path)
            item_state = load_sidecar(path_str)
            if item_state is None:
                item_state = gui_state

            sig = compute_input_signature(item_state, is_raw=item.is_raw)
            if self._gallery_cache.get_input(path_str, sig, preview_max_size) is not None:
                continue

            targets.append((path_str, item.is_raw, item_state))

        if not targets:
            return

        worker = GalleryPrefetchWorker(
            targets=targets,
            cache=self._gallery_cache,
            preview_max_size=preview_max_size,
            load_raw_fn=load_and_process_raw_file,
            load_rgb_fn=load_image_oiio,
            resize_fn=self._resize_for_preview,
            prepare_preview_fn=self._prepare_input_color_preview_image,
            generation=current_gen,
            get_current_generation_fn=lambda: self._prefetch_generation,
        )
        self._thread_pool.start(worker)

    def gallery_remove_current_photo(self) -> None:
        if self._gallery_model is None or self._gallery_model.count == 0:
            return
        self._flush_auto_save_sidecar()

        # If multiple photos are selected, remove all selected photos
        selected_indices = list(self._gallery_model.selected_indices)
        if selected_indices:
            removed_paths = self._gallery_model.remove_items(selected_indices)
            for p in removed_paths:
                self._gallery_cache.invalidate_path(p)
            remaining = self._gallery_model.count
            if remaining > 0:
                set_status(self._viewer, f"Removed {len(removed_paths)} selected photo(s) from filmstrip ({remaining} remaining).")
            else:
                set_status(self._viewer, f"Removed {len(removed_paths)} photo(s). Filmstrip is now empty.")
            return

        # Fallback: remove currently active photo
        curr_item = self._gallery_model.current_item
        curr_name = curr_item.filename if curr_item else "photo"
        curr_path = curr_item.path if curr_item else None
        if self._gallery_model.remove_current_photo():
            if curr_path:
                self._gallery_cache.invalidate_path(curr_path)
            remaining = self._gallery_model.count
            if remaining > 0:
                set_status(self._viewer, f"Removed '{curr_name}' from filmstrip ({remaining} remaining).")
            else:
                set_status(self._viewer, f"Removed '{curr_name}'. Filmstrip is now empty.")

    def gallery_remove_photo_at(self, index: int) -> None:
        if self._gallery_model is None:
            return
        if index in self._gallery_model.selected_indices and self._gallery_model.selected_count > 1:
            self.gallery_remove_current_photo()
        elif index == self._gallery_model.current_index:
            self.gallery_remove_current_photo()
        else:
            item = self._gallery_model.item_at(index)
            if item is not None:
                self._gallery_cache.invalidate_path(item.path)
            self._gallery_model.remove_item(index)

    def gallery_convert_to_dng(self, index: int = -1) -> None:
        if self._gallery_model is None or self._gallery_model.count == 0:
            return

        from spektrafilm.utils.raw_file_processor import convert_raw_to_dng, is_dng_converter_available
        if not is_dng_converter_available():
            QtWidgets.QMessageBox.warning(
                dialog_parent(self._viewer),
                "Adobe DNG Converter Not Found",
                "Adobe DNG Converter is required to convert RAW files to DNG-RAW.\n\n"
                "Please install Adobe DNG Converter or set the SPEKTRAFILM_DNG_CONVERTER environment variable.",
            )
            return

        sel_indices = sorted(self._gallery_model.selected_indices)
        if index in sel_indices and len(sel_indices) > 1:
            target_indices = [
                i for i in sel_indices
                if (it := self._gallery_model.item_at(i)) is not None and it.is_raw and it.extension != ".dng"
            ]
        elif 0 <= index < self._gallery_model.count:
            it = self._gallery_model.item_at(index)
            target_indices = [index] if it is not None and it.is_raw and it.extension != ".dng" else []
        else:
            curr_idx = self._gallery_model.current_index
            it = self._gallery_model.current_item
            target_indices = [curr_idx] if it is not None and it.is_raw and it.extension != ".dng" else []

        if not target_indices:
            return

        converted_count = 0
        last_dng_path = None
        current_idx = self._gallery_model.current_index

        for idx in target_indices:
            item = self._gallery_model.item_at(idx)
            if item is None:
                continue
            orig_path = item.path
            try:
                set_status(self._viewer, f"Converting '{item.filename}' to DNG-RAW...", timeout_ms=0)
                QtWidgets.QApplication.processEvents()
                dng_path = convert_raw_to_dng(orig_path, overwrite=True)
                converted_count += 1
                last_dng_path = dng_path
                self._gallery_cache.invalidate_path(orig_path)
                self._gallery_model.replace_item_path(orig_path, dng_path)

                if idx == current_idx:
                    self.load_raw_image(str(dng_path))
            except Exception as exc:
                logger.error("Failed to convert %s to DNG: %s", item.filename, exc)
                QtWidgets.QMessageBox.critical(
                    dialog_parent(self._viewer),
                    "DNG Conversion Error",
                    f"Failed to convert '{item.filename}' to DNG-RAW:\n\n{exc}",
                )
                break

        if converted_count > 0:
            if converted_count == 1 and last_dng_path is not None:
                set_status(self._viewer, f"Converted to DNG-RAW: {last_dng_path.name}")
            else:
                set_status(self._viewer, f"Converted {converted_count} photos to DNG-RAW.")

    def gallery_open_folder(self) -> None:
        last_dir = load_dialog_dir('gallery_photos')
        folder = QFileDialog.getExistingDirectory(
            dialog_parent(self._viewer),
            'Open Folder of Photos',
            last_dir or '',
        )
        if folder:
            save_dialog_dir('gallery_photos', folder)
            self.load_gallery_folder(folder)

    def load_gallery_folder(self, folder: str | Path) -> int:
        if self._gallery_model is None:
            return 0
        count = self._gallery_model.load_folder(folder)
        set_status(self._viewer, f"Loaded {count} photo(s) from {Path(folder).name}")
        return count

    def gallery_add_files(self) -> None:
        last_dir = load_dialog_dir('gallery_photos')
        paths, _ = QFileDialog.getOpenFileNames(
            dialog_parent(self._viewer),
            'Add Photos to Gallery',
            last_dir or '',
            'Image Files (*.arw *.rw2 *.cr2 *.cr3 *.nef *.dng *.orf *.raf *.jpg *.jpeg *.png *.tif *.tiff);;All Files (*.*)',
        )
        if paths:
            save_dialog_dir('gallery_photos', str(Path(paths[0]).parent))
            self.add_gallery_files(paths)

    def add_gallery_files(self, paths: list[str | Path]) -> int:
        if self._gallery_model is None:
            return 0
        count = self._gallery_model.add_files(paths)
        set_status(self._viewer, f"Added {count} photo(s) to gallery")
        return count

    def gallery_copy_look(self) -> None:
        try:
            state = collect_gui_state(widgets=self._widgets)
            if self._gallery_model is not None:
                self._gallery_model.copy_look(state)
                set_status(self._viewer, "Copied look to clipboard")
        except Exception as exc:
            set_status(self._viewer, f"Failed to copy look: {exc}")

    def gallery_paste_look(self) -> None:
        if self._gallery_model is None:
            return
        pasted_state = self._gallery_model.paste_look()
        if pasted_state is not None:
            self._flush_auto_save_sidecar()
            targets = self._gallery_model.get_target_indices(fallback_to_current=True)
            curr_idx = self._gallery_model.current_index
            for idx in targets:
                item = self._gallery_model.item_at(idx)
                if item is not None:
                    save_sidecar(item.path, pasted_state)
                    self._gallery_cache.invalidate_path(item.path)
                    self._gallery_model.notify_item_saved(item.path)
            if curr_idx in targets or not targets:
                self._suppress_auto_save = True
                try:
                    apply_gui_state(pasted_state, widgets=self._widgets)
                finally:
                    self._suppress_auto_save = False

                # If active photo is a RAW file, reload and re-process the raw file
                # so WB/temperature/tint/lens correction/chroma denoise take effect immediately
                is_raw = (
                    (self._gallery_model.current_item is not None and self._gallery_model.current_item.is_raw)
                    or (self._current_input_path is not None and Path(self._current_input_path).suffix.lower() in RAW_EXTENSIONS)
                )
                if is_raw and self._current_input_path:
                    self.load_raw_image(self._current_input_path)
                else:
                    self.request_auto_preview()
                self.record_history_state(immediate=True)

            target_desc = f"{len(targets)} selected photo(s)" if len(targets) > 1 else "active photo"
            set_status(self._viewer, f"Pasted look to {target_desc}")
        else:
            set_status(self._viewer, "No look in clipboard to paste")

    def gallery_open_sync_dialog(self) -> None:
        if self._gallery_model is None or self._gallery_model.count == 0 or self._gallery_model.current_item is None:
            set_status(self._viewer, "No active photo in gallery to copy parameters from.")
            return

        self._flush_auto_save_sidecar()
        active_state = collect_gui_state(widgets=self._widgets)
        active_filename = self._gallery_model.current_item.filename
        selected_count = self._gallery_model.selected_count
        total_count = self._gallery_model.count

        dialog = SyncParameterDialog(
            dialog_parent(self._viewer),
            active_state=active_state,
            active_filename=active_filename,
            selected_count=selected_count,
            total_count=total_count,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        params_dict = dialog.get_checked_params_dict()
        if not params_dict:
            set_status(self._viewer, "No parameters selected to copy or sync.")
            return

        if dialog.was_copy_only():
            self._gallery_model.copy_params_dict(params_dict)
            if len(params_dict) == 1:
                p_id = next(iter(params_dict.keys()))
                p_def = get_param_def(p_id)
                name = p_def.name if p_def else p_id
                set_status(self._viewer, f"Copied parameter '{name}' to clipboard")
            else:
                set_status(self._viewer, f"Copied {len(params_dict)} parameters to clipboard")
            return

        # Paste to target photos
        if dialog.get_apply_scope() == "all":
            target_indices = list(range(total_count))
        else:
            target_indices = self._gallery_model.get_target_indices(fallback_to_current=True)

        self.gallery_paste_params_dict(params_dict=params_dict, target_indices=target_indices)

    def gallery_copy_single_param(self, param_id: str | None = None) -> None:
        if self._gallery_model is None or self._gallery_model.current_item is None:
            return
        if param_id is None:
            self.gallery_open_sync_dialog()
            return

        p_def = get_param_def(param_id)
        if p_def is None:
            return

        self._flush_auto_save_sidecar()
        active_state = collect_gui_state(widgets=self._widgets)
        val = p_def.extract(active_state)
        self._gallery_model.copy_single_param(param_id, val)
        set_status(self._viewer, f"Copied parameter '{p_def.name}' to clipboard")

    def gallery_copy_params_dict(self, params_dict: dict[str, Any]) -> None:
        if self._gallery_model is None:
            return
        self._gallery_model.copy_params_dict(params_dict)
        set_status(self._viewer, f"Copied {len(params_dict)} parameters to clipboard")

    def gallery_paste_params_dict(
        self,
        params_dict: dict[str, Any] | None = None,
        target_indices: list[int] | None = None,
    ) -> int:
        if self._gallery_model is None:
            return 0

        # If params_dict not specified, retrieve from parameter clipboard
        if params_dict is None:
            params_dict = self._gallery_model.get_params_clipboard_dict()
            if not params_dict:
                self.gallery_open_sync_dialog()
                return 0

        if target_indices is None:
            target_indices = self._gallery_model.get_target_indices(fallback_to_current=True)

        if not target_indices and self._gallery_model.count > 0:
            set_status(self._viewer, "No target photos selected to paste parameters")
            return 0

        self._flush_auto_save_sidecar()
        fallback_state = collect_gui_state(widgets=self._widgets)
        curr_idx = self._gallery_model.current_index
        applied_count = 0

        for idx in target_indices:
            if not (0 <= idx < self._gallery_model.count):
                continue
            item = self._gallery_model.items[idx]
            ok = apply_params_dict_to_file(item.path, params_dict, fallback_state=fallback_state)
            if ok:
                applied_count += 1
                self._gallery_model.notify_item_saved(item.path)

        # If active photo is among the targets (or gallery empty), update GUI controls and request preview
        if curr_idx in target_indices or not target_indices:
            current_state = collect_gui_state(widgets=self._widgets)
            apply_params_dict_to_state(current_state, params_dict)
            self._suppress_auto_save = True
            try:
                apply_gui_state(current_state, widgets=self._widgets)
            finally:
                self._suppress_auto_save = False

            # Check if active photo is a RAW file and any RAW-affecting parameter was changed
            is_raw = (
                (self._gallery_model.current_item is not None and self._gallery_model.current_item.is_raw)
                or (self._current_input_path is not None and Path(self._current_input_path).suffix.lower() in RAW_EXTENSIONS)
            )
            has_raw_affecting = any(pid in RAW_AFFECTING_PARAM_IDS for pid in params_dict)
            if is_raw and has_raw_affecting and self._current_input_path:
                self.load_raw_image(self._current_input_path)
            else:
                self.request_auto_preview()
            self.record_history_state(immediate=True)

        if len(params_dict) == 1:
            p_id = next(iter(params_dict.keys()))
            p_def = get_param_def(p_id)
            desc = p_def.name if p_def else p_id
            set_status(self._viewer, f"Applied '{desc}' to {applied_count} photo(s)")
        else:
            set_status(self._viewer, f"Applied {len(params_dict)} parameters to {applied_count} photo(s)")

        return applied_count

    def gallery_paste_single_param(
        self,
        param_id: str | None = None,
        value: Any = None,
        target_indices: list[int] | None = None,
    ) -> int:
        if param_id is not None and value is not None:
            return self.gallery_paste_params_dict({param_id: value}, target_indices=target_indices)
        return self.gallery_paste_params_dict(None, target_indices=target_indices)

    def gallery_scan_and_save_all(self, scope: str = "all") -> None:
        if self._gallery_model is None or self._gallery_model.count == 0:
            QMessageBox.information(
                dialog_parent(self._viewer),
                "Scan & Save",
                "No photos in gallery to export.",
            )
            return

        self._flush_auto_save_sidecar()

        first_item = self._gallery_model.items[0]
        default_out = first_item.path.parent / "output"
        selected_count = self._gallery_model.selected_count

        if scope == "selected" and selected_count == 0:
            scope = "all"

        dialog = ExportOptionsDialog(
            dialog_parent(self._viewer),
            default_output_dir=default_out,
            total_photos=self._gallery_model.count,
            selected_photos=selected_count,
            default_scope=scope,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        options = dialog.get_options()
        if options is None:
            return

        options.output_dir.mkdir(parents=True, exist_ok=True)
        fallback_state = collect_gui_state(widgets=self._widgets)

        if options.scope == "selected" and selected_count > 0:
            target_indices = sorted(self._gallery_model.selected_indices)
            items_to_export = [self._gallery_model.items[i] for i in target_indices]
        else:
            items_to_export = self._gallery_model.items

        worker = BatchExportWorker(
            items=items_to_export,
            options=options,
            fallback_state=fallback_state,
        )
        progress_dialog = BatchProgressDialog(
            dialog_parent(self._viewer),
            worker=worker,
            total=len(items_to_export),
        )
        self._thread_pool.start(worker)
        progress_dialog.exec_()

    def _preview_input_layer(self) -> NapariImageLayer | None:
        return self._layers.preview_input_layer()

    def _white_border_layer(self) -> NapariImageLayer | None:
        return self._layers.white_border_layer()

    def _set_or_add_output_layer(
        self,
        image: np.ndarray,
        *,
        float_image: np.ndarray,
        output_color_space: str,
        output_cctf_encoding: bool,
        use_display_transform: bool,
        animate: bool = True,
    ) -> None:
        self._layers.set_or_add_output_layer(
            image,
            float_image=float_image,
            output_color_space=output_color_space,
            output_cctf_encoding=output_cctf_encoding,
            use_display_transform=use_display_transform,
            output_interpolation_mode=self._output_interpolation_mode(),
            animate=animate,
        )
        self._sync_histogram(image)

    def _sync_histogram(self, image: np.ndarray | None = None) -> None:
        target_image = image
        if target_image is None:
            out_layer = self._output_layer()
            if out_layer is not None and getattr(out_layer, 'data', None) is not None and getattr(out_layer, 'visible', True):
                target_image = np.asarray(out_layer.data)
            else:
                prev_layer = self._layers.preview_input_layer()
                if prev_layer is not None and getattr(prev_layer, 'data', None) is not None and getattr(prev_layer, 'visible', True):
                    target_image = np.asarray(prev_layer.data)
                elif self._current_preview_image is not None:
                    target_image = self._current_preview_image

        gallery = getattr(self._widgets, 'gallery', None)
        if gallery is not None:
            set_hist_fn = getattr(gallery, 'set_histogram_image', None)
            if callable(set_hist_fn):
                set_hist_fn(target_image)

        chemistry = getattr(self._widgets, 'chemistry', None)
        if chemistry is not None:
            set_curve_hist_fn = getattr(chemistry, 'set_histogram_image', None)
            if callable(set_curve_hist_fn):
                set_curve_hist_fn(target_image)

    def _set_or_add_input_stack(
        self,
        image: np.ndarray,
    ) -> None:
        self._update_preview_cache(
            image,
            home_input_stack=True,
            hide_output=True,
        )

    def _update_preview_cache(
        self,
        image: np.ndarray,
        *,
        home_input_stack: bool,
        hide_output: bool,
    ) -> None:
        state = collect_gui_state(widgets=self._widgets)
        preview_image = self._resize_for_preview(image, max_size=state.gui_only.display.settings.preview_max_size)
        preview_display_image = self._prepare_input_color_preview_image(
            preview_image,
            input_color_space=state.input_image.io.input_color_space,
            apply_cctf_decoding=state.input_image.io.input_cctf_decoding,
        )
        self._current_input_image = image
        self._current_preview_image = preview_image
        self._layers.set_or_add_input_preview_layer(
            preview_display_image,
            watermark_source_size=tuple(int(dimension) for dimension in image.shape[:2]),
            white_padding=state.gui_only.display.white_padding,
            hide_output=hide_output,
            set_active=home_input_stack or self._output_layer() is None,
        )
        if home_input_stack:
            self._home_input_stack()
        if self._output_layer() is None:
            self._sync_histogram(preview_display_image)

    def _sync_white_border(self, *, white_padding: float) -> None:
        self._layers.sync_white_border(white_padding=white_padding)

    def _home_input_stack(self) -> None:
        if self._white_border_layer() is None:
            return
        reset_viewer_camera(self._viewer)
        self._set_active_layer(self._white_border_layer())

    def _ensure_input_loaded(self, *, require_full: bool = False) -> None:
        path = self._current_input_path
        if not path and self._gallery_model is not None and self._gallery_model.current_item is not None:
            path = str(self._gallery_model.current_item.path)
            self._current_input_path = path

        if not path or not os.path.isfile(path):
            return

        gui_state = self._collect_current_gui_state()
        preview_max_size = int(gui_state.gui_only.display.settings.preview_max_size)
        is_raw = Path(path).suffix.lower() in RAW_EXTENSIONS

        sig = compute_input_signature(gui_state, is_raw=is_raw)
        cached = self._gallery_cache.get_input(path, sig, preview_max_size)

        if cached is not None:
            if self._current_preview_image is None:
                self._current_preview_image = cached.preview_image
            if not require_full or cached.full_image is not None:
                if cached.full_image is not None:
                    self._current_input_image = cached.full_image
                return

        if is_raw:
            set_status(self._viewer, "Decoding full RAW for scan...", timeout_ms=0)
            lens_info: dict[str, str] = {}
            try:
                image = load_and_process_raw_file(
                    path,
                    white_balance=gui_state.gui_only.load_raw.white_balance,
                    temperature=gui_state.gui_only.load_raw.temperature,
                    tint=gui_state.gui_only.load_raw.tint,
                    lens_correction=gui_state.gui_only.load_raw.lens_correction,
                    chroma_denoise=gui_state.gui_only.load_raw.chroma_denoise,
                    chroma_denoise_strength=gui_state.gui_only.load_raw.chroma_denoise_strength,
                    raw_exposure_ev=getattr(gui_state.gui_only.load_raw, "raw_exposure_ev", 0.0),
                    highlight_recovery=getattr(gui_state.gui_only.load_raw, "highlight_recovery", 1.0),
                    highlight_mode=getattr(gui_state.gui_only.load_raw, "highlight_mode", "reconstruct"),
                    output_colorspace=gui_state.input_image.io.input_color_space,
                    output_cctf_encoding=gui_state.input_image.io.input_cctf_decoding,
                    lens_info_out=lens_info,
                )
            except Exception as exc:
                logger.error("Failed to reload RAW image %s: %s", path, exc)
                return
        else:
            try:
                image = load_image_oiio(path)[..., :3]
            except Exception as exc:
                logger.error("Failed to reload image %s: %s", path, exc)
                return

        self._current_input_image = image
        if self._current_preview_image is None:
            self._current_preview_image = self._resize_for_preview(image, max_size=preview_max_size)

        if cached is not None:
            cached.full_image = image
        else:
            try:
                canonical = str(Path(path).resolve())
                mtime = os.path.getmtime(canonical)
                preview_img = self._current_preview_image
                preview_disp = self._prepare_input_color_preview_image(
                    preview_img,
                    input_color_space=gui_state.input_image.io.input_color_space,
                    apply_cctf_decoding=gui_state.input_image.io.input_cctf_decoding,
                )
                self._gallery_cache.put_input(
                    CachedInputImage(
                        path=canonical,
                        mtime=mtime,
                        signature=sig,
                        full_image=image,
                        lens_info=lens_info if is_raw else {},
                        preview_image=preview_img,
                        preview_display_image=preview_disp,
                        preview_max_size=preview_max_size,
                    )
                )
            except Exception:
                pass

    def _simulation_input_image(self, *, source_layer_name: str) -> np.ndarray | None:
        if source_layer_name == INPUT_PREVIEW_LAYER_NAME:
            if self._current_preview_image is None:
                self._ensure_input_loaded(require_full=False)
            return self._current_preview_image
        if source_layer_name == INPUT_LAYER_NAME:
            if self._current_input_image is None:
                self._ensure_input_loaded(require_full=True)
            return self._current_input_image
        return None

    def _auto_preview_enabled(self) -> bool:
        simulation_section = getattr(self._widgets, 'simulation', None)
        auto_preview_value = getattr(simulation_section, 'auto_preview_value', None)
        return bool(auto_preview_value()) if callable(auto_preview_value) else False

    def _run_scheduled_auto_preview(self) -> None:
        self._auto_preview_scheduled = False
        if not self._auto_preview_enabled() or self._current_preview_image is None:
            self._pending_auto_preview = False
            return
        if self._active_simulation_worker is not None:
            self._pending_auto_preview = True
            return
        self._run_preview(report_status=False)

    def _replay_pending_auto_preview(self) -> None:
        if not self._pending_auto_preview:
            return
        self._pending_auto_preview = False
        self.request_auto_preview()

    def _output_layer(self) -> NapariImageLayer | None:
        return self._layers.output_layer()

    def _set_active_layer(self, layer: NapariImageLayer | None) -> None:
        self._layers.set_active_layer(layer)

    def _output_layer_float_data(self) -> np.ndarray | None:
        output_layer = self._output_layer()
        if output_layer is None:
            return None
        float_data = output_layer.metadata.get(OUTPUT_FLOAT_DATA_KEY)
        if float_data is None:
            return None
        return np.asarray(float_data)

    def _output_layer_render_settings(
        self,
        *,
        default_color_space: str,
        default_cctf_encoding: bool,
    ) -> tuple[str, bool]:
        output_layer = self._output_layer()
        if output_layer is None:
            return default_color_space, default_cctf_encoding
        color_space = output_layer.metadata.get(OUTPUT_COLOR_SPACE_KEY, default_color_space)
        cctf_encoding = output_layer.metadata.get(OUTPUT_CCTF_ENCODING_KEY, default_cctf_encoding)
        return str(color_space), bool(cctf_encoding)

    def _output_interpolation_mode(self) -> str:
        display_section = getattr(self._widgets, 'display', None)
        editor = getattr(display_section, 'output_interpolation', None)
        value = getattr(editor, 'value', None)
        if isinstance(value, str) and value:
            return value
        current_text = getattr(editor, 'currentText', None)
        if callable(current_text):
            text = current_text()
            if isinstance(text, str) and text:
                return text
        return 'spline36'

    @staticmethod
    def _resize_for_preview(image_data: np.ndarray, *, max_size: int) -> np.ndarray:
        return resize_for_preview(image_data, max_size)

    @staticmethod
    def _prepare_input_color_preview_image(
        image_data: np.ndarray,
        *,
        input_color_space: str,
        apply_cctf_decoding: bool,
    ) -> np.ndarray:
        return runtime.prepare_input_color_preview_image(
            image_data,
            input_color_space=input_color_space,
            apply_cctf_decoding=apply_cctf_decoding,
            colour_module=colour,
        )

    @staticmethod
    def _prepare_output_display_image(
        image_data: np.ndarray,
        *,
        output_color_space: str,
        use_display_transform: bool,
        padding_pixels: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        return runtime.prepare_output_display_image(
            image_data,
            output_color_space=output_color_space,
            use_display_transform=use_display_transform,
            padding_pixels=padding_pixels,
            imagecms_module=ImageCms,
            colour_module=colour,
            pil_image_module=PILImage,
        )

    def _process_image_with_runtime(self, image_data: np.ndarray, params) -> np.ndarray:
        apply_stocks_specifics = (
            self._runtime_simulator is None
            or self._next_runtime_digest_applies_stock_specifics
        )
        digested_params = digest_params(
            params,
            apply_stocks_specifics=apply_stocks_specifics,
        )
        try:
            if self._runtime_simulator is None:
                self._runtime_simulator = runtime_simulator(digested_params)
            else:
                self._runtime_simulator.update_params(digested_params)
            self._next_runtime_digest_applies_stock_specifics = False
            return self._runtime_simulator.process(image_data)
        except Exception:
            self._runtime_simulator = None
            raise

    def _set_display_transform_checked(self, enabled: bool) -> None:
        display_section = getattr(self._widgets, 'display', None)
        toggle = getattr(display_section, 'use_display_transform', None)
        if toggle is None:
            return

        block_signals = getattr(toggle, 'blockSignals', None)
        set_checked = getattr(toggle, 'setChecked', None)
        if not callable(set_checked):
            return

        previous_block_state = None
        if callable(block_signals):
            previous_block_state = block_signals(True)
        try:
            set_checked(enabled)
        finally:
            if callable(block_signals):
                block_signals(bool(previous_block_state))

    def _sync_canvas_background(self) -> None:
        display_section = getattr(self._widgets, 'display', None)
        toggle = getattr(display_section, 'gray_18_canvas', None)
        is_checked = getattr(toggle, 'isChecked', None)
        self.set_gray_18_canvas_enabled(bool(is_checked()) if callable(is_checked) else False)

    def _execute_simulation_request(self, request: SimulationRequest) -> SimulationResult:
        return runtime.execute_simulation_request(
            request,
            run_simulation_fn=self._process_image_with_runtime,
            prepare_output_display_image_fn=self._prepare_output_display_image,
        )

    @staticmethod
    def _configure_simulation_params(params, *, source_layer_name: str):
        settings = getattr(params, 'settings', None)
        if settings is not None and hasattr(settings, 'preview_mode'):
            settings.preview_mode = source_layer_name == INPUT_PREVIEW_LAYER_NAME
        return params

    def _start_simulation(self, *, source_layer_name: str, mode_label: str, report_status: bool = True) -> None:
        if self._active_simulation_worker is not None:
            set_status(self._viewer, 'Simulation already running')
            return

        image_data = self._simulation_input_image(source_layer_name=source_layer_name)
        if image_data is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Load an input image before running the simulation.')
            return

        state = collect_gui_state(widgets=self._widgets)
        self._sync_white_border(white_padding=state.gui_only.display.white_padding)
        params = self._configure_simulation_params(
            build_params_from_state(state),
            source_layer_name=source_layer_name,
        )

        precision = getattr(getattr(params, "settings", None), "precision", "float32")
        dtype = np.float32 if precision == "float32" else np.float64
        image = np.asarray(image_data, dtype=dtype)
        request = SimulationRequest(
            mode_label=mode_label,
            image=image,
            params=params,
            output_color_space=state.simulation.io.output_color_space,
            use_display_transform=state.gui_only.display.use_display_transform,
        )

        worker = runtime.SimulationWorker(request, execute_request=self._execute_simulation_request)
        worker.signals.finished.connect(self._on_simulation_finished)
        worker.signals.failed.connect(self._on_simulation_failed)
        self._active_simulation_worker = worker
        self._active_simulation_label = mode_label
        self._active_simulation_reports_status = report_status
        self._set_simulation_controls_enabled(False)
        if report_status:
            set_status(self._viewer, f'Computing {mode_label.lower()}...', timeout_ms=0)
        self._thread_pool.start(worker)

    def _on_simulation_finished(self, result: SimulationResult) -> None:
        report_status = self._active_simulation_reports_status
        self._active_simulation_worker = None
        self._active_simulation_label = None
        self._active_simulation_reports_status = True
        self._set_simulation_controls_enabled(True)
        is_interactive_preview = (result.mode_label == 'Preview')
        self._set_or_add_output_layer(
            result.display_image,
            float_image=result.float_image,
            output_color_space=result.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=result.use_display_transform,
            animate=not is_interactive_preview,
        )
        if is_interactive_preview and self._current_input_path:
            try:
                gui_state = self._collect_current_gui_state()
                preview_max_size = int(gui_state.gui_only.display.settings.preview_max_size)
                is_raw = self.is_current_image_raw()
                sim_sig = compute_simulation_signature(gui_state, preview_max_size=preview_max_size, is_raw=is_raw)
                self._gallery_cache.put_simulation(self._current_input_path, sim_sig, result)
            except Exception:
                pass
        if report_status:
            set_status(self._viewer, f'{result.mode_label} completed. {result.status_message}')
        self._replay_pending_auto_preview()

    def _on_simulation_failed(self, message: str) -> None:
        self._active_simulation_worker = None
        mode_label = self._active_simulation_label or 'Simulation'
        self._active_simulation_label = None
        self._active_simulation_reports_status = True
        self._set_simulation_controls_enabled(True)
        QMessageBox.critical(dialog_parent(self._viewer), 'Run simulation', f'Simulation failed.\n\n{message}')
        set_status(self._viewer, f'{mode_label} failed')
        self._replay_pending_auto_preview()

    def _set_simulation_controls_enabled(self, enabled: bool) -> None:
        simulation_section = getattr(self._widgets, 'simulation', None)
        if simulation_section is None:
            return
        for button_name in ('preview_button', 'scan_button', 'save_button'):
            button = getattr(simulation_section, button_name, None)
            set_enabled = getattr(button, 'setEnabled', None)
            if callable(set_enabled):
                set_enabled(enabled)

    def _run_simulation(self, *, source_layer_name: str) -> None:
        image_data = self._simulation_input_image(source_layer_name=source_layer_name)
        if image_data is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Load an input image before running the simulation.')
            return

        state = collect_gui_state(widgets=self._widgets)
        self._sync_white_border(white_padding=state.gui_only.display.white_padding)
        params = self._configure_simulation_params(
            build_params_from_state(state),
            source_layer_name=source_layer_name,
        )

        image = np.double(image_data)
        scan = self._process_image_with_runtime(image, params)
        scan_display, display_status = self._prepare_output_display_image(
            scan,
            output_color_space=state.simulation.io.output_color_space,
            use_display_transform=state.gui_only.display.use_display_transform,
        )
        self._set_or_add_output_layer(
            scan_display,
            float_image=scan,
            output_color_space=state.simulation.io.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=state.gui_only.display.use_display_transform,
        )
        set_status(self._viewer, display_status)
