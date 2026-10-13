from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import QObject, QRunnable, Qt, Signal

from spektrafilm.runtime.api import Simulator, digest_params
from spektrafilm.utils.io import read_image_metadata, save_image_oiio, write_image_metadata
from spektrafilm.utils.raw_file_processor import load_and_process_raw_file
from spektrafilm_gui.gallery_model import GalleryItem, GalleryModel
from spektrafilm_gui.gallery_sidecar import load_sidecar
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.state import GuiState, clone_gui_state
from spektrafilm_gui.theme_palette import (
    ACCENT_COLOR_TEXT,
    FONT_WEIGHT_SEMIBOLD,
    GRAY_0,
    GRAY_1,
    GRAY_2,
    GRAY_3,
    TEXT_DIM,
    TEXT_HI,
    TEXT_MAIN,
)

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


@dataclass(frozen=True, slots=True)
class ExportOptions:
    output_dir: Path
    output_format: str
    scale_factor: float
    overwrite: bool
    scope: str = "all"


class ExportOptionsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        default_output_dir: Path,
        total_photos: int,
        selected_photos: int = 0,
        default_scope: str = "all",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scan & Save Photos")
        self.setModal(True)
        self.resize(520, 290)
        self._default_output_dir = default_output_dir
        self._total_photos = total_photos
        self._selected_photos = selected_photos
        self._default_scope = default_scope if selected_photos > 0 else "all"
        self._options: ExportOptions | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # Header info
        self.header = QtWidgets.QLabel()
        self.header.setStyleSheet(f"color: {TEXT_HI}; font-weight: {FONT_WEIGHT_SEMIBOLD}; font-size: 13px;")
        layout.addWidget(self.header)

        # Form layout
        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

        # Scope row (All vs Selected)
        scope_container = QtWidgets.QWidget()
        scope_layout = QtWidgets.QHBoxLayout(scope_container)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        scope_layout.setSpacing(16)

        self.scope_all_radio = QtWidgets.QRadioButton(f"All photos ({self._total_photos})")
        sel_label = f"Selected photos ({self._selected_photos})" if self._selected_photos > 0 else "Selected photos (none)"
        self.scope_selected_radio = QtWidgets.QRadioButton(sel_label)
        self.scope_selected_radio.setEnabled(self._selected_photos > 0)

        if self._selected_photos > 0 and self._default_scope == "selected":
            self.scope_selected_radio.setChecked(True)
        else:
            self.scope_all_radio.setChecked(True)

        self.scope_all_radio.toggled.connect(self._update_scope_display)
        self.scope_selected_radio.toggled.connect(self._update_scope_display)

        scope_layout.addWidget(self.scope_all_radio)
        scope_layout.addWidget(self.scope_selected_radio)
        scope_layout.addStretch(1)
        form.addRow("Scope:", scope_container)

        # Output folder row
        folder_container = QtWidgets.QWidget()
        folder_layout = QtWidgets.QHBoxLayout(folder_container)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(6)

        self.output_dir_edit = QtWidgets.QLineEdit(str(self._default_output_dir))
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(self._choose_folder)
        folder_layout.addWidget(self.output_dir_edit, 1)
        folder_layout.addWidget(browse_btn)
        form.addRow("Output Directory:", folder_container)

        # Format row
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["jpg", "png", "tif", "tiff", "webp"])
        form.addRow("Output Format:", self.format_combo)

        # Resolution scale row
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItem("1.0 (Full resolution)", 1.0)
        self.scale_combo.addItem("0.75 (75%)", 0.75)
        self.scale_combo.addItem("0.5 (Half resolution)", 0.5)
        self.scale_combo.addItem("0.4 (40%)", 0.4)
        self.scale_combo.addItem("0.33 (33%)", 0.33)
        self.scale_combo.addItem("0.25 (Quarter resolution)", 0.25)
        form.addRow("Resolution Scale:", self.scale_combo)

        # Overwrite checkbox
        self.overwrite_check = QtWidgets.QCheckBox("Overwrite existing files")
        self.overwrite_check.setChecked(False)
        form.addRow("", self.overwrite_check)

        layout.addLayout(form)
        layout.addStretch(1)

        # Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.setSpacing(8)
        btn_box.addStretch(1)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        self.start_btn = QtWidgets.QPushButton("🚀 Start Scan & Save")
        self.start_btn.setStyleSheet(f"background-color: {ACCENT_COLOR_TEXT}; color: #000; font-weight: {FONT_WEIGHT_SEMIBOLD}; padding: 6px 14px; border-radius: 6px;")
        self.start_btn.clicked.connect(self._on_start)

        btn_box.addWidget(cancel_btn)
        btn_box.addWidget(self.start_btn)
        layout.addLayout(btn_box)

        self._update_scope_display()

    def _update_scope_display(self) -> None:
        is_selected = self.scope_selected_radio.isChecked() and self._selected_photos > 0
        target_count = self._selected_photos if is_selected else self._total_photos
        desc = "selected" if is_selected else "gallery"
        self.header.setText(f"Batch export {target_count} {desc} photo(s) using their tailored settings:")
        if hasattr(self, "start_btn"):
            self.start_btn.setText(f"🚀 Start Scan & Save ({target_count} Photos)")

    def _choose_folder(self) -> None:
        initial = self.output_dir_edit.text().strip() or str(self._default_output_dir)
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", initial)
        if chosen:
            self.output_dir_edit.setText(chosen)

    def _on_start(self) -> None:
        out_dir = Path(self.output_dir_edit.text().strip())
        out_fmt = self.format_combo.currentText().strip()
        scale = float(self.scale_combo.currentData())
        overwrite = self.overwrite_check.isChecked()
        scope = "selected" if (self.scope_selected_radio.isChecked() and self._selected_photos > 0) else "all"

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid Directory", f"Cannot create output folder:\n\n{exc}")
            return

        self._options = ExportOptions(
            output_dir=out_dir,
            output_format=out_fmt,
            scale_factor=scale,
            overwrite=overwrite,
            scope=scope,
        )
        self.accept()

    def get_options(self) -> ExportOptions | None:
        return self._options


class BatchExportSignals(QObject):
    item_progress = Signal(int, int, str, float, float)  # (current, total, filename, duration_sec, eta_sec)
    item_failed = Signal(str, str)                       # (filename, error_message)
    finished = Signal(int, int)                          # (success_count, fail_count)


def _setup_fast_scale(simulator: Simulator) -> bool:
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cnd

        pipeline = getattr(simulator, "_pipeline", None)
        orig_resize = getattr(pipeline, "_resize_service", None) if pipeline else None
        if orig_resize is None:
            return False

        def fast_crop_and_rescale(image: np.ndarray) -> np.ndarray:
            orig_resize.pixel_size_um = (
                orig_resize.film_format_mm * 1000.0 / float(np.max(image.shape[0:2]))
            )
            orig_resize.origin_x_um = 0.0
            orig_resize.origin_y_um = 0.0

            if orig_resize._io.crop:
                from spektrafilm.utils.crop_resize import crop_image
                image = crop_image(
                    image,
                    center=orig_resize._io.crop_center,
                    size=orig_resize._io.crop_size,
                )

            factor = float(orig_resize._io.upscale_factor)
            if factor != 1.0:
                orig_resize.pixel_size_um /= factor
                g = cp.asarray(image, dtype=np.float32)
                if factor < 1.0:
                    s = (1.0 / factor - 1.0) / 2.0
                    g = cnd.gaussian_filter(g, (s, s, 0.0))
                z = cnd.zoom(g, (factor, factor, 1.0), order=3)
                target_dtype = (
                    np.float32
                    if getattr(getattr(pipeline, "settings", None), "precision", "float32") == "float32"
                    else np.float64
                )
                image = cp.asnumpy(z).astype(target_dtype)
            return image

        orig_resize.crop_and_rescale = fast_crop_and_rescale
        return True
    except Exception:
        return False


class BatchExportWorker(QRunnable):
    def __init__(
        self,
        items: list[GalleryItem],
        options: ExportOptions,
        fallback_state: GuiState,
    ) -> None:
        super().__init__()
        self.items = items
        self.options = options
        self.fallback_state = clone_gui_state(fallback_state)
        self.signals = BatchExportSignals()
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        self.options.output_dir.mkdir(parents=True, exist_ok=True)
        total = len(self.items)
        success_count = 0
        fail_count = 0
        durations: list[float] = []

        for idx, item in enumerate(self.items, 1):
            if self._is_cancelled:
                break

            t0 = time.perf_counter()
            raw_path = item.path
            out_name = f"{raw_path.stem}.{self.options.output_format}"
            out_path = self.options.output_dir / out_name

            if not self.options.overwrite and out_path.exists():
                logger.info("Skipping existing file %s", out_name)
                success_count += 1
                item_dur = time.perf_counter() - t0
                durations.append(item_dur)
                avg_dur = sum(durations[-10:]) / len(durations[-10:])
                eta = avg_dur * (total - idx)
                self.signals.item_progress.emit(idx, total, raw_path.name, item_dur, eta)
                continue

            try:
                # 1. Load settings (from .spktrflm sidecar if present, else fallback state)
                photo_state = load_sidecar(raw_path)
                if photo_state is None:
                    photo_state = clone_gui_state(self.fallback_state)

                params = build_params_from_state(photo_state)
                if self.options.scale_factor != 1.0:
                    params.io.upscale_factor = self.options.scale_factor

                # 2. Load input image
                source_meta = read_image_metadata(str(raw_path))
                if item.is_raw:
                    load_raw = photo_state.gui_only.load_raw
                    input_io = photo_state.input_image.io
                    input_cs = input_io.input_color_space
                    input_cctf = input_io.input_cctf_decoding
                    img_data = load_and_process_raw_file(
                        str(raw_path),
                        white_balance=load_raw.white_balance,
                        temperature=load_raw.temperature,
                        tint=load_raw.tint,
                        lens_correction=load_raw.lens_correction,
                        chroma_denoise=load_raw.chroma_denoise,
                        chroma_denoise_strength=load_raw.chroma_denoise_strength,
                        raw_exposure_ev=getattr(load_raw, "raw_exposure_ev", 0.0),
                        highlight_recovery=getattr(load_raw, "highlight_recovery", 1.0),
                        highlight_mode=getattr(load_raw, "highlight_mode", "reconstruct"),
                        output_colorspace=input_cs,
                        output_cctf_encoding=input_cctf,
                    )
                else:
                    from spektrafilm.utils.io import load_image_oiio
                    img_data = load_image_oiio(str(raw_path))[..., :3]

                # 3. Simulate
                digested_params = digest_params(params)
                simulator = Simulator(digested_params)
                if self.options.scale_factor != 1.0:
                    _setup_fast_scale(simulator)
                out_img = simulator.process(img_data)

                # 4. Save image + EXIF metadata safely via temporary file staging
                workflow = photo_state.simulation.workflow
                saving_cs = workflow.saving_color_space if workflow else "sRGB"
                saving_cctf = workflow.saving_cctf_encoding if workflow else True

                with tempfile.NamedTemporaryFile(suffix=f".{self.options.output_format}", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                try:
                    save_image_oiio(
                        str(tmp_path),
                        out_img,
                        color_space=saving_cs,
                        cctf_encoding=saving_cctf,
                    )
                    try:
                        write_image_metadata(
                            str(tmp_path),
                            source_meta,
                            saving_color_space=saving_cs,
                            saving_cctf_encoding=saving_cctf,
                        )
                    except Exception as meta_exc:
                        logger.warning("Failed to copy metadata for %s: %s", raw_path.name, meta_exc)
                    shutil.move(str(tmp_path), str(out_path))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

                del img_data, out_img, source_meta
                from spektrafilm.cuda.device import free_memory, is_cuda_available
                if is_cuda_available():
                    free_memory()

                success_count += 1
            except Exception as exc:
                fail_count += 1
                logger.error("Failed to process %s: %s", raw_path.name, exc, exc_info=True)
                self.signals.item_failed.emit(raw_path.name, str(exc))

            item_dur = time.perf_counter() - t0
            durations.append(item_dur)
            avg_dur = sum(durations[-10:]) / len(durations[-10:])
            eta = avg_dur * (total - idx)
            self.signals.item_progress.emit(idx, total, raw_path.name, item_dur, eta)

        self.signals.finished.emit(success_count, fail_count)


class BatchProgressDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, worker: BatchExportWorker, total: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exporting Photos...")
        self.setModal(True)
        self.resize(460, 180)
        self.worker = worker
        self._total = total
        self._build_ui()

        # Connect signals
        worker.signals.item_progress.connect(self._on_progress)
        worker.signals.item_failed.connect(self._on_failed)
        worker.signals.finished.connect(self._on_finished)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        self.status_label = QtWidgets.QLabel("Preparing batch export...")
        self.status_label.setStyleSheet(f"color: {TEXT_HI}; font-weight: {FONT_WEIGHT_SEMIBOLD};")
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, self._total)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {GRAY_3};
                background-color: {GRAY_1};
                text-align: center;
                color: {TEXT_HI};
                height: 20px;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT_COLOR_TEXT};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        self.detail_label = QtWidgets.QLabel("Initializing...")
        self.detail_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        layout.addWidget(self.detail_label)

        layout.addStretch(1)

        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_box.addWidget(self.cancel_btn)
        layout.addLayout(btn_box)

    def _on_progress(self, current: int, total: int, filename: str, duration: float, eta: float) -> None:
        self.progress_bar.setValue(current)
        self.status_label.setText(f"[{current}/{total}] {filename}")
        self.detail_label.setText(f"Last photo: {duration:4.1f}s | ETA: {format_duration(eta)}")

    def _on_failed(self, filename: str, error: str) -> None:
        logger.warning("Batch export failed for %s: %s", filename, error)

    def _on_cancel(self) -> None:
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling...")
        self.status_label.setText("Cancelling after current photo...")
        self.worker.cancel()

    def _on_finished(self, success: int, fails: int) -> None:
        msg = f"Export completed!\n\nSuccessfully processed: {success} photo(s)"
        if fails > 0:
            msg += f"\nFailed: {fails} photo(s)"
        QtWidgets.QMessageBox.information(self, "Scan & Save All Finished", msg)
        self.accept()
