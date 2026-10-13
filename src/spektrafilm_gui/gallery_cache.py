"""High-performance multi-tier memory caching and background prefetching for the Spektrafilm gallery filmstrip."""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qtpy import QtCore
from qtpy.QtCore import QObject, QRunnable, Signal

from spektrafilm_gui.controller_runtime import SimulationResult
from spektrafilm_gui.persistence import gui_state_to_dict
from spektrafilm_gui.state import GuiState

logger = logging.getLogger(__name__)


def compute_raw_signature(gui_state: GuiState) -> tuple:
    """Return a hashable signature of all parameters that affect RAW loading/demosaicing."""
    raw = gui_state.gui_only.load_raw
    io = gui_state.input_image.io
    return (
        str(raw.white_balance),
        round(float(raw.temperature), 2),
        round(float(raw.tint), 4),
        bool(raw.lens_correction),
        bool(raw.chroma_denoise),
        round(float(raw.chroma_denoise_strength), 3),
        round(float(getattr(raw, "raw_exposure_ev", 0.0)), 2),
        round(float(getattr(raw, "highlight_recovery", 1.0)), 2),
        str(getattr(raw, "highlight_mode", "reconstruct")),
        str(io.input_color_space),
        bool(io.input_cctf_decoding),
    )


def compute_io_signature(gui_state: GuiState) -> tuple:
    """Return a hashable signature of input image IO decoding parameters."""
    io = gui_state.input_image.io
    return (
        str(io.input_color_space),
        bool(io.input_cctf_decoding),
    )


def compute_input_signature(gui_state: GuiState, is_raw: bool) -> tuple:
    """Return the appropriate input signature depending on whether image is RAW or standard RGB."""
    if is_raw:
        return compute_raw_signature(gui_state)
    return compute_io_signature(gui_state)


def compute_simulation_signature(
    gui_state: GuiState,
    *,
    preview_max_size: int = 640,
    is_raw: bool = False,
) -> str:
    """Return a deterministic string hash of all parameters affecting preview simulation."""
    state_dict = gui_state_to_dict(gui_state)
    state_dict.pop("gui_only", None)
    if is_raw:
        state_dict["__raw_sig__"] = compute_raw_signature(gui_state)
    else:
        state_dict["__io_sig__"] = compute_io_signature(gui_state)
    # Include preview size in signature as spatial scale affects certain filters
    state_dict["__preview_size__"] = int(preview_max_size)
    serialized = json.dumps(state_dict, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CachedInputImage:
    path: str
    mtime: float
    signature: tuple
    full_image: np.ndarray | None
    lens_info: dict[str, str]
    preview_image: np.ndarray
    preview_display_image: np.ndarray
    preview_max_size: int


@dataclass(slots=True)
class CachedSimulationResult:
    path: str
    mtime: float
    param_signature: str
    result: SimulationResult


class GalleryCache:
    """Thread-safe multi-tier LRU cache for gallery images, previews, and simulation results."""

    def __init__(
        self,
        *,
        max_full_images: int = 4,
        max_previews: int = 40,
        max_sim_results: int = 25,
        max_sidecars: int = 100,
    ) -> None:
        self._lock = threading.Lock()
        self._max_full_images = max_full_images
        self._max_previews = max_previews
        self._max_sim_results = max_sim_results
        self._max_sidecars = max_sidecars

        # (canonical_path, mtime, raw_signature) -> CachedInputImage
        self._input_cache: collections.OrderedDict[tuple[str, float, tuple], CachedInputImage] = collections.OrderedDict()
        # (canonical_path, mtime, sim_signature) -> CachedSimulationResult
        self._sim_cache: collections.OrderedDict[tuple[str, float, str], CachedSimulationResult] = collections.OrderedDict()
        # canonical_path -> (mtime, GuiState)
        self._sidecar_cache: collections.OrderedDict[str, tuple[float, GuiState]] = collections.OrderedDict()

    @staticmethod
    def _canonical(path: str | Path) -> str:
        return str(Path(path).resolve())

    @staticmethod
    def _get_mtime(canonical_path: str) -> float:
        try:
            return os.path.getmtime(canonical_path)
        except OSError:
            return 0.0

    # --- Sidecar Cache ---

    def get_sidecar(self, path: str | Path) -> GuiState | None:
        canonical = self._canonical(path)
        mtime = self._get_mtime(canonical)
        with self._lock:
            entry = self._sidecar_cache.get(canonical)
            if entry is not None:
                cached_mtime, state = entry
                if cached_mtime == mtime:
                    self._sidecar_cache.move_to_end(canonical)
                    return state
        return None

    def put_sidecar(self, path: str | Path, state: GuiState) -> None:
        canonical = self._canonical(path)
        mtime = self._get_mtime(canonical)
        with self._lock:
            self._sidecar_cache[canonical] = (mtime, state)
            self._sidecar_cache.move_to_end(canonical)
            while len(self._sidecar_cache) > self._max_sidecars:
                self._sidecar_cache.popitem(last=False)

    def invalidate_sidecar(self, path: str | Path) -> None:
        canonical = self._canonical(path)
        with self._lock:
            self._sidecar_cache.pop(canonical, None)

    # --- Input Image Cache ---

    def get_input(
        self,
        path: str | Path,
        signature: tuple,
        preview_max_size: int,
    ) -> CachedInputImage | None:
        canonical = self._canonical(path)
        mtime = self._get_mtime(canonical)
        key = (canonical, mtime, signature)
        with self._lock:
            entry = self._input_cache.get(key)
            if entry is not None:
                if entry.preview_max_size == preview_max_size:
                    self._input_cache.move_to_end(key)
                    return entry
        return None

    def put_input(self, entry: CachedInputImage) -> None:
        key = (entry.path, entry.mtime, entry.signature)
        with self._lock:
            self._input_cache[key] = entry
            self._input_cache.move_to_end(key)

            # Evict full images exceeding max_full_images while keeping lightweight preview
            full_image_entries = [e for e in self._input_cache.values() if e.full_image is not None]
            if len(full_image_entries) > self._max_full_images:
                # Strip full_image from oldest entries to preserve VRAM/RAM
                for e in list(self._input_cache.values()):
                    if e.full_image is not None and e is not entry:
                        e.full_image = None
                        full_image_entries.pop(0)
                        if len(full_image_entries) <= self._max_full_images:
                            break

            # Evict entire entry if exceeding max_previews
            while len(self._input_cache) > self._max_previews:
                self._input_cache.popitem(last=False)

    # --- Simulation Result Cache ---

    def get_simulation(
        self,
        path: str | Path,
        param_signature: str,
    ) -> SimulationResult | None:
        canonical = self._canonical(path)
        mtime = self._get_mtime(canonical)
        key = (canonical, mtime, param_signature)
        with self._lock:
            entry = self._sim_cache.get(key)
            if entry is not None:
                self._sim_cache.move_to_end(key)
                return entry.result
        return None

    def put_simulation(
        self,
        path: str | Path,
        param_signature: str,
        result: SimulationResult,
    ) -> None:
        canonical = self._canonical(path)
        mtime = self._get_mtime(canonical)
        key = (canonical, mtime, param_signature)
        entry = CachedSimulationResult(
            path=canonical,
            mtime=mtime,
            param_signature=param_signature,
            result=result,
        )
        with self._lock:
            self._sim_cache[key] = entry
            self._sim_cache.move_to_end(key)
            while len(self._sim_cache) > self._max_sim_results:
                self._sim_cache.popitem(last=False)

    def invalidate_path(self, path: str | Path) -> None:
        canonical = self._canonical(path)
        with self._lock:
            self._sidecar_cache.pop(canonical, None)
            input_keys = [k for k in self._input_cache if k[0] == canonical]
            for k in input_keys:
                self._input_cache.pop(k, None)
            sim_keys = [k for k in self._sim_cache if k[0] == canonical]
            for k in sim_keys:
                self._sim_cache.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._input_cache.clear()
            self._sim_cache.clear()
            self._sidecar_cache.clear()


# Global shared cache instance
_GLOBAL_GALLERY_CACHE = GalleryCache()


def get_global_gallery_cache() -> GalleryCache:
    """Return the application-wide singleton GalleryCache."""
    return _GLOBAL_GALLERY_CACHE


# --- Background Prefetch Worker ---

class PrefetchWorkerSignals(QObject):
    prefetched = Signal(str)  # canonical path


class GalleryPrefetchWorker(QRunnable):
    """Background QRunnable task that pre-decodes and caches adjacent gallery photos."""

    def __init__(
        self,
        *,
        targets: list[tuple[str, bool, GuiState]],  # list of (path, is_raw, state)
        cache: GalleryCache,
        preview_max_size: int,
        load_raw_fn: Callable[..., np.ndarray],
        load_rgb_fn: Callable[..., np.ndarray],
        resize_fn: Callable[[np.ndarray, int], np.ndarray],
        prepare_preview_fn: Callable[..., np.ndarray],
        generation: int,
        get_current_generation_fn: Callable[[], int],
    ) -> None:
        super().__init__()
        self._targets = targets
        self._cache = cache
        self._preview_max_size = preview_max_size
        self._load_raw_fn = load_raw_fn
        self._load_rgb_fn = load_rgb_fn
        self._resize_fn = resize_fn
        self._prepare_preview_fn = prepare_preview_fn
        self._generation = generation
        self._get_current_generation_fn = get_current_generation_fn
        self.signals = PrefetchWorkerSignals()

    def run(self) -> None:
        for path_str, is_raw, state in self._targets:
            # Check if prefetch task has been superseded by user navigation
            if self._get_current_generation_fn() != self._generation:
                logger.debug("Prefetch task superseded for %s", path_str)
                return

            try:
                canonical = str(Path(path_str).resolve())
                mtime = os.path.getmtime(canonical)
            except OSError:
                continue

            input_sig = compute_input_signature(state, is_raw)
            if self._cache.get_input(canonical, input_sig, self._preview_max_size) is not None:
                continue

            try:
                lens_info: dict[str, str] = {}
                if is_raw:
                    raw_cfg = state.gui_only.load_raw
                    io_cfg = state.input_image.io
                    image = self._load_raw_fn(
                        canonical,
                        white_balance=raw_cfg.white_balance,
                        temperature=raw_cfg.temperature,
                        tint=raw_cfg.tint,
                        lens_correction=raw_cfg.lens_correction,
                        chroma_denoise=raw_cfg.chroma_denoise,
                        chroma_denoise_strength=raw_cfg.chroma_denoise_strength,
                        raw_exposure_ev=getattr(raw_cfg, "raw_exposure_ev", 0.0),
                        highlight_recovery=getattr(raw_cfg, "highlight_recovery", 1.0),
                        highlight_mode=getattr(raw_cfg, "highlight_mode", "reconstruct"),
                        output_colorspace=io_cfg.input_color_space,
                        output_cctf_encoding=io_cfg.input_cctf_decoding,
                        lens_info_out=lens_info,
                    )
                else:
                    image = self._load_rgb_fn(canonical)[..., :3]

                preview_image = self._resize_fn(image, max_size=self._preview_max_size)
                preview_display = self._prepare_preview_fn(
                    preview_image,
                    input_color_space=state.input_image.io.input_color_space,
                    apply_cctf_decoding=state.input_image.io.input_cctf_decoding,
                )

                cached_entry = CachedInputImage(
                    path=canonical,
                    mtime=mtime,
                    signature=input_sig,
                    full_image=image,
                    lens_info=lens_info,
                    preview_image=preview_image,
                    preview_display_image=preview_display,
                    preview_max_size=self._preview_max_size,
                )
                self._cache.put_input(cached_entry)
                self.signals.prefetched.emit(canonical)
                logger.debug("Successfully prefetched and cached %s", canonical)
            except Exception as exc:
                logger.debug("Prefetch failed for %s: %s", path_str, exc)
