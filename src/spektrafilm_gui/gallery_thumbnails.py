from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QObject, QRunnable, Qt, Signal

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = frozenset({
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".nef",
    ".orf",
    ".raf",
    ".rw2",
})

DEFAULT_MAX_THUMB_SIZE = 640
THUMBNAILS_DIR_NAME = ".thumbnails"
DEFAULT_THUMB_WIDTH = 124
DEFAULT_THUMB_HEIGHT = 84


def get_disk_thumbnail_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    return p.parent / THUMBNAILS_DIR_NAME / f"{p.name}.jpg"


def _rotate_qimage(qimg: QtGui.QImage, degrees: int) -> QtGui.QImage:
    if degrees % 360 == 0:
        return qimg
    transform = QtGui.QTransform()
    transform.rotate(degrees)
    return qimg.transformed(transform, Qt.SmoothTransformation)


def get_expected_orientation(path: str | Path) -> tuple[int, int]:
    """Return (raw_flip, exif_orientation) for a given image file.
    
    raw_flip: LibRaw flip (0: none, 3: 180, 5: 270 CW, 6: 90 CW)
    exif_orientation: EXIF tag (1: normal, 3: 180, 6: 90 CW, 8: 270 CW)
    """
    raw_flip = 0
    exif_orient = 0
    p = Path(path)
    ext = p.suffix.lower()

    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
            with rawpy.imread(str(p)) as raw:
                raw_flip = int(getattr(raw.sizes, "flip", 0))
        except Exception:
            pass

    try:
        import exiv2
        from spektrafilm.utils.raw_file_processor import _open_exiv2_image
        img = _open_exiv2_image(p)
        img.readMetadata()
        exif = img.exifData()
        if exif is not None:
            datum = exiv2.orientation(exif)
            if datum is not None:
                exif_orient = int(datum.toInt64())
    except Exception:
        pass

    return raw_flip, exif_orient


def extract_thumbnail_qimage(
    path: str | Path,
    target_width: int = DEFAULT_THUMB_WIDTH,
    target_height: int = DEFAULT_THUMB_HEIGHT,
    max_cache_size: int = DEFAULT_MAX_THUMB_SIZE,
) -> QtGui.QImage | None:
    """Extract a fast thumbnail QImage from a RAW or RGB file.
    
    1. Checks for a cached 320px thumbnail in the local `.thumbnails/` folder.
    2. For RAW files, extracts camera-embedded JPEG thumbnail or falls back to half-size demosaic,
       correcting orientation so vertical photos appear vertical.
    3. For RGB files, loads scaled image via QImageReader with EXIF auto-transform.
    4. Saves the resulting thumbnail to `.thumbnails/<filename>.jpg` for instant future loading.
    5. Returns QImage scaled to (target_width, target_height) keeping aspect ratio.
    """
    p = Path(path)
    if not p.is_file():
        return None

    raw_flip, exif_orient = get_expected_orientation(p)
    should_be_portrait = (raw_flip in (5, 6)) or (exif_orient in (6, 8))

    disk_thumb = get_disk_thumbnail_path(p)

    # 1. Check disk cache in .thumbnails folder
    if disk_thumb.is_file():
        try:
            if disk_thumb.stat().st_mtime >= p.stat().st_mtime:
                cached_img = QtGui.QImage(str(disk_thumb))
                if not cached_img.isNull():
                    # Invalidate stale / wrong orientation disk cache
                    if should_be_portrait and cached_img.width() > cached_img.height():
                        logger.debug("Disk cache %s is landscape but image is portrait, regenerating", disk_thumb)
                    else:
                        if target_width and target_height and (cached_img.width() > target_width or cached_img.height() > target_height):
                            return cached_img.scaled(
                                target_width,
                                target_height,
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation,
                            )
                        return cached_img
        except Exception as exc:
            logger.debug("Failed reading disk thumbnail %s: %s", disk_thumb, exc)

    ext = p.suffix.lower()
    qimg: QtGui.QImage | None = None

    # 2. RAW files: ultra-fast embedded thumbnail extraction + orientation correction
    if ext in RAW_EXTENSIONS:
        target_p = p
        if ext != ".dng":
            for dng_name in (f"{p.stem}.dng", f"{p.stem}.DNG"):
                cand = p.parent / dng_name
                if cand.is_file() and cand.stat().st_size > 0:
                    target_p = cand
                    break
        try:
            import rawpy
            with rawpy.imread(str(target_p)) as raw:
                # Attempt embedded thumbnail
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        buf = QtCore.QBuffer()
                        buf.setData(QtCore.QByteArray(thumb.data))
                        buf.open(QtCore.QIODevice.ReadOnly)
                        reader = QtGui.QImageReader(buf)
                        reader.setAutoTransform(True)
                        loaded = reader.read()
                        if not loaded.isNull():
                            qimg = loaded
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        h, w, c = thumb.data.shape
                        qimg = QtGui.QImage(
                            thumb.data.data,
                            w,
                            h,
                            c * w,
                            QtGui.QImage.Format_RGB888,
                        ).copy()
                except Exception:
                    pass

                # Fallback to fast half-size demosaicing
                if qimg is None or qimg.isNull():
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    h, w, c = rgb.shape
                    qimg = QtGui.QImage(
                        rgb.data,
                        w,
                        h,
                        c * w,
                        QtGui.QImage.Format_RGB888,
                    ).copy()
        except Exception as exc:
            logger.debug("Failed rawpy thumbnail extraction for %s: %s", p.name, exc)

    # 3. RGB files: fast scaled load via QImageReader (with auto-transform for EXIF orientation)
    if qimg is None or qimg.isNull():
        try:
            reader = QtGui.QImageReader(str(p))
            reader.setAutoTransform(True)
            loaded = reader.read()
            if not loaded.isNull():
                qimg = loaded
        except Exception as exc:
            logger.debug("QImageReader failed for %s: %s", p.name, exc)

    # 4. Fallback via OIIO if available
    if qimg is None or qimg.isNull():
        try:
            from spektrafilm.utils.io import load_image_oiio
            data = load_image_oiio(str(p))[..., :3]
            data = np.clip(data * 255.0, 0, 255).astype(np.uint8)
            h, w, c = data.shape
            qimg = QtGui.QImage(
                data.data,
                w,
                h,
                c * w,
                QtGui.QImage.Format_RGB888,
            ).copy()
        except Exception as exc:
            logger.debug("OIIO fallback failed for %s: %s", p.name, exc)

    if qimg is None or qimg.isNull():
        return None

    # Ensure vertical photos appear vertical:
    if should_be_portrait and qimg.width() > qimg.height():
        rot = 90 if (raw_flip == 6 or exif_orient == 6) else 270
        qimg = _rotate_qimage(qimg, rot)
    elif raw_flip == 3 or exif_orient == 3:
        qimg = _rotate_qimage(qimg, 180)

    # 5. Scale to max_cache_size (e.g. 320px) and save to .thumbnails on disk
    cache_qimg = qimg
    if cache_qimg.width() > max_cache_size or cache_qimg.height() > max_cache_size:
        cache_qimg = cache_qimg.scaled(
            max_cache_size,
            max_cache_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    try:
        disk_thumb.parent.mkdir(parents=True, exist_ok=True)
        cache_qimg.save(str(disk_thumb), "JPEG", 85)
    except Exception as exc:
        logger.debug("Could not save disk thumbnail %s: %s", disk_thumb, exc)

    # 6. Return scaled to requested target bounds
    if target_width and target_height and (cache_qimg.width() > target_width or cache_qimg.height() > target_height):
        return cache_qimg.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    return cache_qimg


class _ThumbnailSignals(QObject):
    ready = Signal(str, QtGui.QImage)  # (canon_path, qimage)
    failed = Signal(str)  # (canon_path)


class _ThumbnailWorker(QRunnable):
    def __init__(
        self,
        path: Path,
        target_width: int,
        target_height: int,
        signals: _ThumbnailSignals,
    ) -> None:
        super().__init__()
        self.path = path
        self.target_width = target_width
        self.target_height = target_height
        self.signals = signals

    def run(self) -> None:
        canon = str(self.path.resolve()).lower()
        try:
            qimg = extract_thumbnail_qimage(
                self.path,
                target_width=self.target_width,
                target_height=self.target_height,
            )
            if qimg is not None and not qimg.isNull():
                self.signals.ready.emit(canon, qimg)
            else:
                self.signals.failed.emit(canon)
        except Exception as exc:
            logger.debug("Thumbnail worker error for %s: %s", self.path, exc)
            self.signals.failed.emit(canon)


class ThumbnailService(QObject):
    """Thread-safe background service that generates and caches photo thumbnails."""

    thumbnail_ready = Signal(str, QtGui.QPixmap)  # (canon_path, pixmap)
    progress_changed = Signal(int, int, int)  # (completed, total, percentage)
    all_completed = Signal()

    _instance: ThumbnailService | None = None

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cache: dict[str, QtGui.QPixmap] = {}
        self._pending: set[str] = set()
        self._total_batch: int = 0
        self._completed_batch: int = 0
        self._pool = QtCore.QThreadPool()
        self._pool.setMaxThreadCount(2)
        self._signals = _ThumbnailSignals()
        self._signals.ready.connect(self._on_qimage_ready)
        self._signals.failed.connect(self._on_worker_failed)

    @classmethod
    def instance(cls) -> ThumbnailService:
        if cls._instance is None:
            cls._instance = ThumbnailService()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        if cls._instance is not None:
            cls._instance.clear_cache()
            cls._instance = None

    def clear_cache(self) -> None:
        self._cache.clear()
        self._pending.clear()
        self._total_batch = 0
        self._completed_batch = 0

    def get_thumbnail(
        self,
        path: str | Path,
        target_width: int = DEFAULT_THUMB_WIDTH,
        target_height: int = DEFAULT_THUMB_HEIGHT,
    ) -> QtGui.QPixmap | None:
        """Return the cached QPixmap thumbnail if available, or queue background extraction."""
        p = Path(path)
        canon = str(p.resolve()).lower()

        if canon in self._cache:
            return self._cache[canon]

        # Check if 320px thumbnail is already on disk in .thumbnails/
        disk_thumb = get_disk_thumbnail_path(p)
        if disk_thumb.is_file():
            try:
                if disk_thumb.stat().st_mtime >= p.stat().st_mtime:
                    pixmap = QtGui.QPixmap(str(disk_thumb))
                    if not pixmap.isNull():
                        # Invalidate disk thumbnail if it is landscape but photo should be portrait
                        if pixmap.width() > pixmap.height():
                            raw_flip, exif_orient = get_expected_orientation(p)
                            if (raw_flip in (5, 6)) or (exif_orient in (6, 8)):
                                pixmap = None
                        if pixmap is not None:
                            if target_width and target_height and (pixmap.width() > target_width or pixmap.height() > target_height):
                                pixmap = pixmap.scaled(
                                    target_width,
                                    target_height,
                                    Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation,
                                )
                            self._cache[canon] = pixmap
                            return pixmap
            except Exception:
                pass

        if canon not in self._pending:
            self._pending.add(canon)
            self._total_batch += 1
            worker = _ThumbnailWorker(
                path=p,
                target_width=target_width,
                target_height=target_height,
                signals=self._signals,
            )
            self._pool.start(worker)
            pct = int((self._completed_batch / max(1, self._total_batch)) * 100)
            self.progress_changed.emit(self._completed_batch, self._total_batch, pct)

        return None

    def _on_qimage_ready(self, canon_path: str, qimage: QtGui.QImage) -> None:
        self._pending.discard(canon_path)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self._cache[canon_path] = pixmap
        self.thumbnail_ready.emit(canon_path, pixmap)
        self._step_progress()

    def _on_worker_failed(self, canon_path: str) -> None:
        self._pending.discard(canon_path)
        self._step_progress()

    def _step_progress(self) -> None:
        self._completed_batch += 1
        total = max(1, self._total_batch)
        pct = min(100, int((self._completed_batch / total) * 100))
        self.progress_changed.emit(self._completed_batch, self._total_batch, pct)
        if self._completed_batch >= self._total_batch:
            self.all_completed.emit()
            self._total_batch = 0
            self._completed_batch = 0
