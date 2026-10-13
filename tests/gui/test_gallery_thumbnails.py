from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.gallery_thumbnails import (
    DEFAULT_MAX_THUMB_SIZE,
    DEFAULT_THUMB_HEIGHT,
    DEFAULT_THUMB_WIDTH,
    THUMBNAILS_DIR_NAME,
    ThumbnailService,
    extract_thumbnail_qimage,
    get_disk_thumbnail_path,
)

_q_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_extract_thumbnail_qimage_from_rgb(tmp_path: Path):
    img_path = tmp_path / "test.png"
    qimg = QtGui.QImage(200, 150, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(200, 50, 80))
    assert qimg.save(str(img_path))

    thumb = extract_thumbnail_qimage(img_path, target_width=98, target_height=62)
    assert thumb is not None
    assert not thumb.isNull()
    assert thumb.width() <= 98
    assert thumb.height() <= 62


def test_extract_thumbnail_nonexistent():
    assert extract_thumbnail_qimage("nonexistent_photo_123.jpg") is None


def test_disk_thumbnail_caching_and_folder_creation(tmp_path: Path):
    img_path = tmp_path / "big_photo.jpg"
    qimg = QtGui.QImage(800, 600, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(120, 180, 240))
    assert qimg.save(str(img_path))

    thumb_dir = tmp_path / THUMBNAILS_DIR_NAME
    assert not thumb_dir.exists()

    # Extract thumbnail
    thumb = extract_thumbnail_qimage(img_path, target_width=124, target_height=84)
    assert thumb is not None

    # .thumbnails directory should have been created on disk
    assert thumb_dir.is_dir()
    disk_file = get_disk_thumbnail_path(img_path)
    assert disk_file.is_file()

    # The cached file on disk should be a valid JPEG with max size <= 320
    disk_img = QtGui.QImage(str(disk_file))
    assert not disk_img.isNull()
    assert max(disk_img.width(), disk_img.height()) <= DEFAULT_MAX_THUMB_SIZE

    # Subsequent extraction should read cleanly from disk cache
    thumb2 = extract_thumbnail_qimage(img_path, target_width=124, target_height=84)
    assert thumb2 is not None
    assert thumb2.width() <= 124


def test_vertical_photo_preserves_portrait_aspect(tmp_path: Path):
    # Create a vertical (portrait) image: width 600, height 900
    img_path = tmp_path / "vertical_photo.png"
    qimg = QtGui.QImage(600, 900, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(90, 200, 120))
    assert qimg.save(str(img_path))

    thumb = extract_thumbnail_qimage(img_path, target_width=124, target_height=84)
    assert thumb is not None
    assert not thumb.isNull()
    # Height must be strictly greater than width for vertical photos!
    assert thumb.height() > thumb.width()


def test_thumbnail_service_disk_cache_fast_hit(tmp_path: Path):
    service = ThumbnailService.instance()
    service.clear_cache()

    img_path = tmp_path / "portrait_cached.jpg"
    qimg = QtGui.QImage(400, 600, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(50, 100, 150))
    qimg.save(str(img_path))

    # Pre-generate disk thumbnail in .thumbnails
    extract_thumbnail_qimage(img_path, target_width=124, target_height=84)
    assert get_disk_thumbnail_path(img_path).is_file()

    # Service should return QPixmap immediately from disk without async wait
    pixmap = service.get_thumbnail(img_path, target_width=124, target_height=84)
    assert pixmap is not None
    assert not pixmap.isNull()
    assert pixmap.height() > pixmap.width()


def test_thumbnail_service_progress_signals():
    service = ThumbnailService.instance()
    service.clear_cache()

    progress_events = []
    completed_events = []
    service.progress_changed.connect(lambda c, t, p: progress_events.append((c, t, p)))
    service.all_completed.connect(lambda: completed_events.append(True))

    # Manually simulate progress tracking
    service._total_batch = 4
    service._completed_batch = 0

    service._step_progress()
    assert len(progress_events) == 1
    assert progress_events[-1] == (1, 4, 25)

    service._step_progress()
    service._step_progress()
    assert progress_events[-1] == (3, 4, 75)

    service._step_progress()
    assert progress_events[-1] == (4, 4, 100)
    assert len(completed_events) == 1
    assert service._total_batch == 0
    assert service._completed_batch == 0


def test_stale_landscape_thumbnail_for_portrait_photo_is_regenerated(tmp_path: Path):
    from PIL import Image

    # Create a portrait photo with EXIF orientation 6 (stored as 300x200 sensor pixels)
    img_path = tmp_path / "portrait_exif.jpg"
    im = Image.new("RGB", (300, 200), color="blue")
    exif = im.getexif()
    exif[0x0112] = 6  # 90 CW (portrait)
    im.save(str(img_path), exif=exif)

    # Intentionally write a BAD STALE LANDSCAPE thumbnail into .thumbnails/
    disk_file = get_disk_thumbnail_path(img_path)
    disk_file.parent.mkdir(parents=True, exist_ok=True)
    bad_thumb = QtGui.QImage(124, 84, QtGui.QImage.Format_RGB32)
    bad_thumb.fill(QtGui.QColor(255, 0, 0))
    bad_thumb.save(str(disk_file))
    assert disk_file.is_file()
    # Confirm it is landscape
    assert QtGui.QImage(str(disk_file)).width() > QtGui.QImage(str(disk_file)).height()

    # Calling extract_thumbnail_qimage should detect the orientation mismatch,
    # reject the stale landscape thumbnail, and regenerate a proper portrait thumbnail!
    thumb = extract_thumbnail_qimage(img_path, target_width=124, target_height=84)
    assert thumb is not None
    assert thumb.height() > thumb.width()

    # The file on disk should also have been updated to portrait!
    disk_img_new = QtGui.QImage(str(disk_file))
    assert disk_img_new.height() > disk_img_new.width()
