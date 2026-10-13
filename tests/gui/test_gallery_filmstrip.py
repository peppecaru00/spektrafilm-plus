from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.gallery_filmstrip import (
    GalleryFilmstripWidget,
    PhotoCardWidget,
    ThumbnailProgressOverlay,
)
from spektrafilm_gui.gallery_model import GalleryItem, GalleryModel
from spektrafilm_gui.gallery_thumbnails import ThumbnailService

_q_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_photo_card_no_black_box_styling(tmp_path: Path):
    img_path = tmp_path / "test.jpg"
    qimg = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(100, 100, 100))
    qimg.save(str(img_path))

    item = GalleryItem(path=img_path)
    card = PhotoCardWidget(index=0, item=item, is_selected=True)

    # Verify transparent label styling
    assert "transparent" in card.badge_label.styleSheet()
    assert "transparent" in card.edit_indicator.styleSheet()
    assert "transparent" in card.name_label.styleSheet()


def test_gallery_filmstrip_vertical_layout(tmp_path: Path):
    model = GalleryModel()
    strip = GalleryFilmstripWidget(model)

    assert strip.minimumWidth() == 180
    assert strip.maximumWidth() == 800
    assert hasattr(strip, "histogram_widget")
    assert isinstance(strip.scroll_area, QtWidgets.QScrollArea)
    assert strip.scroll_area.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
    assert strip.scroll_area.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff

    # Populate 3 items
    for i in range(3):
        p = tmp_path / f"img_{i:02d}.jpg"
        qimg = QtGui.QImage(20, 20, QtGui.QImage.Format_RGB32)
        qimg.fill(QtGui.QColor(i * 50, 100, 150))
        qimg.save(str(p))

    model.load_folder(tmp_path)
    assert len(strip._cards) == 3
    assert strip.counter_label.text() == "1 / 3"

    # Up/Down arrow keys should navigate vertically
    down_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Down, QtCore.Qt.NoModifier)
    strip.keyPressEvent(down_event)
    assert model.current_index == 1
    assert strip.counter_label.text() == "2 / 3"

    up_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Up, QtCore.Qt.NoModifier)
    strip.keyPressEvent(up_event)
    assert model.current_index == 0
    assert strip.counter_label.text() == "1 / 3"


def test_thumbnail_progress_overlay_display():
    parent = QtWidgets.QWidget()
    parent.show()
    overlay = ThumbnailProgressOverlay(parent)
    assert not overlay.isVisible()

    # Update progress
    overlay.set_progress(completed=5, total=10, percentage=50)
    assert overlay.isVisible()
    assert overlay.percent_label.text() == "50%"
    assert overlay.progress_bar.value() == 50
    assert overlay.detail_label.text() == "5 / 10"


def test_gallery_filmstrip_overlay_integration():
    model = GalleryModel()
    strip = GalleryFilmstripWidget(model)
    strip.resize(154, 500)
    strip.show()

    assert hasattr(strip, "overlay")
    assert not strip.overlay.isVisible()

    # Simulate progress event
    strip._on_thumbnail_progress(completed=12, total=24, percentage=50)
    assert strip.overlay.isVisible()
    assert strip.overlay.percent_label.text() == "50%"

    # Simulate all completed
    strip._on_thumbnail_all_completed()
    assert not strip.overlay.isVisible()


def test_gallery_filmstrip_histogram_integration():
    import numpy as np
    from spektrafilm_gui.histogram_widget import HistogramWidget

    model = GalleryModel()
    strip = GalleryFilmstripWidget(model)

    assert isinstance(strip.histogram_widget, HistogramWidget)
    assert strip.histogram_widget.canvas._has_data is False

    # Provide image
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    strip.set_histogram_image(img)
    assert strip.histogram_widget.canvas._has_data is True
    assert strip.histogram_widget.canvas._norm_r is not None
    assert strip.histogram_widget.canvas._norm_r[128] > 0.0

    # Mode switching
    strip.histogram_widget.set_mode("Luma")
    assert strip.histogram_widget.canvas.mode == "Luma"

    # Clear
    strip.set_histogram_image(None)
    assert strip.histogram_widget.canvas._has_data is False


def test_gallery_filmstrip_thumbnails_adapt_to_panel_width(tmp_path: Path):
    model = GalleryModel()
    strip = GalleryFilmstripWidget(model)

    # Populate 2 items
    for i in range(2):
        p = tmp_path / f"adaptive_{i:02d}.jpg"
        qimg = QtGui.QImage(100, 100, QtGui.QImage.Format_RGB32)
        qimg.fill(QtGui.QColor(i * 100, 100, 200))
        qimg.save(str(p))

    model.load_folder(tmp_path)
    assert len(strip._cards) == 2

    # Sizing at panel width 220
    strip.resize(220, 600)
    strip.show()
    QtWidgets.QApplication.processEvents()

    card_0 = strip._cards[0]
    w_narrow = card_0.width()
    thumb_w_narrow = card_0.thumb_label.width()

    # Resize panel to wider width 400
    strip.resize(400, 600)
    QtWidgets.QApplication.processEvents()

    w_wide = card_0.width()
    thumb_w_wide = card_0.thumb_label.width()

    # Thumbnails and cards must grow with the panel width
    assert w_wide > w_narrow
    assert thumb_w_wide > thumb_w_narrow

    # Resize panel to 500
    strip.resize(500, 600)
    QtWidgets.QApplication.processEvents()
    assert card_0.width() > w_wide
    assert card_0.thumb_label.width() > thumb_w_wide

