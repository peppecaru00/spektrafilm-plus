from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QPoint, Qt, Signal

from spektrafilm_gui.gallery_model import GalleryItem, GalleryModel
from spektrafilm_gui.gallery_params import get_param_def
from spektrafilm_gui.gallery_thumbnails import (
    DEFAULT_THUMB_HEIGHT,
    DEFAULT_THUMB_WIDTH,
    ThumbnailService,
)
from spektrafilm_gui.theme_palette import (
    ACCENT_COLOR_TEXT,
    ACCENT_COLOR_TEXT_SECONDARY,
    FONT_WEIGHT_BOLD,
    FONT_WEIGHT_SEMIBOLD,
    GRAY_0,
    GRAY_1,
    GRAY_2,
    GRAY_3,
    GRAY_4,
    TEXT_DIM,
    TEXT_HI,
    TEXT_MAIN,
)
from spektrafilm_gui.theme_styles import CHECKMARK_ICON_PATH
from spektrafilm_gui.histogram_widget import HistogramWidget

logger = logging.getLogger(__name__)

DEFAULT_GALLERY_WIDTH = 250
MIN_GALLERY_WIDTH = 180
MAX_GALLERY_WIDTH = 800
CARD_WIDTH = 194
CARD_HEIGHT = 164
CARD_THUMB_WIDTH = 184
CARD_THUMB_HEIGHT = 122


def compute_card_dimensions(card_width: int) -> tuple[int, int, int]:
    """Compute (thumb_width, thumb_height, card_height) adapting to card_width.

    Maintains photographic 3:2 bounding box while preserving necessary vertical
    space for control badges, filename label, and margins.
    """
    thumb_w = max(80, int(card_width) - 10)
    thumb_h = max(50, round(thumb_w / 1.5))
    card_h = thumb_h + 42
    return thumb_w, thumb_h, card_h


class PhotoCardWidget(QtWidgets.QFrame):
    clicked = Signal(int)
    remove_clicked = Signal(int)
    selection_toggled = Signal(int, bool)
    range_select_requested = Signal(int)
    context_menu_requested = Signal(int, QPoint)

    def __init__(
        self,
        index: int,
        item: GalleryItem,
        is_active: bool = False,
        is_multi_selected: bool = False,
        is_selected: bool | None = None,
        card_width: int | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.index = index
        self.item = item
        if is_selected is not None:
            is_active = is_selected
        self._is_active = is_active
        self._is_multi_selected = is_multi_selected
        self._card_width = card_width if card_width is not None else CARD_WIDTH
        self._pixmap: QtGui.QPixmap | None = None
        self._build_ui()
        self.update_style()
        self._load_thumbnail()

    def _build_ui(self) -> None:
        thumb_w, thumb_h, card_h = compute_card_dimensions(self._card_width)
        self.setFixedSize(self._card_width, card_h)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{self.item.filename}\nPath: {self.item.path}")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 4)
        layout.setSpacing(2)

        # 1. Thumbnail image on top (centered container for both landscape and vertical photos)
        self.thumb_label = QtWidgets.QLabel()
        self.thumb_label.setFixedSize(thumb_w, thumb_h)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet(f"background-color: {GRAY_0}; border-radius: 4px; border: 1px solid {GRAY_2};")
        self.thumb_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.thumb_label)

        # 2. Middle row: selection checkbox + format badge + edit indicator + remove button
        mid_row = QtWidgets.QHBoxLayout()
        mid_row.setContentsMargins(0, 0, 0, 0)
        mid_row.setSpacing(3)

        # Selection Checkbox
        self.select_cb = QtWidgets.QCheckBox()
        self.select_cb.setFixedSize(14, 14)
        self.select_cb.setChecked(self._is_multi_selected)
        self.select_cb.setToolTip("Select photo for parameter sync / batch operations (Ctrl+Click)")
        self.select_cb.setStyleSheet(f"""
            QCheckBox {{
                background: transparent;
                border: none;
            }}
            QCheckBox::indicator {{
                width: 12px;
                height: 12px;
                border: 1px solid {GRAY_4};
                border-radius: 3px;
                background-color: {GRAY_0};
            }}
            QCheckBox::indicator:hover {{
                border-color: {ACCENT_COLOR_TEXT};
            }}
            QCheckBox::indicator:checked {{
                background-color: {ACCENT_COLOR_TEXT};
                border-color: {ACCENT_COLOR_TEXT};
                image: url('{CHECKMARK_ICON_PATH}');
            }}
        """)
        self.select_cb.toggled.connect(self._on_cb_toggled)
        mid_row.addWidget(self.select_cb)

        fmt_text = "RAW" if self.item.is_raw else "RGB"
        fmt_color = ACCENT_COLOR_TEXT_SECONDARY if self.item.is_raw else TEXT_DIM
        self.badge_label = QtWidgets.QLabel(fmt_text)
        self.badge_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.badge_label.setFixedHeight(14)
        self.badge_label.setStyleSheet(
            f"color: {fmt_color}; font-size: 9px; font-weight: {FONT_WEIGHT_BOLD}; background: transparent; border: none;"
        )
        mid_row.addWidget(self.badge_label)

        self.edit_indicator = QtWidgets.QLabel("●" if self.item.has_settings else "○")
        self.edit_indicator.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.edit_indicator.setFixedHeight(14)
        edit_color = ACCENT_COLOR_TEXT if self.item.has_settings else GRAY_4
        self.edit_indicator.setStyleSheet(
            f"color: {edit_color}; font-size: 10px; background: transparent; border: none;"
        )
        self.edit_indicator.setToolTip("Edited (.spktrflm present)" if self.item.has_settings else "Unedited (uses active look)")
        mid_row.addWidget(self.edit_indicator)

        mid_row.addStretch(1)

        # Remove button: small '✕'
        self.remove_btn = QtWidgets.QPushButton("✕")
        self.remove_btn.setFixedSize(16, 16)
        self.remove_btn.setToolTip("Remove from gallery strip (Del)")
        self.remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_DIM};
                font-size: 10px;
                border: none;
                border-radius: 4px;
                padding: 0;
            }}
            QPushButton:hover {{
                background-color: #d9534f;
                color: #ffffff;
            }}
        """)
        self.remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self.index))
        mid_row.addWidget(self.remove_btn)

        layout.addLayout(mid_row)

        # 3. Bottom row: filename
        self.name_label = QtWidgets.QLabel(self.item.filename)
        self.name_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.name_label.setFixedHeight(14)
        self.name_label.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 10px; background: transparent; border: none;")
        layout.addWidget(self.name_label)

    def set_card_width(self, width: int) -> None:
        width = max(100, int(width))
        if width == self._card_width:
            return
        self._card_width = width
        thumb_w, thumb_h, card_h = compute_card_dimensions(width)
        self.setFixedSize(width, card_h)
        self.thumb_label.setFixedSize(thumb_w, thumb_h)
        self._update_thumbnail_display()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_thumbnail_display()

    def _load_thumbnail(self) -> None:
        service = ThumbnailService.instance()
        pixmap = service.get_thumbnail(
            self.item.path,
            target_width=MAX_GALLERY_WIDTH,
            target_height=MAX_GALLERY_WIDTH,
        )
        if pixmap is not None:
            self.set_thumbnail_pixmap(pixmap)
        else:
            self._pixmap = None
            self.thumb_label.setText("📷")
            self.thumb_label.setStyleSheet(f"background-color: {GRAY_0}; color: {GRAY_3}; font-size: 15px; border-radius: 4px;")

    def set_thumbnail_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self._update_thumbnail_display()

    def _update_thumbnail_display(self) -> None:
        if self._pixmap is not None and not self._pixmap.isNull():
            self.thumb_label.setStyleSheet(f"background-color: {GRAY_0}; border-radius: 4px; border: 1px solid {GRAY_2};")
            scaled = self._pixmap.scaled(self.thumb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.thumb_label.setPixmap(scaled)
        else:
            self.thumb_label.setText("📷")
            self.thumb_label.setStyleSheet(f"background-color: {GRAY_0}; color: {GRAY_3}; font-size: 15px; border-radius: 4px;")

    def set_active(self, active: bool) -> None:
        if self._is_active != active:
            self._is_active = active
            self.update_style()

    def set_selected(self, selected: bool) -> None:
        """Backwards compatibility alias for set_active."""
        self.set_active(selected)

    def set_multi_selected(self, multi_selected: bool) -> None:
        if self._is_multi_selected != multi_selected:
            self._is_multi_selected = multi_selected
            self.select_cb.blockSignals(True)
            self.select_cb.setChecked(multi_selected)
            self.select_cb.blockSignals(False)
            self.update_style()

    def _on_cb_toggled(self, checked: bool) -> None:
        if self._is_multi_selected != checked:
            self._is_multi_selected = checked
            self.update_style()
            self.selection_toggled.emit(self.index, checked)

    def refresh_item(self, item: GalleryItem) -> None:
        self.item = item
        self.name_label.setText(self.item.filename)
        fmt_text = "RAW" if self.item.is_raw else "RGB"
        fmt_color = ACCENT_COLOR_TEXT_SECONDARY if self.item.is_raw else TEXT_DIM
        self.badge_label.setText(fmt_text)
        self.badge_label.setStyleSheet(
            f"color: {fmt_color}; font-size: 9px; font-weight: {FONT_WEIGHT_BOLD}; background: transparent; border: none;"
        )
        self.refresh_status()
        self._load_thumbnail()

    def refresh_status(self) -> None:
        has_s = self.item.has_settings
        self.edit_indicator.setText("●" if has_s else "○")
        edit_color = ACCENT_COLOR_TEXT if has_s else GRAY_4
        self.edit_indicator.setStyleSheet(
            f"color: {edit_color}; font-size: 10px; background: transparent; border: none;"
        )
        self.edit_indicator.setToolTip("Edited (.spktrflm present)" if has_s else "Unedited (uses active look)")

    def update_style(self) -> None:
        # Distinct styling for Active vs Multi-Selected states
        if self._is_active and self._is_multi_selected:
            border_style = f"2px solid {ACCENT_COLOR_TEXT}"
            bg_color = "#2a3338"
            text_color = TEXT_HI
            font_weight = FONT_WEIGHT_BOLD
        elif self._is_active:
            border_style = f"2px solid {ACCENT_COLOR_TEXT}"
            bg_color = GRAY_3
            text_color = TEXT_HI
            font_weight = FONT_WEIGHT_SEMIBOLD
        elif self._is_multi_selected:
            border_style = "1.5px solid #63c3cf"
            bg_color = "#1e2930"
            text_color = "#c8f1f7"
            font_weight = FONT_WEIGHT_SEMIBOLD
        else:
            border_style = f"1px solid {GRAY_3}"
            bg_color = GRAY_1
            text_color = TEXT_MAIN
            font_weight = "normal"

        self.setStyleSheet(f"""
            PhotoCardWidget {{
                background-color: {bg_color};
                border: {border_style};
                border-radius: 8px;
            }}
            PhotoCardWidget QLabel {{
                background: transparent;
                background-color: transparent;
                border: none;
            }}
            PhotoCardWidget QCheckBox {{
                background: transparent;
                border: none;
            }}
        """)
        self.name_label.setStyleSheet(
            f"color: {text_color}; font-weight: {font_weight}; font-size: 10px; background: transparent; border: none;"
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers & Qt.ControlModifier:
                # Ctrl+Click toggles multi-selection
                new_state = not self._is_multi_selected
                self.set_multi_selected(new_state)
                self.selection_toggled.emit(self.index, new_state)
                event.accept()
                return
            elif modifiers & Qt.ShiftModifier:
                # Shift+Click requests range selection
                self.range_select_requested.emit(self.index)
                event.accept()
                return
            else:
                self.clicked.emit(self.index)
        elif event.button() == Qt.RightButton:
            self.context_menu_requested.emit(self.index, event.globalPos())
            event.accept()
            return

        super().mousePressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        event.ignore()


class HorizontalScrollArea(QtWidgets.QScrollArea):
    """Retained for backwards compatibility: horizontal scroll area with wheel support."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewport().installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj == self.viewport() and event.type() == QtCore.QEvent.Wheel:
            self.wheelEvent(event)
            return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.angleDelta().x()

        if delta != 0:
            h_bar = self.horizontalScrollBar()
            if h_bar is not None and h_bar.maximum() > 0:
                step = int((delta / 120.0) * 80)
                h_bar.setValue(h_bar.value() - step)
                event.accept()
                return

        super().wheelEvent(event)


class ThumbnailProgressOverlay(QtWidgets.QWidget):
    """Semi-transparent floating overlay that displays thumbnail generation progress."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("thumbnailProgressOverlay")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._build_ui()
        self.hide()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 10)
        layout.setAlignment(Qt.AlignCenter)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("thumbnailProgressCard")
        self.card.setStyleSheet(f"""
            QFrame#thumbnailProgressCard {{
                background-color: rgba(16, 16, 16, 0.92);
                border: 1px solid {GRAY_3};
                border-radius: 8px;
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
        """)
        card_layout = QtWidgets.QVBoxLayout(self.card)
        card_layout.setContentsMargins(8, 10, 8, 10)
        card_layout.setSpacing(5)
        card_layout.setAlignment(Qt.AlignCenter)

        self.title_label = QtWidgets.QLabel("Generating\nthumbnails...")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"color: {TEXT_HI}; font-size: 10px; font-weight: {FONT_WEIGHT_BOLD};")
        card_layout.addWidget(self.title_label)

        self.percent_label = QtWidgets.QLabel("0%")
        self.percent_label.setAlignment(Qt.AlignCenter)
        self.percent_label.setStyleSheet(f"color: {ACCENT_COLOR_TEXT}; font-size: 16px; font-weight: {FONT_WEIGHT_BOLD};")
        card_layout.addWidget(self.percent_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {GRAY_3};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT_COLOR_TEXT};
                border-radius: 2px;
            }}
        """)
        card_layout.addWidget(self.progress_bar)

        self.detail_label = QtWidgets.QLabel("0 / 0")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9px;")
        card_layout.addWidget(self.detail_label)

        layout.addWidget(self.card)

    def set_progress(self, completed: int, total: int, percentage: int) -> None:
        self.percent_label.setText(f"{percentage}%")
        self.progress_bar.setValue(percentage)
        self.detail_label.setText(f"{completed} / {total}")
        if not self.isVisible():
            self.show()
            self.raise_()


class GalleryFilmstripWidget(QtWidgets.QWidget):
    open_folder_requested = Signal()
    add_files_requested = Signal()
    scan_all_requested = Signal((str,), ())
    copy_look_requested = Signal()
    paste_look_requested = Signal()
    sync_param_requested = Signal()
    copy_param_requested = Signal()
    paste_param_requested = Signal()
    remove_requested = Signal()
    remove_item_requested = Signal(int)
    convert_dng_requested = Signal(int)

    def __init__(self, model: GalleryModel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._cards: list[PhotoCardWidget] = []
        self._is_expanded = True
        self.setFocusPolicy(Qt.ClickFocus)
        self._build_ui()

        # Connect model signals
        self.model.items_changed.connect(self._rebuild_cards)
        self.model.current_changed.connect(self._on_current_changed)
        self.model.selection_changed.connect(self._on_selection_changed)
        self.model.item_status_updated.connect(self._on_item_status_updated)

        # Connect thumbnail service
        service = ThumbnailService.instance()
        service.thumbnail_ready.connect(self._on_thumbnail_ready)
        service.progress_changed.connect(self._on_thumbnail_progress)
        service.all_completed.connect(self._on_thumbnail_all_completed)

    def _build_ui(self) -> None:
        self.setObjectName("galleryFilmstrip")
        self.setMinimumWidth(MIN_GALLERY_WIDTH)
        self.setMaximumWidth(MAX_GALLERY_WIDTH)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(0)

        # 1. Vertical Splitter to allow resizing Histogram/Parade Scope by height
        self.v_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.v_splitter.setObjectName("galleryVerticalSplitter")
        self.v_splitter.setChildrenCollapsible(False)
        self.v_splitter.setHandleWidth(4)

        # Scope (Histogram & DaVinci Resolve RGB Parade)
        self.histogram_widget = HistogramWidget(self)
        self.v_splitter.addWidget(self.histogram_widget)

        # Bottom section: Filmstrip Content (Toolbar + Photo cards scroll area)
        self.filmstrip_content = QtWidgets.QWidget()
        fc_layout = QtWidgets.QVBoxLayout(self.filmstrip_content)
        fc_layout.setContentsMargins(0, 2, 0, 0)
        fc_layout.setSpacing(4)

        # 2. Compact Vertical Header Toolbar
        self.toolbar = QtWidgets.QWidget()
        tb_layout = QtWidgets.QVBoxLayout(self.toolbar)
        tb_layout.setContentsMargins(2, 2, 2, 2)
        tb_layout.setSpacing(4)

        # Row 1: 📁 Folder & ➕ Add
        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(4)

        self.open_folder_btn = QtWidgets.QPushButton("📁 Folder")
        self.open_folder_btn.setToolTip("Open a directory of photos (RAW / RGB)")
        self.open_folder_btn.setFixedHeight(24)
        self.open_folder_btn.clicked.connect(self.open_folder_requested.emit)
        row1.addWidget(self.open_folder_btn)

        self.add_files_btn = QtWidgets.QPushButton("➕ Add")
        self.add_files_btn.setToolTip("Add individual photo files to gallery")
        self.add_files_btn.setFixedHeight(24)
        self.add_files_btn.clicked.connect(self.add_files_requested.emit)
        row1.addWidget(self.add_files_btn)

        tb_layout.addLayout(row1)

        # Row 2: Nav controls: ▲ / Photo X of Y / ▼
        row2 = QtWidgets.QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(2)

        self.prev_btn = QtWidgets.QPushButton("▲")
        self.prev_btn.setToolTip("Previous photo (Up / Left arrow)")
        self.prev_btn.setFixedSize(22, 22)
        self.prev_btn.clicked.connect(self.model.previous_photo)
        row2.addWidget(self.prev_btn)

        self.counter_label = QtWidgets.QLabel("No photos")
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setStyleSheet(
            f"color: {TEXT_HI}; font-size: 10px; font-weight: {FONT_WEIGHT_SEMIBOLD}; background: transparent; border: none;"
        )
        row2.addWidget(self.counter_label, 1)

        self.next_btn = QtWidgets.QPushButton("▼")
        self.next_btn.setToolTip("Next photo (Down / Right arrow)")
        self.next_btn.setFixedSize(22, 22)
        self.next_btn.clicked.connect(self.model.next_photo)
        row2.addWidget(self.next_btn)

        tb_layout.addLayout(row2)

        # Row 3: Selection: [☑ All] [☐ None] Sel: 0
        row_sel = QtWidgets.QHBoxLayout()
        row_sel.setContentsMargins(0, 0, 0, 0)
        row_sel.setSpacing(2)

        self.select_all_btn = QtWidgets.QPushButton("☑ All")
        self.select_all_btn.setToolTip("Select all photos in filmstrip (Ctrl+A)")
        self.select_all_btn.setFixedHeight(20)
        self.select_all_btn.setStyleSheet(f"font-size: 10px; padding: 1px 3px;")
        self.select_all_btn.clicked.connect(self.model.select_all)
        row_sel.addWidget(self.select_all_btn)

        self.deselect_all_btn = QtWidgets.QPushButton("☐ None")
        self.deselect_all_btn.setToolTip("Deselect all photos (Escape)")
        self.deselect_all_btn.setFixedHeight(20)
        self.deselect_all_btn.setStyleSheet(f"font-size: 10px; padding: 1px 3px;")
        self.deselect_all_btn.clicked.connect(self.model.deselect_all)
        row_sel.addWidget(self.deselect_all_btn)

        self.selection_label = QtWidgets.QLabel("Sel: 0")
        self.selection_label.setAlignment(Qt.AlignCenter)
        self.selection_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; background: transparent; border: none;"
        )
        row_sel.addWidget(self.selection_label, 1)

        tb_layout.addLayout(row_sel)

        # Row 4: Action buttons: 📋 Copy Look | 📌 Paste Look | 🗑️ Del
        row4 = QtWidgets.QHBoxLayout()
        row4.setContentsMargins(0, 0, 0, 0)
        row4.setSpacing(3)

        self.copy_btn = QtWidgets.QPushButton("📋")
        self.copy_btn.setToolTip("Copy active photo's entire look (Ctrl+C)")
        self.copy_btn.setFixedHeight(24)
        self.copy_btn.clicked.connect(self.copy_look_requested.emit)
        row4.addWidget(self.copy_btn)

        self.paste_btn = QtWidgets.QPushButton("📌")
        self.paste_btn.setToolTip("Paste entire look to selected or active photo and auto-save (Ctrl+V)")
        self.paste_btn.setFixedHeight(24)
        self.paste_btn.clicked.connect(self.paste_look_requested.emit)
        row4.addWidget(self.paste_btn)

        self.remove_btn = QtWidgets.QPushButton("🗑️")
        self.remove_btn.setToolTip("Remove selected or active photo from filmstrip (Del / Backspace)")
        self.remove_btn.setFixedHeight(24)
        self.remove_btn.clicked.connect(self.remove_requested.emit)
        row4.addWidget(self.remove_btn)

        tb_layout.addLayout(row4)

        # Row 5: ⚡ Sync Settings...
        self.sync_param_btn = QtWidgets.QPushButton("⚡ Sync Settings...")
        self.sync_param_btn.setToolTip("Sync or copy active settings across selected photos...")
        self.sync_param_btn.setFixedHeight(24)
        self.sync_param_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {GRAY_2};
                color: {TEXT_HI};
                border: 1px solid {GRAY_4};
                border-radius: 6px;
                font-size: 10px;
                font-weight: {FONT_WEIGHT_SEMIBOLD};
            }}
            QPushButton:hover {{
                background-color: {GRAY_3};
                border-color: {ACCENT_COLOR_TEXT};
            }}
        """)
        self.sync_param_btn.clicked.connect(self.sync_param_requested.emit)
        tb_layout.addWidget(self.sync_param_btn)

        # Row 6: Primary Export Action: SCAN
        self.scan_all_btn = QtWidgets.QPushButton("🚀 SCAN ALL")
        self.scan_all_btn.setToolTip("Batch develop and export photos in the gallery using their .spktrflm settings")
        self.scan_all_btn.setFixedHeight(26)
        self.scan_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR_TEXT};
                color: #000;
                font-weight: {FONT_WEIGHT_BOLD};
                padding: 0 4px;
                border-radius: 6px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: #fca987;
            }}
            QPushButton:disabled {{
                background-color: {GRAY_3};
                color: {TEXT_DIM};
            }}
            QPushButton::menu-indicator {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                right: 4px;
            }}
        """)

        # Dropdown menu to choose between "Scan All Photos" and "Scan Selected Photos"
        self.scan_menu = QtWidgets.QMenu(self.scan_all_btn)
        self.scan_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {GRAY_1};
                color: {TEXT_MAIN};
                border: 1px solid {GRAY_3};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 5px 18px 5px 14px;
                border-radius: 5px;
            }}
            QMenu::item:selected {{
                background-color: {ACCENT_COLOR_TEXT};
                color: #000000;
            }}
            QMenu::item:disabled {{
                color: {TEXT_DIM};
            }}
        """)
        self.scan_all_action = self.scan_menu.addAction("🚀 Scan All Photos")
        self.scan_all_action.triggered.connect(lambda: self.scan_all_requested.emit("all"))
        self.scan_selected_action = self.scan_menu.addAction("🚀 Scan Selected Photos")
        self.scan_selected_action.triggered.connect(lambda: self.scan_all_requested.emit("selected"))
        self.scan_all_btn.setMenu(self.scan_menu)

        tb_layout.addWidget(self.scan_all_btn)

        fc_layout.addWidget(self.toolbar)

        # 3. Vertical Scroll Area for Cards
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {GRAY_0};
                border: 1px solid {GRAY_2};
                border-radius: 8px;
            }}
        """)
        self.scroll_area.viewport().installEventFilter(self)

        self.cards_container = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(4, 4, 4, 4)
        self.cards_layout.setSpacing(6)
        self.cards_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self.scroll_area.setWidget(self.cards_container)
        fc_layout.addWidget(self.scroll_area, 1)

        self.v_splitter.addWidget(self.filmstrip_content)

        # Sizing: scope retains compact default height (~130px), cards area expands
        self.v_splitter.setStretchFactor(0, 0)
        self.v_splitter.setStretchFactor(1, 1)
        self.v_splitter.setSizes([130, 600])

        main_layout.addWidget(self.v_splitter, 1)

        # 4. Floating progress overlay for thumbnail generation
        self.overlay = ThumbnailProgressOverlay(self)
        self.overlay.hide()

        # Update overlay position when vertical splitter is moved
        self.v_splitter.splitterMoved.connect(lambda *_: self._update_overlay_geometry())

        self._update_controls_state()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            hasattr(self, "scroll_area")
            and obj == self.scroll_area.viewport()
            and event.type() == QtCore.QEvent.Resize
        ):
            self._update_cards_width()
        return super().eventFilter(obj, event)

    def _current_card_width(self) -> int:
        if hasattr(self, "scroll_area") and self.scroll_area.viewport() is not None:
            vp_w = self.scroll_area.viewport().width()
            if vp_w > 50:
                card_w = vp_w - 10
                return max(100, min(card_w, MAX_GALLERY_WIDTH - 20))
        w = self.width() if self.width() > 50 else DEFAULT_GALLERY_WIDTH
        return max(100, min(w - 34, MAX_GALLERY_WIDTH - 20))

    def _update_cards_width(self) -> None:
        if not self._cards:
            return
        target_w = self._current_card_width()
        for card in self._cards:
            card.set_card_width(target_w)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_overlay_geometry()
        self._update_cards_width()

    def _update_overlay_geometry(self) -> None:
        if hasattr(self, "overlay") and hasattr(self, "scroll_area"):
            top_left = self.scroll_area.mapTo(self, QPoint(0, 0))
            self.overlay.setGeometry(QtCore.QRect(top_left, self.scroll_area.size()))

    def _toggle_expanded(self) -> None:
        self._is_expanded = not self._is_expanded
        self.scroll_area.setVisible(self._is_expanded)

    def _rebuild_cards(self) -> None:
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        current_idx = self.model.current_index
        items = self.model.items

        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        target_w = self._current_card_width()
        for idx, item in enumerate(items):
            card = PhotoCardWidget(
                index=idx,
                item=item,
                is_active=(idx == current_idx),
                is_multi_selected=self.model.is_selected(idx),
                card_width=target_w,
                parent=self.cards_container,
            )
            card.clicked.connect(self.model.select_index)
            card.remove_clicked.connect(self.remove_item_requested.emit)
            card.selection_toggled.connect(self._on_card_selection_toggled)
            card.range_select_requested.connect(self._on_card_range_select)
            card.context_menu_requested.connect(self._show_card_context_menu)
            self._cards.append(card)
            self.cards_layout.addWidget(card)

        self.cards_layout.addStretch(1)
        self._update_controls_state()

        if hasattr(self, "overlay") and self.overlay.isVisible():
            self.overlay.raise_()

        if 0 <= current_idx < len(self._cards):
            self._scroll_to_card(current_idx)

    def _on_card_selection_toggled(self, index: int, is_selected: bool) -> None:
        self.model.set_selected(index, is_selected)

    def _on_card_range_select(self, target_index: int) -> None:
        from_idx = self.model.current_index if self.model.current_index >= 0 else target_index
        self.model.select_range(from_idx, target_index)

    def _on_current_changed(self, current_idx: int, item: GalleryItem | None) -> None:
        for idx, card in enumerate(self._cards):
            card.set_active(idx == current_idx)

        self._update_controls_state()
        if 0 <= current_idx < len(self._cards):
            self._scroll_to_card(current_idx)

    def _on_selection_changed(self) -> None:
        for card in self._cards:
            card.set_multi_selected(self.model.is_selected(card.index))

        sel_count = self.model.selected_count
        self.selection_label.setText(f"Sel: {sel_count}")
        self.deselect_all_btn.setEnabled(sel_count > 0)
        self.select_all_btn.setEnabled(self.model.count > 0)
        self._update_controls_state()

    def _on_item_status_updated(self, index: int) -> None:
        if 0 <= index < len(self._cards):
            item = self.model.item_at(index)
            if item is not None:
                self._cards[index].refresh_item(item)
            else:
                self._cards[index].refresh_status()

    def _on_thumbnail_ready(self, canon_path: str, pixmap: QtGui.QPixmap) -> None:
        for card in self._cards:
            if str(card.item.path.resolve()).lower() == canon_path:
                card.set_thumbnail_pixmap(pixmap)

    def _on_thumbnail_progress(self, completed: int, total: int, percentage: int) -> None:
        if hasattr(self, "overlay") and total > 0:
            self._update_overlay_geometry()
            self.overlay.set_progress(completed, total, percentage)

    def _on_thumbnail_all_completed(self) -> None:
        if hasattr(self, "overlay"):
            self.overlay.hide()

    def _scroll_to_card(self, index: int) -> None:
        if 0 <= index < len(self._cards):
            card = self._cards[index]
            self.scroll_area.ensureWidgetVisible(card, 0, 50)

    def _show_card_context_menu(self, index: int, global_pos: QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {GRAY_1};
                color: {TEXT_MAIN};
                border: 1px solid {GRAY_3};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 4px 18px 4px 18px;
                border-radius: 5px;
            }}
            QMenu::item:selected {{
                background-color: {ACCENT_COLOR_TEXT};
                color: #000;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {GRAY_3};
                margin: 4px 8px;
            }}
        """)

        # 1. Selection actions
        is_sel = self.model.is_selected(index)
        toggle_sel_act = menu.addAction("☑ Deselect Photo" if is_sel else "☑ Select Photo")
        sel_all_act = menu.addAction("Select All (Ctrl+A)")
        desel_all_act = menu.addAction("Deselect All")

        menu.addSeparator()

        # 2. Look actions
        copy_look_act = menu.addAction("📋 Copy Entire Look (Ctrl+C)")
        if self.model.selected_count > 0 and index in self.model.selected_indices:
            paste_look_act = menu.addAction(f"📌 Paste Entire Look to {self.model.selected_count} Selected Photos (Ctrl+V)")
        else:
            paste_look_act = menu.addAction("📌 Paste Entire Look (Ctrl+V)")
        paste_look_act.setEnabled(self.model.clipboard_state is not None)

        menu.addSeparator()

        # 3. Parameter sync
        sync_param_act = menu.addAction("⚡ Sync / Copy Active Settings...")

        clip_dict = self.model.get_params_clipboard_dict()
        if clip_dict:
            sel_count = self.model.selected_count
            target_str = f"{sel_count} Selected Photos" if sel_count > 0 else "Active Photo"
            if len(clip_dict) == 1:
                p_id = next(iter(clip_dict.keys()))
                p_def = get_param_def(p_id)
                p_name = p_def.name if p_def else p_id
                paste_param_act = menu.addAction(f"📌 Paste '{p_name}' to {target_str}")
            else:
                paste_param_act = menu.addAction(f"📌 Paste {len(clip_dict)} Settings to {target_str}")
        else:
            paste_param_act = None

        menu.addSeparator()

        # 4. RAW to DNG Conversion
        convert_dng_act = None
        sel_raws = [
            i for i in self.model.selected_indices
            if (it := self.model.item_at(i)) is not None and it.is_raw and it.extension != ".dng"
        ]
        curr_it = self.model.item_at(index)
        if len(sel_raws) > 1 and index in self.model.selected_indices:
            convert_dng_act = menu.addAction(f"📷 Convert {len(sel_raws)} Selected RAWs to DNG-RAW")
        elif curr_it is not None and curr_it.is_raw and curr_it.extension != ".dng":
            convert_dng_act = menu.addAction("📷 Convert to DNG-RAW")

        if convert_dng_act is not None:
            menu.addSeparator()

        # 5. Batch Scan
        scan_sel_act = None
        scan_all_act = None
        if self.model.count > 0:
            if self.model.selected_count > 0:
                scan_sel_act = menu.addAction(f"🚀 Scan {self.model.selected_count} Selected Photos...")
            scan_all_act = menu.addAction(f"🚀 Scan All {self.model.count} Photos...")
            menu.addSeparator()

        # 6. Remove
        if self.model.selected_count > 1 and index in self.model.selected_indices:
            remove_act = menu.addAction(f"🗑️ Remove {self.model.selected_count} Selected Photos from Strip (Del)")
        else:
            remove_act = menu.addAction("🗑️ Remove from Strip (Del)")

        chosen = menu.exec_(global_pos)
        if chosen is None:
            return

        if chosen == toggle_sel_act:
            self.model.toggle_selection(index)
        elif chosen == sel_all_act:
            self.model.select_all()
        elif chosen == desel_all_act:
            self.model.deselect_all()
        elif chosen == copy_look_act:
            self.copy_look_requested.emit()
        elif chosen == paste_look_act:
            self.paste_look_requested.emit()
        elif chosen == sync_param_act:
            self.sync_param_requested.emit()
        elif paste_param_act is not None and chosen == paste_param_act:
            self.paste_param_requested.emit()
        elif convert_dng_act is not None and chosen == convert_dng_act:
            self.convert_dng_requested.emit(index)
        elif scan_sel_act is not None and chosen == scan_sel_act:
            self.scan_all_requested.emit("selected")
        elif scan_all_act is not None and chosen == scan_all_act:
            self.scan_all_requested.emit("all")
        elif chosen == remove_act:
            if self.model.selected_count > 1 and index in self.model.selected_indices:
                self.remove_requested.emit()
            else:
                self.remove_item_requested.emit(index)

    def _update_controls_state(self) -> None:
        count = self.model.count
        curr = self.model.current_index
        sel_count = self.model.selected_count

        if count == 0:
            self.counter_label.setText("No photos")
            self.selection_label.setText("Sel: 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.remove_btn.setEnabled(False)
            self.remove_btn.setToolTip("Remove photo(s) from filmstrip (Del / Backspace)")
            self.scan_all_btn.setEnabled(False)
            self.scan_all_btn.setText("🚀 SCAN ALL")
            if hasattr(self, "scan_all_action"):
                self.scan_all_action.setText("🚀 Scan All Photos")
            if hasattr(self, "scan_selected_action"):
                self.scan_selected_action.setText("🚀 Scan Selected Photos")
                self.scan_selected_action.setEnabled(False)
            self.copy_btn.setEnabled(False)
            self.copy_btn.setToolTip("Copy active photo's entire look (Ctrl+C)")
            self.paste_btn.setEnabled(False)
            self.paste_btn.setToolTip("Paste entire look and auto-save (Ctrl+V)")
            self.select_all_btn.setEnabled(False)
            self.deselect_all_btn.setEnabled(False)
            self.sync_param_btn.setEnabled(False)
        else:
            self.counter_label.setText(f"{curr + 1} / {count}")
            self.selection_label.setText(f"Sel: {sel_count}")
            self.prev_btn.setEnabled(curr > 0)
            self.next_btn.setEnabled(curr + 1 < count)
            self.remove_btn.setEnabled(curr >= 0 or sel_count > 0)
            if sel_count > 0:
                self.remove_btn.setToolTip(f"Remove {sel_count} selected photo(s) from filmstrip (Del / Backspace)")
            else:
                self.remove_btn.setToolTip("Remove active photo from filmstrip (Del / Backspace)")

            self.scan_all_btn.setEnabled(True)
            if hasattr(self, "scan_all_action"):
                self.scan_all_action.setText(f"🚀 Scan All Photos ({count})")
            if hasattr(self, "scan_selected_action"):
                if sel_count > 0:
                    self.scan_all_btn.setText(f"🚀 SCAN ({sel_count} Sel)")
                    self.scan_selected_action.setText(f"🚀 Scan Selected Photos ({sel_count})")
                    self.scan_selected_action.setEnabled(True)
                else:
                    self.scan_all_btn.setText("🚀 SCAN ALL")
                    self.scan_selected_action.setText("🚀 Scan Selected Photos (none)")
                    self.scan_selected_action.setEnabled(False)
            self.copy_btn.setEnabled(curr >= 0)
            self.copy_btn.setToolTip("Copy active photo's entire look (Ctrl+C)")
            self.paste_btn.setEnabled(self.model.clipboard_state is not None and (curr >= 0 or sel_count > 0))
            if sel_count > 0:
                self.paste_btn.setToolTip(f"Paste entire look to {sel_count} selected photo(s) and auto-save (Ctrl+V)")
            else:
                self.paste_btn.setToolTip("Paste entire look to active photo and auto-save (Ctrl+V)")
            self.select_all_btn.setEnabled(True)
            self.deselect_all_btn.setEnabled(sel_count > 0)
            self.sync_param_btn.setEnabled(curr >= 0)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        modifiers = event.modifiers()
        if modifiers & Qt.ControlModifier and event.key() == Qt.Key_A:
            self.model.select_all()
            event.accept()
            return
        elif event.key() == Qt.Key_Escape:
            self.model.deselect_all()
            event.accept()
            return
        elif modifiers & Qt.ControlModifier and event.key() == Qt.Key_C:
            self.copy_look_requested.emit()
            event.accept()
            return
        elif modifiers & Qt.ControlModifier and event.key() == Qt.Key_V:
            self.paste_look_requested.emit()
            event.accept()
            return
        elif event.key() in (Qt.Key_Left, Qt.Key_Up):
            self.model.previous_photo()
            event.accept()
            return
        elif event.key() in (Qt.Key_Right, Qt.Key_Down):
            self.model.next_photo()
            event.accept()
            return
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def set_histogram_image(self, image: np.ndarray | None) -> None:
        """Update the displayed histogram with new RGB or float image data."""
        if hasattr(self, 'histogram_widget'):
            self.histogram_widget.set_image(image)
