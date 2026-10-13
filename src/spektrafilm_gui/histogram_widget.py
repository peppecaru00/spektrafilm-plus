from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QPointF, QRectF, Qt, Signal

from spektrafilm_gui.theme_palette import (
    ACCENT_COLOR_TEXT,
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

logger = logging.getLogger(__name__)

COLOR_RED = QtGui.QColor(235, 75, 75)
COLOR_GREEN = QtGui.QColor(65, 195, 95)
COLOR_BLUE = QtGui.QColor(60, 140, 245)
COLOR_LUMA = QtGui.QColor(215, 215, 215)
COLOR_GRID = QtGui.QColor(55, 55, 55, 110)
COLOR_GRID_SOLID = QtGui.QColor(70, 70, 70, 160)

COLOR_RED_FILL = QtGui.QColor(235, 75, 75, 50)
COLOR_GREEN_FILL = QtGui.QColor(65, 195, 95, 50)
COLOR_BLUE_FILL = QtGui.QColor(60, 140, 245, 50)
COLOR_LUMA_FILL = QtGui.QColor(215, 215, 215, 35)

# Histogram Channel Display Modes
MODE_RGB = "RGB"
MODE_LUMA = "Luma"
MODE_RED = "R"
MODE_GREEN = "G"
MODE_BLUE = "B"
ALL_MODES = [MODE_RGB, MODE_LUMA, MODE_RED, MODE_GREEN, MODE_BLUE]

# Scope Types
SCOPE_HISTOGRAM = "Histogram"
SCOPE_PARADE = "Parade"
ALL_SCOPES = [SCOPE_HISTOGRAM, SCOPE_PARADE]


def compute_channel_histogram(channel_data: np.ndarray, num_bins: int = 256) -> tuple[np.ndarray, float, float]:
    """Compute 256-bin histogram, shadow clip fraction, and highlight clip fraction."""
    flat = channel_data.ravel()
    if flat.size == 0:
        return np.zeros(num_bins, dtype=np.float32), 0.0, 0.0

    counts = np.bincount(flat, minlength=num_bins)[:num_bins].astype(np.float32)
    total_pixels = float(flat.size)
    shadow_clip = float(counts[0] / total_pixels) if total_pixels > 0 else 0.0
    highlight_clip = float(counts[-1] / total_pixels) if total_pixels > 0 else 0.0
    return counts, shadow_clip, highlight_clip


def compute_parade_image(
    image_arr: np.ndarray | None,
    target_w: int,
    target_h: int,
    gap: int = 3,
) -> QtGui.QImage | None:
    """Compute an RGB Parade waveform image partitioned into Red, Green, and Blue traces.

    Similar to DaVinci Resolve's RGB Parade:
    - Left partition: Red channel trace
    - Middle partition: Green channel trace
    - Right partition: Blue channel trace
    - Horizontal axis across each partition corresponds to the image width.
    - Vertical axis corresponds to pixel signal levels from 0.0 (bottom) to 1.0 (top).
    - Color density scales non-linearly with phosphor glow for an authentic CRT video scope look.
    """
    if image_arr is None or image_arr.size == 0 or target_w < 30 or target_h < 10:
        return None

    part_w = max(8, (target_w - 2 * gap) // 3)
    part_h = max(8, target_h)

    h_img, w_img = image_arr.shape[:2]
    h_sub = min(h_img, 256)

    row_idx = np.linspace(0, h_img - 1, h_sub, dtype=np.int32)
    col_idx = np.linspace(0, w_img - 1, part_w, dtype=np.int32)

    # Subsample image to [h_sub, part_w, 3] in [0.0, 1.0]
    sub = image_arr[np.ix_(row_idx, col_idx)]
    if np.issubdtype(sub.dtype, np.floating):
        sub = np.nan_to_num(sub, nan=0.0, posinf=1.0, neginf=0.0)
        max_val = float(np.max(sub)) if sub.size > 0 else 1.0
        if max_val > 1.05:
            sub = sub / 255.0
        sub = np.clip(sub, 0.0, 1.0)
    else:
        sub = np.clip(sub.astype(np.float32) / 255.0, 0.0, 1.0)

    if sub.ndim == 2:
        sub = np.stack([sub, sub, sub], axis=-1)
    elif sub.shape[-1] > 3:
        sub = sub[..., :3]

    # Map signal levels: 1.0 is at top (y = 0), 0.0 is at bottom (y = part_h - 1)
    y_idx = np.clip(((1.0 - sub) * (part_h - 1)).astype(np.int32), 0, part_h - 1)
    x_idx = np.broadcast_to(np.arange(part_w, dtype=np.int32), (h_sub, part_w))
    x_flat = x_idx.ravel()

    full_w = part_w * 3 + 2 * gap
    rgba = np.zeros((part_h, full_w, 4), dtype=np.uint8)

    # Channel phosphor colors (warm CRT red, vibrant green, vivid blue)
    colors = [
        np.array([240, 70, 70], dtype=np.float32),   # Red
        np.array([55, 220, 95], dtype=np.float32),   # Green
        np.array([65, 155, 255], dtype=np.float32),  # Blue
    ]

    for c in range(3):
        y_flat = y_idx[..., c].ravel()
        linear = y_flat * part_w + x_flat
        counts = np.bincount(linear, minlength=part_h * part_w).reshape((part_h, part_w)).astype(np.float32)

        has_signal = counts > 0
        if not np.any(has_signal):
            continue

        # Non-linear density scaling (sqrt scale) for CRT scope glow
        norm = np.clip(np.sqrt(counts) / 3.0, 0.0, 1.0)

        # Base trace tint with bright phosphor highlight at dense points
        cr = np.clip(colors[c][0] * (0.6 + 0.4 * norm) + 120.0 * (norm ** 2), 0, 255)
        cg = np.clip(colors[c][1] * (0.6 + 0.4 * norm) + 120.0 * (norm ** 2), 0, 255)
        cb = np.clip(colors[c][2] * (0.6 + 0.4 * norm) + 120.0 * (norm ** 2), 0, 255)
        alpha = np.clip(75.0 + 180.0 * norm, 0, 255)

        part_rgba = np.zeros((part_h, part_w, 4), dtype=np.uint8)
        part_rgba[has_signal, 0] = cr[has_signal].astype(np.uint8)
        part_rgba[has_signal, 1] = cg[has_signal].astype(np.uint8)
        part_rgba[has_signal, 2] = cb[has_signal].astype(np.uint8)
        part_rgba[has_signal, 3] = alpha[has_signal].astype(np.uint8)

        x_start = c * (part_w + gap)
        rgba[:, x_start:x_start + part_w] = part_rgba

    return QtGui.QImage(rgba.data, full_w, part_h, full_w * 4, QtGui.QImage.Format_RGBA8888).copy()


class HistogramCanvas(QtWidgets.QWidget):
    """Drawing surface for RGB / Luminance histogram curves or DaVinci Resolve style RGB Parade."""

    scope_type_changed = Signal(str)
    mode_changed = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.setMaximumHeight(800)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._scope_type: str = SCOPE_HISTOGRAM
        self._mode: str = MODE_RGB

        # Raw/subsampled image cache for parade recalculation upon resize
        self._current_image: np.ndarray | None = None
        self._parade_image: QtGui.QImage | None = None
        self._parade_cache_size: tuple[int, int] | None = None

        # Histogram data
        self._counts_r: np.ndarray | None = None
        self._counts_g: np.ndarray | None = None
        self._counts_b: np.ndarray | None = None
        self._counts_l: np.ndarray | None = None

        self._norm_r: np.ndarray | None = None
        self._norm_g: np.ndarray | None = None
        self._norm_b: np.ndarray | None = None
        self._norm_l: np.ndarray | None = None

        self._shadow_clip: float = 0.0
        self._highlight_clip: float = 0.0
        self._has_data = False

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(240, 104)

    @property
    def scope_type(self) -> str:
        return self._scope_type

    @scope_type.setter
    def scope_type(self, val: str) -> None:
        if val in ALL_SCOPES and val != self._scope_type:
            self._scope_type = val
            self._update_tooltip()
            self.scope_type_changed.emit(val)
            self.update()

    def toggle_scope_type(self) -> str:
        next_scope = SCOPE_PARADE if self._scope_type == SCOPE_HISTOGRAM else SCOPE_HISTOGRAM
        self.scope_type = next_scope
        return next_scope

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, val: str) -> None:
        if val in ALL_MODES and val != self._mode:
            self._mode = val
            self.mode_changed.emit(val)
            self.update()

    def cycle_mode(self) -> str:
        idx = ALL_MODES.index(self._mode)
        next_mode = ALL_MODES[(idx + 1) % len(ALL_MODES)]
        self.mode = next_mode
        return next_mode

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Invalidate cached parade image so it recalculates to match new width/height
        self._parade_image = None
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            if self._scope_type == SCOPE_HISTOGRAM:
                new_mode = self.cycle_mode()
                parent = self.parent()
                if isinstance(parent, HistogramWidget):
                    parent._on_canvas_mode_cycled(new_mode)
            else:
                # In Parade mode, clicking canvas cycles back or toggles
                parent = self.parent()
                if isinstance(parent, HistogramWidget):
                    parent.toggle_scope_type()
            event.accept()
            return
        super().mousePressEvent(event)

    def set_image(self, image: np.ndarray | None) -> None:
        self._current_image = image
        self._parade_image = None
        self._parade_cache_size = None

        if image is None:
            self._has_data = False
            self._counts_r = self._counts_g = self._counts_b = self._counts_l = None
            self._norm_r = self._norm_g = self._norm_b = self._norm_l = None
            self._shadow_clip = 0.0
            self._highlight_clip = 0.0
            self.setToolTip("No active image to display.")
            self.update()
            return

        arr = np.asarray(image)
        if arr.ndim < 2:
            self.set_image(None)
            return

        # Ensure 3 channels
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]

        # Fast subsampling for large images (> 300,000 pixels) to ensure instant rendering (< 2ms)
        h, w = arr.shape[:2]
        total_pixels = h * w
        if total_pixels > 350_000:
            step = max(1, int(np.sqrt(total_pixels / 300_000)))
            arr_hist = arr[::step, ::step]
        else:
            arr_hist = arr

        # Convert to uint8 [0, 255] for histogram
        if np.issubdtype(arr_hist.dtype, np.floating):
            arr_clean = np.nan_to_num(arr_hist, nan=0.0, posinf=1.0, neginf=0.0)
            max_val = float(np.max(arr_clean)) if arr_clean.size > 0 else 1.0
            if max_val <= 1.05:
                arr_u8 = np.clip(arr_clean * 255.0, 0, 255).astype(np.uint8)
            else:
                arr_u8 = np.clip(arr_clean, 0, 255).astype(np.uint8)
        else:
            arr_u8 = np.clip(arr_hist, 0, 255).astype(np.uint8)

        r_data = arr_u8[..., 0]
        g_data = arr_u8[..., 1]
        b_data = arr_u8[..., 2]

        # Compute luma using standard Rec.709 coefficients
        l_data = (0.2126 * r_data.astype(np.float32) + 0.7152 * g_data.astype(np.float32) + 0.0722 * b_data.astype(np.float32)).astype(np.uint8)

        cr, sr_c, hr_c = compute_channel_histogram(r_data)
        cg, sg_c, hg_c = compute_channel_histogram(g_data)
        cb, sb_c, hb_c = compute_channel_histogram(b_data)
        cl, sl_c, hl_c = compute_channel_histogram(l_data)

        self._counts_r = cr
        self._counts_g = cg
        self._counts_b = cb
        self._counts_l = cl

        self._shadow_clip = max(sr_c, sg_c, sb_c)
        self._highlight_clip = max(hr_c, hg_c, hb_c)

        # Scale with square root (standard photo histogram scaling) so both huge spikes and fine tones are visible
        def normalize_counts(counts: np.ndarray) -> np.ndarray:
            sq = np.sqrt(counts)
            mx = float(np.max(sq)) if sq.size > 0 else 1.0
            if mx < 1e-6:
                return np.zeros_like(sq)
            return (sq / mx).astype(np.float32)

        self._norm_r = normalize_counts(cr)
        self._norm_g = normalize_counts(cg)
        self._norm_b = normalize_counts(cb)
        self._norm_l = normalize_counts(cl)

        self._has_data = True
        self._update_tooltip()
        self.update()

    def _update_tooltip(self) -> None:
        if not self._has_data or self._counts_r is None:
            self.setToolTip("No active image to display.")
            return

        if self._scope_type == SCOPE_PARADE:
            tip_lines = [
                "<b>RGB Parade Scope</b> (DaVinci Resolve style)",
                "R: Left | G: Center | B: Right",
                "Vertical axis: 0% (Shadows) to 100% (Highlights)",
                "<i>Click canvas to switch to Histogram</i>",
            ]
        else:
            mean_r = float(np.mean(self._counts_r))
            tip_lines = [
                "<b>RGB Histogram</b> (Click to toggle channel mode)",
                "<i>Use header buttons to switch to RGB Parade</i>",
            ]
            if self._shadow_clip > 0.0005:
                tip_lines.append(f"<span style='color: #60a5fa;'>Shadow clip: {self._shadow_clip * 100.0:.2f}%</span>")
            if self._highlight_clip > 0.0005:
                tip_lines.append(f"<span style='color: #f87171;'>Highlight clip: {self._highlight_clip * 100.0:.2f}%</span>")

        self.setToolTip("<br>".join(tip_lines))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()
        rect = QRectF(1.0, 1.0, float(w - 2), float(h - 2))

        # 1. Dark Background with subtle border
        painter.setPen(QtGui.QPen(QtGui.QColor(GRAY_2), 1.0))
        painter.setBrush(QtGui.QColor(GRAY_0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        if not self._has_data:
            painter.setPen(QtGui.QPen(QtGui.QColor(TEXT_DIM), 1.0))
            painter.drawText(rect, Qt.AlignCenter, "No image")
            return

        if self._scope_type == SCOPE_PARADE:
            self._paint_parade(painter, rect)
        else:
            self._paint_histogram(painter, rect)

    def _paint_histogram(self, painter: QtGui.QPainter, rect: QRectF) -> None:
        # Zone Grid Lines (20%, 40%, 60%, 80%)
        painter.setPen(QtGui.QPen(COLOR_GRID, 1.0, Qt.DotLine))
        for frac in (0.2, 0.4, 0.6, 0.8):
            gx = rect.left() + frac * rect.width()
            painter.drawLine(QPointF(gx, rect.top() + 3.0), QPointF(gx, rect.bottom() - 3.0))

        plot_rect = rect.adjusted(2.0, 2.0, -2.0, -2.0)
        px_w = plot_rect.width()
        px_h = plot_rect.height()
        y_bot = plot_rect.bottom()
        x_left = plot_rect.left()

        def draw_channel_curve(norm_arr: np.ndarray, stroke_color: QtGui.QColor, fill_color: QtGui.QColor) -> None:
            if norm_arr is None or len(norm_arr) != 256:
                return

            fill_path = QtGui.QPainterPath()
            fill_path.moveTo(x_left, y_bot)
            stroke_path = QtGui.QPainterPath()

            for i in range(256):
                cx = x_left + (i / 255.0) * px_w
                val = float(norm_arr[i])
                cy = y_bot - val * px_h
                fill_path.lineTo(cx, cy)
                if i == 0:
                    stroke_path.moveTo(cx, cy)
                else:
                    stroke_path.lineTo(cx, cy)

            fill_path.lineTo(x_left + px_w, y_bot)
            fill_path.closeSubpath()

            # Fill area
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill_color)
            painter.drawPath(fill_path)

            # Outline curve
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QtGui.QPen(stroke_color, 1.2))
            painter.drawPath(stroke_path)

        # Draw Channels According to Mode
        if self._mode == MODE_RGB:
            draw_channel_curve(self._norm_r, COLOR_RED, COLOR_RED_FILL)
            draw_channel_curve(self._norm_g, COLOR_GREEN, COLOR_GREEN_FILL)
            draw_channel_curve(self._norm_b, COLOR_BLUE, COLOR_BLUE_FILL)
        elif self._mode == MODE_LUMA:
            draw_channel_curve(self._norm_l, COLOR_LUMA, COLOR_LUMA_FILL)
        elif self._mode == MODE_RED:
            draw_channel_curve(self._norm_r, COLOR_RED, COLOR_RED_FILL)
        elif self._mode == MODE_GREEN:
            draw_channel_curve(self._norm_g, COLOR_GREEN, COLOR_GREEN_FILL)
        elif self._mode == MODE_BLUE:
            draw_channel_curve(self._norm_b, COLOR_BLUE, COLOR_BLUE_FILL)

        self._paint_clipping_badges(painter, rect)

    def _paint_parade(self, painter: QtGui.QPainter, rect: QRectF) -> None:
        """Render DaVinci Resolve style RGB Parade scope with phosphor traces and graticule."""
        scale_w = 16.0  # Width reserved for left scale markings (0, 25, 50, 75, 100)
        plot_rect = rect.adjusted(scale_w, 4.0, -4.0, -4.0)
        gap = 3

        px_w = plot_rect.width()
        px_h = plot_rect.height()
        top_y = plot_rect.top()
        bot_y = plot_rect.bottom()
        part_w = max(8, int((px_w - 2 * gap) // 3))

        # 1. Graticule Scale Markings (0, 25, 50, 75, 100) on the left margin
        painter.setPen(QtGui.QPen(QtGui.QColor(TEXT_DIM), 1.0))
        scale_font = painter.font()
        scale_font.setPointSize(6)
        painter.setFont(scale_font)

        scale_levels = [(1.0, "100"), (0.75, "75"), (0.50, "50"), (0.25, "25"), (0.0, "0")]
        for frac, text in scale_levels:
            sy = top_y + (1.0 - frac) * px_h
            painter.drawText(QRectF(rect.left() + 1.0, sy - 6.0, scale_w - 2.0, 12.0), Qt.AlignRight | Qt.AlignVCenter, text)

        # 2. Horizontal Graticule Grid Lines across all 3 partitions
        for frac, _ in scale_levels:
            sy = top_y + (1.0 - frac) * px_h
            if frac in (0.0, 1.0):
                painter.setPen(QtGui.QPen(COLOR_GRID_SOLID, 1.0))
            elif frac == 0.5:
                painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60, 140), 1.0, Qt.DotLine))
            else:
                painter.setPen(QtGui.QPen(COLOR_GRID, 1.0, Qt.DotLine))
            painter.drawLine(QPointF(plot_rect.left(), sy), QPointF(plot_rect.left() + 3 * part_w + 2 * gap, sy))

        # 3. Vertical Dividers separating R | G and G | B
        painter.setPen(QtGui.QPen(QtGui.QColor(GRAY_3), 1.0))
        x_div1 = plot_rect.left() + part_w + 1.0
        x_div2 = plot_rect.left() + 2 * part_w + gap + 1.0
        painter.drawLine(QPointF(x_div1, top_y), QPointF(x_div1, bot_y))
        painter.drawLine(QPointF(x_div2, top_y), QPointF(x_div2, bot_y))

        # 4. Generate & Draw the RGB Parade waveform trace image
        current_size = (int(px_w), int(px_h))
        if self._parade_image is None or self._parade_cache_size != current_size:
            self._parade_image = compute_parade_image(self._current_image, int(px_w), int(px_h), gap=gap)
            self._parade_cache_size = current_size

        if self._parade_image is not None:
            painter.drawImage(QPointF(plot_rect.left(), top_y), self._parade_image)

        # 5. Channel Badges: R, G, B at top of each partition
        badge_font = painter.font()
        badge_font.setPointSize(8)
        badge_font.setWeight(QtGui.QFont.Weight(FONT_WEIGHT_BOLD))
        painter.setFont(badge_font)

        labels = [(0, "R", COLOR_RED), (1, "G", COLOR_GREEN), (2, "B", COLOR_BLUE)]
        for idx, text, color in labels:
            x_pos = plot_rect.left() + idx * (part_w + gap)
            badge_rect = QRectF(x_pos + 4.0, top_y + 2.0, 12.0, 12.0)
            painter.setPen(QtGui.QPen(color, 1.0))
            painter.drawText(badge_rect, Qt.AlignLeft | Qt.AlignTop, text)

        self._paint_clipping_badges(painter, rect)

    def _paint_clipping_badges(self, painter: QtGui.QPainter, rect: QRectF) -> None:
        if self._shadow_clip > 0.001:
            # Blue shadow clipping triangle in top-left
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QColor(60, 140, 255, 230))
            tri = QtGui.QPolygonF([
                QPointF(rect.left() + 2.0, rect.top() + 2.0),
                QPointF(rect.left() + 9.0, rect.top() + 2.0),
                QPointF(rect.left() + 2.0, rect.top() + 9.0),
            ])
            painter.drawPolygon(tri)

        if self._highlight_clip > 0.001:
            # Red highlight clipping triangle in top-right
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QColor(235, 75, 75, 230))
            tri = QtGui.QPolygonF([
                QPointF(rect.right() - 2.0, rect.top() + 2.0),
                QPointF(rect.right() - 9.0, rect.top() + 2.0),
                QPointF(rect.right() - 2.0, rect.top() + 9.0),
            ])
            painter.drawPolygon(tri)


class HistogramWidget(QtWidgets.QFrame):
    """Complete Scope panel supporting both Histogram and DaVinci Resolve style RGB Parade scopes."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("histogramWidget")
        self.setMinimumHeight(70)
        self.setMaximumHeight(850)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._mode_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._build_ui()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 130)

    def _build_ui(self) -> None:
        self.setStyleSheet(f"""
            QFrame#histogramWidget {{
                background-color: {GRAY_1};
                border: 1px solid {GRAY_3};
                border-radius: 8px;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Header row: Scope Selector Buttons (HIST / PARADE) + Mode Buttons / Legend
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(2, 0, 2, 0)
        header.setSpacing(3)

        # Scope Switcher: HIST / PARADE
        scope_group = QtWidgets.QHBoxLayout()
        scope_group.setContentsMargins(0, 0, 0, 0)
        scope_group.setSpacing(2)

        self.scope_hist_btn = QtWidgets.QPushButton("HIST")
        self.scope_hist_btn.setFixedHeight(16)
        self.scope_hist_btn.setFixedWidth(36)
        self.scope_hist_btn.setCursor(Qt.PointingHandCursor)
        self.scope_hist_btn.setToolTip("Show Histogram scope (Click to switch)")
        self.scope_hist_btn.clicked.connect(lambda: self.set_scope_type(SCOPE_HISTOGRAM))
        scope_group.addWidget(self.scope_hist_btn)

        self.scope_parade_btn = QtWidgets.QPushButton("PARADE")
        self.scope_parade_btn.setFixedHeight(16)
        self.scope_parade_btn.setFixedWidth(46)
        self.scope_parade_btn.setCursor(Qt.PointingHandCursor)
        self.scope_parade_btn.setToolTip("Show RGB Parade scope (DaVinci Resolve style)")
        self.scope_parade_btn.clicked.connect(lambda: self.set_scope_type(SCOPE_PARADE))
        scope_group.addWidget(self.scope_parade_btn)

        header.addLayout(scope_group)
        header.addStretch(1)

        # Histogram Channel Buttons container
        self.channels_container = QtWidgets.QWidget()
        chan_layout = QtWidgets.QHBoxLayout(self.channels_container)
        chan_layout.setContentsMargins(0, 0, 0, 0)
        chan_layout.setSpacing(2)

        for mode_key in ALL_MODES:
            btn = QtWidgets.QPushButton(mode_key)
            btn.setFixedHeight(16)
            btn.setFixedWidth(22 if len(mode_key) <= 2 else (28 if mode_key == MODE_RGB else 30))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"Switch histogram to {mode_key} mode")
            btn.clicked.connect(lambda checked=False, m=mode_key: self.set_mode(m))
            chan_layout.addWidget(btn)
            self._mode_buttons[mode_key] = btn

        header.addWidget(self.channels_container)

        # Parade Legend Label container
        self.parade_legend_label = QtWidgets.QLabel(
            "<span style='color: #eb4b4b; font-weight: bold;'>R</span> · "
            "<span style='color: #41c35f; font-weight: bold;'>G</span> · "
            "<span style='color: #3c8cf5; font-weight: bold;'>B</span>"
        )
        self.parade_legend_label.setStyleSheet("font-size: 9px; background: transparent; border: none;")
        self.parade_legend_label.hide()
        header.addWidget(self.parade_legend_label)

        layout.addLayout(header)

        # Canvas surface
        self.canvas = HistogramCanvas(self)
        layout.addWidget(self.canvas)

        self._update_scope_button_styles()
        self._update_button_styles()

    @property
    def scope_type(self) -> str:
        return self.canvas.scope_type

    @scope_type.setter
    def scope_type(self, val: str) -> None:
        self.set_scope_type(val)

    def set_scope_type(self, scope: str) -> None:
        self.canvas.scope_type = scope
        self._update_scope_button_styles()

    def toggle_scope_type(self) -> str:
        new_scope = self.canvas.toggle_scope_type()
        self._update_scope_button_styles()
        return new_scope

    def set_mode(self, mode_key: str) -> None:
        self.canvas.mode = mode_key
        self._update_button_styles()

    def _on_canvas_mode_cycled(self, new_mode: str) -> None:
        self._update_button_styles()

    def _update_scope_button_styles(self) -> None:
        current_scope = self.canvas.scope_type
        is_hist = current_scope == SCOPE_HISTOGRAM

        active_style = f"""
            QPushButton {{
                background-color: {ACCENT_COLOR_TEXT};
                color: #000;
                font-weight: {FONT_WEIGHT_BOLD};
                font-size: 8px;
                border-radius: 4px;
                border: none;
                padding: 0;
            }}
        """
        inactive_style = f"""
            QPushButton {{
                background-color: {GRAY_2};
                color: {TEXT_DIM};
                font-size: 8px;
                border-radius: 4px;
                border: 1px solid {GRAY_3};
                padding: 0;
            }}
            QPushButton:hover {{
                background-color: {GRAY_3};
                color: {TEXT_HI};
            }}
        """

        self.scope_hist_btn.setStyleSheet(active_style if is_hist else inactive_style)
        self.scope_parade_btn.setStyleSheet(inactive_style if is_hist else active_style)

        # Show channel selector for histogram, show RGB legend for parade
        self.channels_container.setVisible(is_hist)
        self.parade_legend_label.setVisible(not is_hist)

    def _update_button_styles(self) -> None:
        current = self.canvas.mode
        for mode_key, btn in self._mode_buttons.items():
            if mode_key == current:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ACCENT_COLOR_TEXT};
                        color: #000;
                        font-weight: {FONT_WEIGHT_BOLD};
                        font-size: 8px;
                        border-radius: 4px;
                        border: none;
                        padding: 0;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {GRAY_2};
                        color: {TEXT_DIM};
                        font-size: 8px;
                        border-radius: 4px;
                        border: 1px solid {GRAY_3};
                        padding: 0;
                    }}
                    QPushButton:hover {{
                        background-color: {GRAY_3};
                        color: {TEXT_HI};
                    }}
                """)


    def set_image(self, image: np.ndarray | None) -> None:
        self.canvas.set_image(image)

    def clear(self) -> None:
        self.canvas.set_image(None)


__all__ = [
    "HistogramWidget",
    "HistogramCanvas",
    "compute_parade_image",
    "compute_channel_histogram",
    "SCOPE_HISTOGRAM",
    "SCOPE_PARADE",
    "ALL_SCOPES",
    "MODE_RGB",
    "MODE_LUMA",
    "MODE_RED",
    "MODE_GREEN",
    "MODE_BLUE",
    "ALL_MODES",
]
