from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QPointF, QRectF, Qt, Signal

from spektrafilm_gui.theme_palette import (
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

COLOR_Y = QtGui.QColor(240, 240, 240)
COLOR_R = QtGui.QColor(255, 85, 85)
COLOR_G = QtGui.QColor(80, 250, 123)
COLOR_B = QtGui.QColor(97, 175, 239)

COLOR_Y_DIM = QtGui.QColor(240, 240, 240, 90)
COLOR_R_DIM = QtGui.QColor(255, 85, 85, 90)
COLOR_G_DIM = QtGui.QColor(80, 250, 123, 90)
COLOR_B_DIM = QtGui.QColor(97, 175, 239, 90)

CHANNEL_COLORS = {
    "Y": (COLOR_Y, COLOR_Y_DIM, "#f0f0f0"),
    "R": (COLOR_R, COLOR_R_DIM, "#ff5555"),
    "G": (COLOR_G, COLOR_G_DIM, "#50fa7b"),
    "B": (COLOR_B, COLOR_B_DIM, "#61afef"),
}


def pchip_interpolate(x_pts: np.ndarray, y_pts: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Monotone Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).

    Guarantees monotonicity without overshoot or ringing, preserving shape
    strictly within the data bounds.
    """
    n = len(x_pts)
    if n == 0:
        return x_eval
    if n == 1:
        return np.full_like(x_eval, y_pts[0])
    if n == 2:
        denom = max(float(x_pts[1] - x_pts[0]), 1e-9)
        slope = (y_pts[1] - y_pts[0]) / denom
        return y_pts[0] + slope * (x_eval - x_pts[0])

    h = np.diff(x_pts)
    delta = np.diff(y_pts) / np.maximum(h, 1e-9)

    d = np.zeros(n, dtype=float)
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            d[k] = 0.0

    d0 = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if d0 * delta[0] <= 0:
        d0 = 0.0
    elif (delta[0] * delta[1] < 0) and (abs(d0) > abs(3 * delta[0])):
        d0 = 3 * delta[0]
    d[0] = d0

    dn = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if dn * delta[-1] <= 0:
        dn = 0.0
    elif (delta[-1] * delta[-2] < 0) and (abs(dn) > abs(3 * delta[-1])):
        dn = 3 * delta[-1]
    d[-1] = dn

    idx = np.searchsorted(x_pts, x_eval, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    x_k = x_pts[idx]
    y_k = y_pts[idx]
    x_k1 = x_pts[idx + 1]
    y_k1 = y_pts[idx + 1]
    h_k = np.maximum(x_k1 - x_k, 1e-9)
    d_k = d[idx]
    d_k1 = d[idx + 1]

    t = (x_eval - x_k) / h_k
    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    return h00 * y_k + h10 * h_k * d_k + h01 * y_k1 + h11 * h_k * d_k1


class CurveCanvas(QtWidgets.QWidget):
    """Interactive canvas plotting YRGB curves with control point manipulation."""

    curves_updated = Signal()
    point_hovered = Signal(float, float)
    point_unhovered = Signal()

    POINT_RADIUS = 5.0
    HIT_RADIUS = 10.0
    MARGIN = 14.0

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setFixedHeight(190)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._active_channel = "Y"
        self._points: dict[str, list[tuple[float, float]]] = {
            "Y": [(0.0, 0.0), (1.0, 1.0)],
            "R": [(0.0, 0.0), (1.0, 1.0)],
            "G": [(0.0, 0.0), (1.0, 1.0)],
            "B": [(0.0, 0.0), (1.0, 1.0)],
        }
        self._selected_point_idx: int = -1
        self._hover_point_idx: int = -1
        self._is_dragging: bool = False

        # Live background histogram data
        self._show_histogram: bool = True
        self._has_histogram: bool = False
        self._norm_r: np.ndarray | None = None
        self._norm_g: np.ndarray | None = None
        self._norm_b: np.ndarray | None = None
        self._norm_y: np.ndarray | None = None

    @property
    def show_histogram(self) -> bool:
        return self._show_histogram

    @show_histogram.setter
    def show_histogram(self, val: bool) -> None:
        val = bool(val)
        if self._show_histogram != val:
            self._show_histogram = val
            self.update()

    def set_histogram_image(self, image: np.ndarray | None) -> None:
        """Update the background live-histogram with newly processed RGB or luminance data."""
        if image is None:
            self._has_histogram = False
            self._norm_r = self._norm_g = self._norm_b = self._norm_y = None
            self.update()
            return

        arr = np.asarray(image)
        if arr.size == 0:
            self._has_histogram = False
            self.update()
            return

        # Ensure at least 3 channels
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]

        # Fast subsampling for multi-megapixel images to keep rendering < 1.5ms
        h, w = arr.shape[:2]
        total_pixels = h * w
        if total_pixels > 350_000:
            step = max(1, int(np.sqrt(total_pixels / 300_000)))
            arr_hist = arr[::step, ::step]
        else:
            arr_hist = arr

        # Normalize to uint8 [0, 255]
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
        # Standard Rec.709 Luminance weights
        y_data = (
            0.2126 * r_data.astype(np.float32)
            + 0.7152 * g_data.astype(np.float32)
            + 0.0722 * b_data.astype(np.float32)
        ).astype(np.uint8)

        def _calc_norm_hist(data_u8: np.ndarray) -> np.ndarray:
            counts = np.bincount(data_u8.ravel(), minlength=256)[:256].astype(np.float32)
            # Square root scaling (photography industry standard for tone curves)
            sq = np.sqrt(counts)
            mx = float(np.max(sq)) if sq.size > 0 else 1.0
            if mx < 1e-6:
                return np.zeros(256, dtype=np.float32)
            return (sq / mx).astype(np.float32)

        self._norm_r = _calc_norm_hist(r_data)
        self._norm_g = _calc_norm_hist(g_data)
        self._norm_b = _calc_norm_hist(b_data)
        self._norm_y = _calc_norm_hist(y_data)
        self._has_histogram = True
        self.update()

    def clear_histogram(self) -> None:
        self.set_histogram_image(None)

    @property
    def active_channel(self) -> str:
        return self._active_channel

    @active_channel.setter
    def active_channel(self, ch: str) -> None:
        if ch in CHANNEL_COLORS and ch != self._active_channel:
            self._active_channel = ch
            self._selected_point_idx = -1
            self._hover_point_idx = -1
            self.update()

    def get_points(self, ch: str) -> list[tuple[float, float]]:
        return list(self._points.get(ch, [(0.0, 0.0), (1.0, 1.0)]))

    def set_points(self, ch: str, points: Sequence[tuple[float, float]]) -> None:
        if not points:
            points = [(0.0, 0.0), (1.0, 1.0)]
        pts = sorted(points, key=lambda p: p[0])
        if pts[0][0] > 0.0:
            pts.insert(0, (0.0, 0.0))
        if pts[-1][0] < 1.0:
            pts.append((1.0, 1.0))
        self._points[ch] = pts
        self.update()

    def reset_channel(self, ch: str) -> None:
        self._points[ch] = [(0.0, 0.0), (1.0, 1.0)]
        self._selected_point_idx = -1
        self.update()
        self.curves_updated.emit()

    def reset_all(self) -> None:
        for ch in self._points:
            self._points[ch] = [(0.0, 0.0), (1.0, 1.0)]
        self._selected_point_idx = -1
        self.update()
        self.curves_updated.emit()

    def evaluate_curve(self, ch: str, x_eval: np.ndarray) -> np.ndarray:
        pts = self._points.get(ch, [(0.0, 0.0), (1.0, 1.0)])
        x_pts = np.array([p[0] for p in pts], dtype=float)
        y_pts = np.array([p[1] for p in pts], dtype=float)
        return pchip_interpolate(x_pts, y_pts, x_eval)

    def _graph_rect(self) -> QRectF:
        m = self.MARGIN
        return QRectF(m, m, max(self.width() - 2 * m, 10), max(self.height() - 2 * m, 10))

    def _norm_to_pixel(self, x: float, y: float) -> QPointF:
        r = self._graph_rect()
        px = r.left() + x * r.width()
        py = r.bottom() - y * r.height()
        return QPointF(px, py)

    def _pixel_to_norm(self, px: float, py: float) -> tuple[float, float]:
        r = self._graph_rect()
        x = np.clip((px - r.left()) / max(r.width(), 1e-9), 0.0, 1.0)
        y = np.clip((r.bottom() - py) / max(r.height(), 1e-9), 0.0, 1.0)
        return float(x), float(y)

    def _find_point_near(self, px: float, py: float) -> int:
        pts = self._points[self._active_channel]
        for idx, (x, y) in enumerate(pts):
            qpt = self._norm_to_pixel(x, y)
            dist = np.hypot(px - qpt.x(), py - qpt.y())
            if dist <= self.HIT_RADIUS:
                return idx
        return -1

    def _paint_histogram_background(self, painter: QtGui.QPainter, rect: QRectF) -> None:
        """Render live tone histogram in the background behind grid lines and curves."""
        painter.save()
        clip_path = QtGui.QPainterPath()
        clip_path.addRoundedRect(rect, 6.0, 6.0)
        painter.setClipPath(clip_path)

        active = self._active_channel
        if active == "Y":
            # Master / Luminance mode: render soft composite of R, G, B channels + Luma
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_r,
                fill_color=QtGui.QColor(255, 85, 85, 28),
                stroke_color=QtGui.QColor(255, 85, 85, 80),
                stroke_width=1.0,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_g,
                fill_color=QtGui.QColor(80, 250, 123, 24),
                stroke_color=QtGui.QColor(80, 250, 123, 80),
                stroke_width=1.0,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_b,
                fill_color=QtGui.QColor(97, 175, 239, 28),
                stroke_color=QtGui.QColor(97, 175, 239, 80),
                stroke_width=1.0,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_y,
                fill_color=QtGui.QColor(240, 240, 240, 35),
                stroke_color=QtGui.QColor(240, 240, 240, 135),
                stroke_width=1.2,
            )
        elif active == "R":
            # Red channel mode: subtle ghost of G & B + prominent Red histogram
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_g,
                fill_color=QtGui.QColor(80, 250, 123, 14),
                stroke_color=QtGui.QColor(80, 250, 123, 40),
                stroke_width=0.8,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_b,
                fill_color=QtGui.QColor(97, 175, 239, 14),
                stroke_color=QtGui.QColor(97, 175, 239, 40),
                stroke_width=0.8,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_r,
                fill_color=QtGui.QColor(255, 85, 85, 55),
                stroke_color=QtGui.QColor(255, 85, 85, 175),
                stroke_width=1.5,
            )
        elif active == "G":
            # Green channel mode: subtle ghost of R & B + prominent Green histogram
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_r,
                fill_color=QtGui.QColor(255, 85, 85, 14),
                stroke_color=QtGui.QColor(255, 85, 85, 40),
                stroke_width=0.8,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_b,
                fill_color=QtGui.QColor(97, 175, 239, 14),
                stroke_color=QtGui.QColor(97, 175, 239, 40),
                stroke_width=0.8,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_g,
                fill_color=QtGui.QColor(80, 250, 123, 50),
                stroke_color=QtGui.QColor(80, 250, 123, 175),
                stroke_width=1.5,
            )
        elif active == "B":
            # Blue channel mode: subtle ghost of R & G + prominent Blue histogram
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_r,
                fill_color=QtGui.QColor(255, 85, 85, 14),
                stroke_color=QtGui.QColor(255, 85, 85, 40),
                stroke_width=0.8,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_g,
                fill_color=QtGui.QColor(80, 250, 123, 14),
                stroke_color=QtGui.QColor(80, 250, 123, 40),
                stroke_width=0.8,
            )
            self._draw_hist_channel(
                painter,
                rect,
                self._norm_b,
                fill_color=QtGui.QColor(97, 175, 239, 50),
                stroke_color=QtGui.QColor(97, 175, 239, 175),
                stroke_width=1.5,
            )

        painter.restore()

    def _draw_hist_channel(
        self,
        painter: QtGui.QPainter,
        rect: QRectF,
        norm: np.ndarray | None,
        fill_color: QtGui.QColor,
        stroke_color: QtGui.QColor,
        stroke_width: float = 1.0,
    ) -> None:
        if norm is None or len(norm) == 0:
            return

        num_bins = len(norm)
        w = rect.width()
        h = rect.height() * 0.92
        bot_y = rect.bottom()
        left_x = rect.left()

        x_coords = left_x + (np.arange(num_bins, dtype=np.float32) / (num_bins - 1.0)) * w
        y_coords = bot_y - norm * h

        fill_path = QtGui.QPainterPath()
        fill_path.moveTo(left_x, bot_y)

        stroke_path = QtGui.QPainterPath()
        stroke_path.moveTo(x_coords[0], y_coords[0])

        for i in range(num_bins):
            fill_path.lineTo(x_coords[i], y_coords[i])
            stroke_path.lineTo(x_coords[i], y_coords[i])

        fill_path.lineTo(rect.right(), bot_y)
        fill_path.closeSubpath()

        painter.setPen(Qt.NoPen)
        painter.fillPath(fill_path, QtGui.QBrush(fill_color))

        painter.setPen(QtGui.QPen(stroke_color, stroke_width))
        painter.drawPath(stroke_path)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self._graph_rect()

        # Canvas background & border
        painter.fillRect(self.rect(), QtGui.QColor(GRAY_1))
        painter.setPen(QtGui.QPen(QtGui.QColor(GRAY_3), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Live background histogram overlay
        if self._show_histogram and self._has_histogram:
            self._paint_histogram_background(painter, rect)

        # 4x4 Grid lines
        grid_pen = QtGui.QPen(QtGui.QColor(48, 48, 48), 1.0, Qt.DotLine)
        painter.setPen(grid_pen)
        for frac in (0.25, 0.5, 0.75):
            x_pos = rect.left() + frac * rect.width()
            y_pos = rect.top() + frac * rect.height()
            painter.drawLine(QPointF(x_pos, rect.top()), QPointF(x_pos, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y_pos), QPointF(rect.right(), y_pos))

        # Identity reference diagonal
        diag_pen = QtGui.QPen(QtGui.QColor(70, 70, 70), 1.0, Qt.DashLine)
        painter.setPen(diag_pen)
        painter.drawLine(QPointF(rect.left(), rect.bottom()), QPointF(rect.right(), rect.top()))

        x_eval = np.linspace(0.0, 1.0, 128)

        # Draw inactive channels first
        for ch in ("Y", "R", "G", "B"):
            if ch == self._active_channel:
                continue
            y_eval = self.evaluate_curve(ch, x_eval)
            path = QtGui.QPainterPath()
            p0 = self._norm_to_pixel(x_eval[0], y_eval[0])
            path.moveTo(p0)
            for xi, yi in zip(x_eval[1:], y_eval[1:]):
                path.lineTo(self._norm_to_pixel(xi, yi))
            _, dim_color, _ = CHANNEL_COLORS[ch]
            painter.setPen(QtGui.QPen(dim_color, 1.5))
            painter.drawPath(path)

        # Draw active channel curve
        active_ch = self._active_channel
        y_eval = self.evaluate_curve(active_ch, x_eval)
        path = QtGui.QPainterPath()
        p0 = self._norm_to_pixel(x_eval[0], y_eval[0])
        path.moveTo(p0)
        for xi, yi in zip(x_eval[1:], y_eval[1:]):
            path.lineTo(self._norm_to_pixel(xi, yi))

        bright_color, _, _ = CHANNEL_COLORS[active_ch]
        painter.setPen(QtGui.QPen(bright_color, 2.4))
        painter.drawPath(path)

        # Draw control point handles for active channel
        pts = self._points[active_ch]
        for idx, (x, y) in enumerate(pts):
            qpt = self._norm_to_pixel(x, y)
            is_selected = idx == self._selected_point_idx
            is_hover = idx == self._hover_point_idx

            if is_selected:
                painter.setPen(QtGui.QPen(bright_color, 1.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(qpt, self.POINT_RADIUS + 3.5, self.POINT_RADIUS + 3.5)

            rad = self.POINT_RADIUS + (1.0 if is_hover else 0.0)
            painter.setPen(QtGui.QPen(QtGui.QColor(GRAY_0), 1.2))
            painter.setBrush(QtGui.QBrush(bright_color))
            painter.drawEllipse(qpt, rad, rad)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            idx = self._find_point_near(event.x(), event.y())
            if idx >= 0:
                self._selected_point_idx = idx
                self._is_dragging = True
            else:
                x, y = self._pixel_to_norm(event.x(), event.y())
                pts = self._points[self._active_channel]
                # Insert new control point
                pts.append((x, y))
                pts.sort(key=lambda p: p[0])
                self._points[self._active_channel] = pts
                self._selected_point_idx = pts.index((x, y))
                self._is_dragging = True
                self.curves_updated.emit()
            self.update()
        elif event.button() == Qt.RightButton:
            # Right-click deletes internal point
            idx = self._find_point_near(event.x(), event.y())
            pts = self._points[self._active_channel]
            if 0 < idx < len(pts) - 1:
                pts.pop(idx)
                self._selected_point_idx = -1
                self.update()
                self.curves_updated.emit()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._is_dragging and self._selected_point_idx >= 0:
            x, y = self._pixel_to_norm(event.x(), event.y())
            pts = self._points[self._active_channel]
            idx = self._selected_point_idx
            n = len(pts)

            if idx == 0:
                # Lock left anchor x=0, allow y in [0, 1]
                pts[0] = (0.0, np.clip(y, 0.0, 1.0))
            elif idx == n - 1:
                # Lock right anchor x=1, allow y in [0, 1]
                pts[-1] = (1.0, np.clip(y, 0.0, 1.0))
            else:
                prev_x = pts[idx - 1][0] + 0.02
                next_x = pts[idx + 1][0] - 0.02
                x_clamped = np.clip(x, prev_x, next_x)
                pts[idx] = (float(x_clamped), float(np.clip(y, 0.0, 1.0)))

            self.point_hovered.emit(pts[idx][0], pts[idx][1])
            self.update()
            self.curves_updated.emit()
        else:
            near_idx = self._find_point_near(event.x(), event.y())
            if near_idx != self._hover_point_idx:
                self._hover_point_idx = near_idx
                self.setCursor(Qt.PointingHandCursor if near_idx >= 0 else Qt.CrossCursor)
                self.update()
            if near_idx >= 0:
                pt = self._points[self._active_channel][near_idx]
                self.point_hovered.emit(pt[0], pt[1])
            else:
                self.point_unhovered.emit()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._is_dragging = False

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        # Double click on internal point deletes it
        idx = self._find_point_near(event.x(), event.y())
        pts = self._points[self._active_channel]
        if 0 < idx < len(pts) - 1:
            pts.pop(idx)
            self._selected_point_idx = -1
            self.update()
            self.curves_updated.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            pts = self._points[self._active_channel]
            idx = self._selected_point_idx
            if 0 < idx < len(pts) - 1:
                pts.pop(idx)
                self._selected_point_idx = -1
                self.update()
                self.curves_updated.emit()
        super().keyPressEvent(event)


def _eval_sublayer_multiplier(x_arr: np.ndarray, fast: float, slow: float) -> np.ndarray:
    """Piecewise smoothstep sublayer modulation around midtone x=0.5.

    Guarantees exactly 1.0 at x=0.5 with zero derivative (C1 continuity),
    smoothly ramping to `fast` in highlights and `slow` in shadows without
    affecting the midtone parameter.
    """
    res = np.ones_like(x_arr, dtype=float)
    hi_mask = x_arr >= 0.5
    lo_mask = ~hi_mask
    if np.any(hi_mask):
        t_hi = (x_arr[hi_mask] - 0.5) * 2.0
        w_hi = t_hi**2 * (3.0 - 2.0 * t_hi)
        res[hi_mask] = 1.0 + w_hi * (fast - 1.0)
    if np.any(lo_mask):
        t_lo = (0.5 - x_arr[lo_mask]) * 2.0
        w_lo = t_lo**2 * (3.0 - 2.0 * t_lo)
        res[lo_mask] = 1.0 + w_lo * (slow - 1.0)
    return res


class YRGBCurveEditor(QtWidgets.QWidget):
    """YRGB curve editor widget communicating directly and live with the chemistry section."""

    chemistry_changed = Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Header: Channel selection buttons + Reset button
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        self._channel_group = QtWidgets.QButtonGroup(self)
        self._channel_group.setExclusive(True)

        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        for ch in ("Y", "R", "G", "B"):
            btn = QtWidgets.QPushButton(ch)
            btn.setCheckable(True)
            btn.setChecked(ch == "Y")
            btn.setFixedWidth(28)
            btn.setFixedHeight(22)
            _, _, hex_color = CHANNEL_COLORS[ch]
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {GRAY_2};
                    color: {hex_color};
                    border: 1px solid {GRAY_3};
                    border-radius: 5px;
                    font-weight: 600;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {GRAY_3};
                }}
                QPushButton:checked {{
                    background-color: {GRAY_4};
                    border: 1px solid {hex_color};
                }}
            """)
            self._channel_group.addButton(btn)
            btn.clicked.connect(lambda checked=False, c=ch: self._on_channel_selected(c))
            header.addWidget(btn)
            self._buttons[ch] = btn

        header.addStretch()

        self.hist_btn = QtWidgets.QPushButton("HIST")
        self.hist_btn.setCheckable(True)
        self.hist_btn.setChecked(True)
        self.hist_btn.setFixedHeight(22)
        self.hist_btn.setFixedWidth(38)
        self.hist_btn.setToolTip("Toggle background live histogram overlay")
        self.hist_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {GRAY_2};
                color: {TEXT_DIM};
                border: 1px solid {GRAY_3};
                border-radius: 5px;
                font-size: 10px;
                font-weight: {FONT_WEIGHT_SEMIBOLD};
            }}
            QPushButton:hover {{
                background-color: {GRAY_3};
                color: {TEXT_HI};
            }}
            QPushButton:checked {{
                background-color: {GRAY_4};
                color: {TEXT_HI};
                border: 1px solid {TEXT_DIM};
            }}
        """)
        self.hist_btn.toggled.connect(lambda checked: setattr(self.canvas, "show_histogram", checked))
        header.addWidget(self.hist_btn)

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setFixedHeight(22)
        reset_btn.setToolTip("Reset the active channel curve to linear (Hold Shift to reset all channels)")
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {GRAY_2};
                color: {TEXT_DIM};
                border: 1px solid {GRAY_3};
                border-radius: 5px;
                padding: 0 6px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {GRAY_3};
                color: {TEXT_HI};
            }}
        """)
        reset_btn.clicked.connect(self._on_reset_clicked)
        header.addWidget(reset_btn)

        root.addLayout(header)

        # Interactive Canvas
        self.canvas = CurveCanvas(self)
        self.canvas.curves_updated.connect(self._on_canvas_curves_updated)
        self.canvas.point_hovered.connect(self._on_point_hovered)
        self.canvas.point_unhovered.connect(self._on_point_unhovered)
        root.addWidget(self.canvas)

        # Footer coordinate label
        self.readout_label = QtWidgets.QLabel("Y Channel  •  Linear")
        self.readout_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; padding: 1px 2px;")
        root.addWidget(self.readout_label)

    def set_histogram_image(self, image: np.ndarray | None) -> None:
        """Update live background histogram in the curve editor canvas."""
        self.canvas.set_histogram_image(image)

    def _on_channel_selected(self, ch: str) -> None:
        self.canvas.active_channel = ch
        self.readout_label.setText(f"{ch} Channel")

    def _on_reset_clicked(self) -> None:
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers & Qt.ShiftModifier:
            self.canvas.reset_all()
        else:
            self.canvas.reset_channel(self.canvas.active_channel)

    def _on_point_hovered(self, x: float, y: float) -> None:
        ch = self.canvas.active_channel
        self.readout_label.setText(f"{ch} Channel  •  In: {x:.2f}  Out: {y:.2f}")

    def _on_point_unhovered(self) -> None:
        ch = self.canvas.active_channel
        self.readout_label.setText(f"{ch} Channel")

    def _on_canvas_curves_updated(self) -> None:
        if self._syncing:
            return
        updates = self.get_chemistry_updates()
        self.chemistry_changed.emit(updates)

    def get_chemistry_updates(self) -> dict[str, float]:
        """Extract and solve exact chemistry morph parameters from current curves."""
        # 1. Master Y curve gives gamma_factor, gamma_factor_fast, gamma_factor_slow
        x_eval = np.array([0.25, 0.5, 0.75], dtype=float)
        y_y = self.canvas.evaluate_curve("Y", x_eval)
        y_lo, y_mid, y_hi = float(y_y[0]), float(y_y[1]), float(y_y[2])

        # Clamp evaluated y values away from 0 and 1 to prevent log singularities
        y_mid = np.clip(y_mid, 1e-4, 1.0 - 1e-4)
        y_lo = np.clip(y_lo, 1e-4, 1.0 - 1e-4)
        y_hi = np.clip(y_hi, 1e-4, 1.0 - 1e-4)

        G_mid = float(np.log(y_mid) / np.log(0.5))
        G_lo = float(np.log(y_lo) / np.log(0.25))
        G_hi = float(np.log(y_hi) / np.log(0.75))

        gf = float(np.clip(G_mid, 0.5, 2.0))

        # Check if user has explicit highlight/shadow control points on Y
        pts_y = self.canvas.get_points("Y")
        if len(pts_y) <= 3:
            g_fast = 1.0
            g_slow = 1.0
        else:
            # At x=0.75, sublayer multiplier is (0.5 + 0.5*g_fast)
            g_fast_est = 2.0 * (G_hi / max(gf, 0.1)) - 1.0
            # At x=0.25, sublayer multiplier is (0.5 + 0.5*g_slow)
            g_slow_est = 2.0 * (G_lo / max(gf, 0.1)) - 1.0

            g_fast = float(np.clip(g_fast_est, 0.5, 2.0))
            g_slow = float(np.clip(g_slow_est, 0.5, 2.0))

        # Effective Gamma_Y at evaluation points [0.25, 0.5, 0.75]
        Gamma_Y_eval = gf * _eval_sublayer_multiplier(x_eval, g_fast, g_slow)

        # 2. Channel curves give per-channel gammas + per-channel fast & slow
        ch_params: dict[str, float] = {}
        for ch, name in (("R", "red"), ("G", "green"), ("B", "blue")):
            y_ch = self.canvas.evaluate_curve(ch, x_eval)
            y_ch_lo, y_ch_mid, y_ch_hi = (
                float(np.clip(y_ch[0], 1e-4, 1.0 - 1e-4)),
                float(np.clip(y_ch[1], 1e-4, 1.0 - 1e-4)),
                float(np.clip(y_ch[2], 1e-4, 1.0 - 1e-4)),
            )

            G_ch_mid = float(np.log(y_ch_mid) / np.log(0.5))
            G_ch_lo = float(np.log(y_ch_lo) / np.log(0.25))
            G_ch_hi = float(np.log(y_ch_hi) / np.log(0.75))

            g_ch = float(np.clip(G_ch_mid / max(Gamma_Y_eval[1], 0.1), 0.5, 2.0))

            pts_ch = self.canvas.get_points(ch)
            if len(pts_ch) <= 3:
                g_ch_fast = 1.0
                g_ch_slow = 1.0
            else:
                Gamma_base_hi = Gamma_Y_eval[2] * g_ch
                Gamma_base_lo = Gamma_Y_eval[0] * g_ch

                g_ch_fast_est = 2.0 * (G_ch_hi / max(Gamma_base_hi, 0.1)) - 1.0
                g_ch_slow_est = 2.0 * (G_ch_lo / max(Gamma_base_lo, 0.1)) - 1.0

                g_ch_fast = float(np.clip(g_ch_fast_est, 0.5, 2.0))
                g_ch_slow = float(np.clip(g_ch_slow_est, 0.5, 2.0))

            ch_params[f"gamma_factor_{name}"] = round(g_ch, 2)
            ch_params[f"gamma_factor_{name}_fast"] = round(g_ch_fast, 2)
            ch_params[f"gamma_factor_{name}_slow"] = round(g_ch_slow, 2)

        return {
            "gamma_factor": round(gf, 2),
            "gamma_factor_fast": round(g_fast, 2),
            "gamma_factor_slow": round(g_slow, 2),
            **ch_params,
        }

    def set_chemistry_params(self, params: object) -> None:
        """Update canvas curves and control points from incoming chemistry parameters."""
        self._syncing = True
        try:
            gf = float(getattr(params, "gamma_factor", 1.0))
            g_fast = float(getattr(params, "gamma_factor_fast", 1.0))
            g_slow = float(getattr(params, "gamma_factor_slow", 1.0))

            gr = float(getattr(params, "gamma_factor_red", 1.0))
            gr_fast = float(getattr(params, "gamma_factor_red_fast", 1.0))
            gr_slow = float(getattr(params, "gamma_factor_red_slow", 1.0))

            gg = float(getattr(params, "gamma_factor_green", 1.0))
            gg_fast = float(getattr(params, "gamma_factor_green_fast", 1.0))
            gg_slow = float(getattr(params, "gamma_factor_green_slow", 1.0))

            gb = float(getattr(params, "gamma_factor_blue", 1.0))
            gb_fast = float(getattr(params, "gamma_factor_blue_fast", 1.0))
            gb_slow = float(getattr(params, "gamma_factor_blue_slow", 1.0))

            def eval_channel_model(x_arr: np.ndarray, ch_gamma: float, ch_fast: float, ch_slow: float) -> np.ndarray:
                Gamma_Y = gf * _eval_sublayer_multiplier(x_arr, g_fast, g_slow)
                Gamma_ch = Gamma_Y * ch_gamma * _eval_sublayer_multiplier(x_arr, ch_fast, ch_slow)
                return np.power(x_arr, Gamma_ch)

            # Build control points for Y:
            if abs(g_fast - 1.0) < 0.02 and abs(g_slow - 1.0) < 0.02:
                if abs(gf - 1.0) < 0.02:
                    pts_y = [(0.0, 0.0), (1.0, 1.0)]
                else:
                    ym = float(eval_channel_model(np.array([0.5]), 1.0, 1.0, 1.0)[0])
                    pts_y = [(0.0, 0.0), (0.5, ym), (1.0, 1.0)]
            else:
                x_pts = np.array([0.25, 0.5, 0.75])
                y_pts = eval_channel_model(x_pts, 1.0, 1.0, 1.0)
                pts_y = [(0.0, 0.0), (0.25, float(y_pts[0])), (0.5, float(y_pts[1])), (0.75, float(y_pts[2])), (1.0, 1.0)]
            self.canvas.set_points("Y", pts_y)

            # Build control points for R, G, B:
            channels_data = [
                ("R", gr, gr_fast, gr_slow),
                ("G", gg, gg_fast, gg_slow),
                ("B", gb, gb_fast, gb_slow),
            ]
            for ch, g_ch, c_fast, c_slow in channels_data:
                is_sublayer_flat = abs(c_fast - 1.0) < 0.02 and abs(c_slow - 1.0) < 0.02
                is_y_sublayer_flat = abs(g_fast - 1.0) < 0.02 and abs(g_slow - 1.0) < 0.02
                is_linear = (
                    abs(g_ch - 1.0) < 0.02
                    and abs(gf - 1.0) < 0.02
                    and is_sublayer_flat
                    and is_y_sublayer_flat
                )
                if is_linear:
                    pts = [(0.0, 0.0), (1.0, 1.0)]
                elif is_sublayer_flat and is_y_sublayer_flat:
                    ym = float(eval_channel_model(np.array([0.5]), g_ch, c_fast, c_slow)[0])
                    pts = [(0.0, 0.0), (0.5, ym), (1.0, 1.0)]
                else:
                    x_pts = np.array([0.25, 0.5, 0.75])
                    y_pts = eval_channel_model(x_pts, g_ch, c_fast, c_slow)
                    pts = [(0.0, 0.0), (0.25, float(y_pts[0])), (0.5, float(y_pts[1])), (0.75, float(y_pts[2])), (1.0, 1.0)]
                self.canvas.set_points(ch, pts)

            self.canvas.update()
        finally:
            self._syncing = False
