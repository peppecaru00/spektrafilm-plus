"""Adobe-style startup splash screen for SpektraFilm Plus."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QRectF, Qt

from spektrafilm_gui.icons import get_app_icon

SPLASH_WIDTH = 560
SPLASH_HEIGHT = 330


def _load_splash_logo(size: int = 88) -> QtGui.QPixmap:
    candidates: list[Path] = [
        Path(__file__).resolve().parent / "assets" / "spektrafilm.png",
        Path(__file__).resolve().parent / "assets" / "spektrafilm.ico",
    ]
    if getattr(sys, "frozen", False):
        bundle_dir = Path(getattr(sys, "_MEIPASS", sys.executable)).resolve()
        candidates.extend([
            bundle_dir / "spektrafilm_gui" / "assets" / "spektrafilm.png",
            bundle_dir / "spektrafilm.png",
            bundle_dir / "spektrafilm.ico",
        ])
    for cand in candidates:
        if cand.is_file():
            pix = QtGui.QPixmap(str(cand))
            if not pix.isNull():
                return pix.scaled(
                    size,
                    size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
    icon = get_app_icon()
    if not icon.isNull():
        return icon.pixmap(size, size)
    return QtGui.QPixmap()


class StartupSplashScreen(QtWidgets.QWidget):
    """Frameless Adobe-style startup splash screen showing app icon, stages, and progress bar."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        flags = (
            Qt.SplashScreen
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )
        super().__init__(parent, flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setFixedSize(SPLASH_WIDTH, SPLASH_HEIGHT)

        self._status_text = "Starting SpektraFilm Plus..."
        self._detail_text = "Initializing environment..."
        self._progress = 0.05
        self._logo = _load_splash_logo(92)
        self._version_text = "Version 0.1.0  •  CUDA & Spectral Engine"
        self._center_on_screen()

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def detail_text(self) -> str:
        return self._detail_text

    @property
    def progress(self) -> float:
        return self._progress

    def _center_on_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)

    def set_progress(self, status: str, progress: float | None = None, detail: str = "") -> None:
        self._status_text = status
        if detail:
            self._detail_text = detail
        if progress is not None:
            self._progress = max(0.0, min(1.0, float(progress)))
        self.update()
        self.repaint()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def finish(self, main_window: QtWidgets.QWidget | None = None) -> None:
        self.set_progress("Ready", 1.0, "Launching workspace...")
        self.hide()
        self.close()
        if main_window is not None:
            main_window.raise_()
            main_window.activateWindow()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

        rect = QRectF(8.0, 8.0, float(self.width() - 16), float(self.height() - 16))
        radius = 16.0

        # Outer subtle shadow / glow border
        shadow_pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 160), 2.0)
        painter.setPen(shadow_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(-1.0, -1.0, 1.0, 1.0), radius + 1.0, radius + 1.0)

        # Card background gradient (deep dark studio aesthetic)
        bg_grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        bg_grad.setColorAt(0.0, QtGui.QColor(22, 23, 27, 252))
        bg_grad.setColorAt(0.55, QtGui.QColor(17, 18, 21, 252))
        bg_grad.setColorAt(1.0, QtGui.QColor(13, 14, 17, 254))

        clip_path = QtGui.QPainterPath()
        clip_path.addRoundedRect(rect, radius, radius)
        painter.save()
        painter.setClipPath(clip_path)

        painter.fillPath(clip_path, QtGui.QBrush(bg_grad))

        # Subtle top-right spectral ambient glow (Adobe style)
        glow = QtGui.QRadialGradient(rect.right() - 90.0, rect.top() + 50.0, 230.0)
        glow.setColorAt(0.0, QtGui.QColor(56, 189, 248, 34))
        glow.setColorAt(0.45, QtGui.QColor(244, 114, 182, 18))
        glow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        painter.fillRect(rect, QtGui.QBrush(glow))

        # Top spectral accent bar
        bar_rect = QRectF(rect.left(), rect.top(), rect.width(), 3.5)
        accent_grad = QtGui.QLinearGradient(bar_rect.topLeft(), bar_rect.topRight())
        accent_grad.setColorAt(0.0, QtGui.QColor("#ef4444"))   # Red spectral
        accent_grad.setColorAt(0.33, QtGui.QColor("#f59e0b"))  # Amber
        accent_grad.setColorAt(0.66, QtGui.QColor("#10b981"))  # Green
        accent_grad.setColorAt(1.0, QtGui.QColor("#3b82f6"))   # Blue
        painter.fillRect(bar_rect, QtGui.QBrush(accent_grad))

        # Left column: App Icon
        logo_x = rect.left() + 38.0
        logo_y = rect.top() + 50.0
        if not self._logo.isNull():
            painter.drawPixmap(int(logo_x), int(logo_y), self._logo)

        # Right column: Title, Subtitle, Version
        text_left = logo_x + 114.0
        text_width = rect.right() - text_left - 32.0

        # Product Title
        title_font = QtGui.QFont("Segoe UI", 22, QtGui.QFont.Bold)
        title_font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 0.4)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#f5f6f8"))
        painter.drawText(
            QRectF(text_left, rect.top() + 48.0, text_width, 36.0),
            int(Qt.AlignLeft | Qt.AlignVCenter),
            "SpektraFilm Plus",
        )

        # Product Subtitle
        sub_font = QtGui.QFont("Segoe UI", 10, QtGui.QFont.DemiBold)
        painter.setFont(sub_font)
        painter.setPen(QtGui.QColor("#9ca3af"))
        painter.drawText(
            QRectF(text_left, rect.top() + 86.0, text_width, 22.0),
            int(Qt.AlignLeft | Qt.AlignVCenter),
            "Analog Spectral Film & Darkroom Print Simulation",
        )

        # Version & Engine Tag
        ver_font = QtGui.QFont("Segoe UI", 8)
        painter.setFont(ver_font)
        painter.setPen(QtGui.QColor("#6b7280"))
        painter.drawText(
            QRectF(text_left, rect.top() + 110.0, text_width, 18.0),
            int(Qt.AlignLeft | Qt.AlignVCenter),
            self._version_text,
        )

        # Horizontal divider
        div_y = rect.top() + 168.0
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 18), 1.0))
        painter.drawLine(
            QtCore.QPointF(rect.left() + 38.0, div_y),
            QtCore.QPointF(rect.right() - 38.0, div_y),
        )

        # Loading stage information (Adobe style bottom section)
        status_font = QtGui.QFont("Segoe UI", 10, QtGui.QFont.DemiBold)
        painter.setFont(status_font)
        painter.setPen(QtGui.QColor("#e5e7eb"))
        painter.drawText(
            QRectF(rect.left() + 38.0, div_y + 20.0, rect.width() - 130.0, 22.0),
            int(Qt.AlignLeft | Qt.AlignVCenter),
            self._status_text,
        )

        # Percentage readout on right
        pct_text = f"{int(round(self._progress * 100))}%"
        pct_font = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold)
        painter.setFont(pct_font)
        painter.setPen(QtGui.QColor("#9ca3af"))
        painter.drawText(
            QRectF(rect.right() - 92.0, div_y + 20.0, 54.0, 22.0),
            int(Qt.AlignRight | Qt.AlignVCenter),
            pct_text,
        )

        # Secondary detail text
        detail_font = QtGui.QFont("Segoe UI", 8)
        painter.setFont(detail_font)
        painter.setPen(QtGui.QColor("#6b7280"))
        painter.drawText(
            QRectF(rect.left() + 38.0, div_y + 44.0, rect.width() - 76.0, 18.0),
            int(Qt.AlignLeft | Qt.AlignVCenter),
            self._detail_text,
        )

        # Progress bar track
        track_rect = QRectF(rect.left() + 38.0, div_y + 74.0, rect.width() - 76.0, 5.0)
        track_path = QtGui.QPainterPath()
        track_path.addRoundedRect(track_rect, 2.5, 2.5)
        painter.fillPath(track_path, QtGui.QColor(255, 255, 255, 22))

        # Progress bar fill
        fill_width = max(8.0, track_rect.width() * self._progress)
        fill_rect = QRectF(track_rect.left(), track_rect.top(), fill_width, track_rect.height())
        fill_path = QtGui.QPainterPath()
        fill_path.addRoundedRect(fill_rect, 2.5, 2.5)
        fill_grad = QtGui.QLinearGradient(track_rect.topLeft(), track_rect.topRight())
        fill_grad.setColorAt(0.0, QtGui.QColor("#3b82f6"))
        fill_grad.setColorAt(0.5, QtGui.QColor("#60a5fa"))
        fill_grad.setColorAt(1.0, QtGui.QColor("#34d399"))
        painter.fillPath(fill_path, QtGui.QBrush(fill_grad))

        # Footer copyright / info
        footer_font = QtGui.QFont("Segoe UI", 8)
        painter.setFont(footer_font)
        painter.setPen(QtGui.QColor("#4b5563"))
        painter.drawText(
            QRectF(rect.left() + 38.0, rect.bottom() - 32.0, rect.width() - 76.0, 18.0),
            int(Qt.AlignLeft | Qt.AlignVCenter),
            "© IanoLiotta  •  Physical Emulsion & Color Science Pipeline",
        )

        painter.restore()

        # Crisp 1px inner border
        border_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 32), 1.0)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, radius, radius)
        painter.end()


ProgressCallback = Callable[[str, float, str], None]
