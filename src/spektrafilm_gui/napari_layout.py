from __future__ import annotations

from importlib import import_module
import sys
from typing import TYPE_CHECKING, Callable
from typing import Any, cast

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import QRectF, QTimer, Qt

if TYPE_CHECKING:
    import napari

QFrame = QtWidgets.QFrame
QIcon = QtGui.QIcon
QLabel = QtWidgets.QLabel
QMainWindow = QtWidgets.QMainWindow
QPushButton = QtWidgets.QPushButton
QStatusBar = QtWidgets.QStatusBar
QScrollArea = QtWidgets.QScrollArea
QWidget = QtWidgets.QWidget

from spektrafilm_gui.icons import get_app_icon
from spektrafilm_gui.theme import APP_STYLE_SHEET
from spektrafilm_gui.theme_palette import (
    ACCENT_COLOR_TEXT,
    FONT_WEIGHT_BOLD,
    GRAY_0,
    GRAY_3,
    GRAY_18,
    PANEL_BG,
    RADIUS_CARD_NUM,
    SIZE_APP_MARGIN,
    SIZE_FOOTER_BOTTOM_INSET,
    SIZE_FOOTER_MIN_HEIGHT,
    SIZE_FOOTER_ITEM_SPACING,
    SIZE_FOOTER_TOP_SPACING,
    SIZE_PANEL_MARGIN,
    SIZE_SPLITTER_HANDLE_MARGIN_LEFT,
    SIZE_TAB_CONTENT_TOP_MARGIN,
    TEXT_DIM,
    TEXT_HI,
)
from spektrafilm_gui.widgets import WidgetBundle
from spektrafilm_gui.widgets import CollapsibleSection, platform_default_font


DEFAULT_CONTROLS_PANEL_WIDTH = 420
DEFAULT_VIEWER_SPLITTER_WIDTH = 1040
DEFAULT_GALLERY_WIDTH = 250
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
class AppMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._viewer_status_bar: QStatusBar | None = None

    def set_viewer_status_bar(self, status_bar: QStatusBar) -> None:
        self._viewer_status_bar = status_bar

    def statusBar(self) -> QStatusBar:  # noqa: N802 - Qt API name
        if self._viewer_status_bar is not None:
            return self._viewer_status_bar
        return super().statusBar()

def _get_current_stylesheet() -> str:
    try:
        return str(import_module('napari.qt').get_current_stylesheet())
    except (ImportError, AttributeError):
        return ''


def _request_dark_title_bar(window: QWidget) -> bool:
    if sys.platform != 'win32':
        return False

    try:
        hwnd = int(window.winId())
    except (AttributeError, TypeError, ValueError):
        return False

    if hwnd == 0:
        return False

    try:
        ctypes = import_module('ctypes')
        dwmapi = ctypes.windll.dwmapi
        set_window_attribute = dwmapi.DwmSetWindowAttribute
    except (ImportError, AttributeError):
        return False

    value = ctypes.c_int(1)
    value_size = ctypes.sizeof(value)
    for attribute in (_DWMWA_USE_IMMERSIVE_DARK_MODE, _DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1):
        try:
            result = set_window_attribute(hwnd, attribute, ctypes.byref(value), value_size)
        except (OSError, AttributeError, TypeError, ValueError):
            continue
        if result == 0:
            return True
    return False
def _canvas_background_color(*, gray_18_canvas: bool) -> str:
    return GRAY_18 if gray_18_canvas else GRAY_0


def set_canvas_background(viewer: napari.Viewer, *, gray_18_canvas: bool) -> None:
    background = _canvas_background_color(gray_18_canvas=gray_18_canvas)

    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is None:
        return

    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is None:
        return

    if hasattr(qt_viewer, 'setStyleSheet'):
        qt_viewer.setStyleSheet(f'background: {background};')

    canvas = getattr(qt_viewer, 'canvas', None)
    if canvas is not None:
        if hasattr(canvas, 'bgcolor'):
            setattr(canvas, 'bgcolor', background)
        native = getattr(canvas, 'native', None)
        if native is not None and hasattr(native, 'setStyleSheet'):
            native.setStyleSheet(f'background: {background};')


def configure_napari_chrome(viewer: napari.Viewer, *, gray_18_canvas: bool = True) -> None:
    qt_window = getattr(viewer.window, '_qt_window', None)
    if qt_window is not None:
        menu_bar = qt_window.menuBar()
        if menu_bar is not None:
            menu_bar.hide()

    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    if qt_viewer is None:
        return

    set_canvas_background(viewer, gray_18_canvas=gray_18_canvas)

    set_welcome_visible = getattr(qt_viewer, 'set_welcome_visible', None)
    if callable(set_welcome_visible):
        set_welcome_visible(False)

    layer_controls = getattr(qt_viewer, 'dockLayerControls', None)
    if layer_controls is not None:
        layer_controls.hide()

    layer_list = getattr(qt_viewer, 'dockLayerList', None)
    if layer_list is not None:
        layer_list.hide()


_HOST_WINDOWS_BY_VIEWER_ID: dict[int, QWidget] = {}


def set_host_window(viewer: napari.Viewer, host_window: QWidget) -> None:
    if viewer is not None:
        _HOST_WINDOWS_BY_VIEWER_ID[id(viewer)] = host_window
        try:
            setattr(viewer, '_agx_host_window', host_window)
        except (AttributeError, TypeError, ValueError):
            pass
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is not None:
        try:
            setattr(viewer_window, '_agx_host_window', host_window)
        except (AttributeError, TypeError, ValueError):
            pass


def _host_window(viewer: napari.Viewer) -> QWidget | None:
    if viewer is None:
        return None
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is not None:
        host_window = getattr(viewer_window, '_agx_host_window', None)
        if host_window is not None:
            return host_window
    direct_host = getattr(viewer, '_agx_host_window', None)
    if direct_host is not None:
        return direct_host
    mapped_host = _HOST_WINDOWS_BY_VIEWER_ID.get(id(viewer))
    if mapped_host is not None:
        return mapped_host
    if viewer_window is not None:
        qt_window = getattr(viewer_window, '_qt_window', None)
        if qt_window is not None:
            return qt_window
    return None


def _strip_viewer_margins(widget: object) -> None:
    """Recursively clear structural layout margins and spacing around the viewer canvas."""
    layout_getter = getattr(widget, 'layout', None)
    if not callable(layout_getter):
        return
    layout = layout_getter()
    if layout is not None and hasattr(layout, 'setContentsMargins'):
        layout.setContentsMargins(0, 0, 0, 0)
        if hasattr(layout, 'setSpacing'):
            layout.setSpacing(0)
    find_children = getattr(widget, 'findChildren', None)
    if not callable(find_children):
        return
    for child in find_children(QtWidgets.QWidget):
        child_layout = child.layout()
        if child_layout is not None and isinstance(child_layout, (QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout)):
            contains_canvas = False
            for i in range(child_layout.count()):
                item = child_layout.itemAt(i)
                w = item.widget() if item is not None else None
                if w is not None and (
                    'Canvas' in w.__class__.__name__
                    or 'WidgetOverlay' in w.__class__.__name__
                    or 'QtViewer' in w.__class__.__name__
                ):
                    contains_canvas = True
                    break
            if contains_canvas:
                child_layout.setContentsMargins(0, 0, 0, 0)
                child_layout.setSpacing(0)


def take_viewer_widget(viewer: napari.Viewer) -> QWidget:
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is None:
        raise RuntimeError('Napari viewer window is not available')

    qt_window = getattr(viewer_window, '_qt_window', None)
    if qt_window is not None:
        take_central_widget = getattr(qt_window, 'takeCentralWidget', None)
        if callable(take_central_widget):
            central_widget = take_central_widget()
            if central_widget is not None:
                _strip_viewer_margins(central_widget)
                return central_widget

    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is not None:
        _strip_viewer_margins(qt_viewer)
        return qt_viewer
    raise RuntimeError('Napari Qt viewer widget is not available')


def _wrap_scrollable(widget: QWidget) -> QScrollArea:
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setSizeAdjustPolicy(QScrollArea.AdjustIgnored)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(widget)
    return scroll


def _build_sidebar(controls_panel: QWidget) -> QFrame:
    sidebar = QtWidgets.QFrame()
    sidebar.setObjectName('sidebarPanel')

    layout = QtWidgets.QVBoxLayout(sidebar)
    layout.setContentsMargins(SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_FOOTER_BOTTOM_INSET)
    layout.setSpacing(0)
    layout.addWidget(controls_panel, 1)
    return sidebar


def _build_controls_tab(*widgets: QWidget) -> QWidget:
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)
    layout.setContentsMargins(6, SIZE_TAB_CONTENT_TOP_MARGIN, 6, 10)
    layout.setSpacing(10)
    layout.setAlignment(Qt.AlignTop)
    for widget in widgets:
        layout.addWidget(widget)
    return tab


def _borrow_layer_list_widget(viewer: napari.Viewer) -> QWidget | None:
    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    layer_list = getattr(qt_viewer, 'dockLayerList', None) if qt_viewer is not None else None
    if layer_list is None or not hasattr(layer_list, 'widget'):
        return None
    widget = layer_list.widget()
    if isinstance(widget, QWidget):
        widget.setStyleSheet(_get_current_stylesheet())
    return widget


def reset_viewer_camera(viewer: napari.Viewer) -> None:
    reset_view = getattr(viewer, 'reset_view', None)
    if callable(reset_view):
        reset_view()


def _viewer_device_pixel_ratio(viewer: napari.Viewer) -> float:
    viewer_window = getattr(viewer, 'window', None)
    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is None:
        return 1.0

    device_pixel_ratio = getattr(qt_viewer, 'devicePixelRatioF', None)
    if callable(device_pixel_ratio):
        try:
            return max(float(device_pixel_ratio()), 0.01)
        except (TypeError, ValueError):
            return 1.0

    if hasattr(qt_viewer, 'devicePixelRatio'):
        try:
            return max(float(qt_viewer.devicePixelRatio()), 0.01)
        except (TypeError, ValueError):
            return 1.0

    return 1.0


def _layer_pixel_world_size(layer: object | None) -> float:
    if layer is None:
        return 1.0

    scale = getattr(layer, 'scale', None)
    if scale is None:
        return 1.0

    if isinstance(scale, (int, float)):
        try:
            return max(abs(float(scale)), 1e-12)
        except (TypeError, ValueError):
            return 1.0

    try:
        scale_values = tuple(scale)
    except TypeError:
        return 1.0

    axis_scales: list[float] = []
    for value in scale_values[:2]:
        try:
            numeric_value = abs(float(value))
        except (TypeError, ValueError):
            continue
        if numeric_value > 0.0:
            axis_scales.append(numeric_value)

    if not axis_scales:
        return 1.0
    return sum(axis_scales) / len(axis_scales)


def _home_view_target_layer(viewer: napari.Viewer) -> object | None:
    layers = getattr(viewer, 'layers', None)
    if layers is None:
        return None

    selection = getattr(layers, 'selection', None)
    active_layer = getattr(selection, 'active', None)
    if active_layer is not None and getattr(active_layer, 'visible', True):
        return active_layer

    for layer in reversed(list(layers)):
        if getattr(layer, 'visible', True):
            return layer
    return None


def set_viewer_zoom_percent(viewer: napari.Viewer, percent: float) -> None:
    viewer_window = getattr(viewer, 'window', None)
    camera = getattr(viewer, 'camera', None)
    if camera is None:
        camera = getattr(viewer_window, 'camera', None)
    if camera is None or not hasattr(camera, 'zoom'):
        return

    pixel_ratio = _viewer_device_pixel_ratio(viewer)
    target_layer = _home_view_target_layer(viewer)
    layer_pixel_world_size = _layer_pixel_world_size(target_layer)
    zoom_scale = max(0.0, float(percent)) / 100.0
    camera.zoom = max((pixel_ratio * zoom_scale) / layer_pixel_world_size, pixel_ratio * 0.01)


class ViewportRoundedOverlay(QtWidgets.QWidget):
    """Transparent overlay painting an antialiased internal rounded border and corner mask."""

    def __init__(self, parent: QtWidgets.QWidget, radius: float = RADIUS_CARD_NUM) -> None:
        super().__init__(parent)
        self._radius = radius
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            rect = QRectF(self.rect())
            r = self._radius

            stroke_rect = rect.adjusted(0.5, 0.5, -0.5, -0.5)
            outer_path = QtGui.QPainterPath()
            outer_path.addRect(rect)
            inner_path = QtGui.QPainterPath()
            inner_path.addRoundedRect(stroke_rect, r, r)
            corner_cutout = outer_path.subtracted(inner_path)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QColor(PANEL_BG))
            painter.drawPath(corner_cutout)

            # 2. Draw subtle crisp internal rounded border
            painter.setPen(QtGui.QPen(QtGui.QColor(GRAY_3), 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(stroke_rect, r, r)
        finally:
            painter.end()


class LoadingSpinner(QtWidgets.QWidget):
    """Smooth animated rotating ring spinner in SpektraFilm accent color."""

    def __init__(self, size: int = 36, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = size
        self._angle: float = 0.0
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._on_tick)

    def _on_tick(self) -> None:
        self._angle = (self._angle + 10.0) % 360.0
        self.update()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            w = float(self.width())
            h = float(self.height())
            stroke_width = 3.2
            inset = stroke_width / 2.0 + 1.0
            rect = QRectF(inset, inset, w - 2 * inset, h - 2 * inset)

            track_pen = QtGui.QPen(QtGui.QColor(GRAY_3), stroke_width)
            track_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(track_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(rect)

            arc_pen = QtGui.QPen(QtGui.QColor(ACCENT_COLOR_TEXT), stroke_width)
            arc_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(arc_pen)
            span_angle = int(100 * 16)
            start_angle = int(-self._angle * 16)
            painter.drawArc(rect, start_angle, span_angle)
        finally:
            painter.end()


class ViewportLoadingOverlay(QtWidgets.QWidget):
    """Centered HUD loading card displayed over the viewport during operations."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setAlignment(Qt.AlignCenter)

        self._card = QtWidgets.QFrame(self)
        self._card.setObjectName("viewportLoadingCard")
        self._card.setStyleSheet(
            f"""
            QFrame#viewportLoadingCard {{
                background-color: rgba(18, 18, 18, 0.94);
                border: 1px solid {GRAY_3};
                border-radius: 10px;
            }}
            """
        )

        card_layout = QtWidgets.QVBoxLayout(self._card)
        card_layout.setContentsMargins(28, 22, 28, 22)
        card_layout.setSpacing(12)
        card_layout.setAlignment(Qt.AlignCenter)

        self._spinner = LoadingSpinner(size=36, parent=self._card)
        card_layout.addWidget(self._spinner, 0, Qt.AlignCenter)

        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(4)
        text_layout.setAlignment(Qt.AlignCenter)

        self._title_label = QtWidgets.QLabel("Loading RAW...", self._card)
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setStyleSheet(
            f"color: {TEXT_HI}; font-size: 13px; font-weight: {FONT_WEIGHT_BOLD}; background: transparent; border: none;"
        )
        text_layout.addWidget(self._title_label)

        self._detail_label = QtWidgets.QLabel("", self._card)
        self._detail_label.setAlignment(Qt.AlignCenter)
        self._detail_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; background: transparent; border: none;"
        )
        self._detail_label.hide()
        text_layout.addWidget(self._detail_label)

        card_layout.addLayout(text_layout)
        outer_layout.addWidget(self._card, 0, Qt.AlignCenter)
        self.hide()

    def set_text(self, title: str, detail: str = "") -> None:
        self._title_label.setText(title)
        if detail:
            self._detail_label.setText(detail)
            self._detail_label.show()
        else:
            self._detail_label.setText("")
            self._detail_label.hide()
        self._card.adjustSize()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 90))
        finally:
            painter.end()


class ViewportContainer(QtWidgets.QFrame):
    """Container frame wrapping the viewer widget with an internal rounded border."""

    def __init__(
        self,
        viewer_widget: QWidget,
        radius: float = RADIUS_CARD_NUM,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName('viewportContainer')
        self._radius = radius
        self.viewer_widget = viewer_widget

        _strip_viewer_margins(viewer_widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(viewer_widget, 1)

        self._overlay = ViewportRoundedOverlay(self, radius=radius)
        self._overlay.setGeometry(self.rect())

        self._loading_overlay = ViewportLoadingOverlay(self)
        self._loading_overlay.setGeometry(self.rect())

    def show_loading(self, title: str, detail: str = "") -> None:
        self._loading_overlay.set_text(title, detail)
        self._loading_overlay.setGeometry(self.rect())
        self._loading_overlay.show()
        self._loading_overlay.raise_()
        QtWidgets.QApplication.processEvents()

    def hide_loading(self) -> None:
        self._loading_overlay.hide()
        QtWidgets.QApplication.processEvents()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())
        self._overlay.raise_()
        self._loading_overlay.setGeometry(self.rect())
        if self._loading_overlay.isVisible():
            self._loading_overlay.raise_()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        _strip_viewer_margins(self.viewer_widget)
        self._overlay.setGeometry(self.rect())
        self._overlay.raise_()
        self._loading_overlay.setGeometry(self.rect())
        if self._loading_overlay.isVisible():
            self._loading_overlay.raise_()


def _build_viewer_panel(
    viewer_widget: QWidget,
    status_bar: QStatusBar,
    *,
    gallery_widget: QWidget | None = None,
    on_rotate_ccw: Callable[[], None] | None = None,
    on_rotate_cw: Callable[[], None] | None = None,
    on_zoom_100: Callable[[], None] | None = None,
    on_zoom_200: Callable[[], None] | None = None,
    on_zoom_400: Callable[[], None] | None = None,
    on_home_view: Callable[[], None] | None = None,
    on_undo: Callable[[], None] | None = None,
    on_redo: Callable[[], None] | None = None,
) -> QFrame:
    panel = QtWidgets.QFrame()
    panel.setObjectName('viewerPanel')
    divider_gap = int(SIZE_SPLITTER_HANDLE_MARGIN_LEFT.removesuffix('px'))

    status_container = QtWidgets.QWidget()
    status_layout = QtWidgets.QHBoxLayout(status_container)
    status_layout.setContentsMargins(0, 1, 0, 1)
    status_layout.setSpacing(SIZE_FOOTER_ITEM_SPACING)
    status_bar.setContentsMargins(0, 0, 0, 0)
    status_bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    status_layout.addWidget(status_bar, 1)

    undo_button = QPushButton('↺ undo')
    undo_button.setObjectName('undoButton')
    undo_button.setToolTip('Undo parameter change (Ctrl+Z)')
    undo_button.setEnabled(False)
    if on_undo is not None:
        undo_button.clicked.connect(on_undo)

    redo_button = QPushButton('↻ redo')
    redo_button.setObjectName('redoButton')
    redo_button.setToolTip('Redo parameter change (Ctrl+Y, Ctrl+Shift+Z)')
    redo_button.setEnabled(False)
    if on_redo is not None:
        redo_button.clicked.connect(on_redo)

    rotate_ccw_button = QPushButton('ccw rotate')
    rotate_ccw_button.setObjectName('rotateCcwButton')
    if on_rotate_ccw is not None:
        rotate_ccw_button.clicked.connect(on_rotate_ccw)

    rotate_cw_button = QPushButton('cw rotate')
    rotate_cw_button.setObjectName('rotateCwButton')
    if on_rotate_cw is not None:
        rotate_cw_button.clicked.connect(on_rotate_cw)

    zoom_100_button = QPushButton('100%')
    zoom_100_button.setObjectName('zoom100Button')
    zoom_100_button.setToolTip('Pixel of the screen mapped 1 to 1 to the image pixel')
    if on_zoom_100 is not None:
        zoom_100_button.clicked.connect(on_zoom_100)

    zoom_200_button = QPushButton('200%')
    zoom_200_button.setObjectName('zoom200Button')
    zoom_200_button.setToolTip('2 screen pixels mapped to 1 image pixel')
    if on_zoom_200 is not None:
        zoom_200_button.clicked.connect(on_zoom_200)

    zoom_400_button = QPushButton('400%')
    zoom_400_button.setObjectName('zoom400Button')
    zoom_400_button.setToolTip('4 screen pixels mapped to 1 image pixel')
    if on_zoom_400 is not None:
        zoom_400_button.clicked.connect(on_zoom_400)

    home_button = QPushButton('reset view')
    home_button.setObjectName('homeViewButton')
    if on_home_view is not None:
        home_button.clicked.connect(on_home_view)
    row_height = 26
    undo_button.setFixedHeight(row_height)
    redo_button.setFixedHeight(row_height)
    rotate_ccw_button.setFixedHeight(row_height)
    rotate_cw_button.setFixedHeight(row_height)
    zoom_100_button.setFixedHeight(row_height)
    zoom_200_button.setFixedHeight(row_height)
    zoom_400_button.setFixedHeight(row_height)
    home_button.setFixedHeight(row_height)
    status_bar.setFixedHeight(row_height)
    status_container.setFixedHeight(row_height + 2)
    gpu_badge = QLabel()
    gpu_badge.setObjectName('gpuBadge')
    gpu_badge.setFixedHeight(row_height)
    gpu_badge.setAlignment(Qt.AlignCenter)
    try:
        from spektrafilm.cuda.device import is_cuda_available, get_device_info
        if is_cuda_available():
            dev_info = get_device_info()
            dev_name = dev_info.get("device_name", "CUDA GPU")
            mem_gb = dev_info.get("total_memory_gb", 0.0)
            gpu_badge.setText(f"⚡ GPU: {dev_name}")
            gpu_badge.setStyleSheet(
                "color: #4ade80; font-size: 11px; font-weight: 600; padding: 0 8px; "
                "background-color: rgba(74, 222, 128, 0.12); border: 1px solid rgba(74, 222, 128, 0.3); "
                "border-radius: 4px;"
            )
            gpu_badge.setToolTip(
                f"NVIDIA CUDA acceleration ACTIVE\n"
                f"Device: {dev_name}\n"
                f"VRAM: {mem_gb:.1f} GB\n\n"
                f"Note: In Windows Task Manager, CUDA compute activity appears under the 'Compute_0' or 'CUDA' engine graph, "
                f"not the default '3D' graph."
            )
        else:
            dev_info = get_device_info()
            err = dev_info.get("error")
            gpu_badge.setText("⚙ CPU")
            gpu_badge.setStyleSheet(
                "color: #facc15; font-size: 11px; font-weight: 500; padding: 0 8px; "
                "background-color: rgba(250, 204, 21, 0.12); border: 1px solid rgba(250, 204, 21, 0.3); "
                "border-radius: 4px;"
            )
            tooltip = "GPU acceleration not available (using multi-core CPU)"
            if err:
                tooltip += f"\nReason: {err}"
            gpu_badge.setToolTip(tooltip)
    except Exception:
        gpu_badge.setText("⚙ CPU")
        gpu_badge.setStyleSheet("color: #facc15; font-size: 11px; padding: 0 8px;")

    status_layout.addWidget(gpu_badge)
    status_layout.addWidget(undo_button)
    status_layout.addWidget(redo_button)
    status_layout.addWidget(rotate_ccw_button)
    status_layout.addWidget(rotate_cw_button)
    status_layout.addWidget(zoom_100_button)
    status_layout.addWidget(zoom_200_button)
    status_layout.addWidget(zoom_400_button)
    status_layout.addWidget(home_button)

    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, divider_gap, max(8, SIZE_FOOTER_BOTTOM_INSET))
    layout.setSpacing(0)
    if isinstance(viewer_widget, QWidget):
        viewport_container = ViewportContainer(viewer_widget, radius=RADIUS_CARD_NUM, parent=panel)
        layout.addWidget(viewport_container, 1)
        panel.viewport_container = viewport_container
    else:
        layout.addWidget(viewer_widget, 1)
    layout.addSpacing(SIZE_FOOTER_TOP_SPACING)
    layout.addWidget(status_container)
    panel.viewer_widget = viewer_widget
    panel.undo_button = undo_button
    panel.redo_button = redo_button
    panel.gpu_badge = gpu_badge
    return panel


def build_controls_panel(viewer: napari.Viewer, widgets: WidgetBundle) -> QWidget:
    panel = QtWidgets.QTabWidget()
    panel.setObjectName('controlsTabWidget')
    panel.setDocumentMode(True)
    panel.setUsesScrollButtons(False)
    panel.tabBar().setDrawBase(False)
    panel.addTab(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.filepicker,
                widgets.load_raw,
                widgets.preview_crop,
                widgets.input_image,
                widgets.camera,
                widgets.simulation,
                widgets.exposure_control,
                widgets.enlarger,
                widgets.scanner,
                widgets.output,
            ),
        ),
        'MAIN',
    )
    panel.addTab(
        _wrap_scrollable(_build_controls_tab(widgets.halation, widgets.couplers, widgets.grain, widgets.camera_diffusion)),
        'FILM',
    )
    panel.addTab(_wrap_scrollable(_build_controls_tab(widgets.chemistry, widgets.glare, widgets.preflashing, widgets.enlarger_diffusion)), 'PRINT')
    panel.addTab(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.spectral_upsampling,
                widgets.input_gamut_compress,
                widgets.output_gamut_compress,
                widgets.compute_precision,
                widgets.tune,
                widgets.special,
            ),
        ),
        'ADVANCED',
    )

    napari_layers_content = QtWidgets.QWidget()
    napari_layers_content_layout = QtWidgets.QVBoxLayout(napari_layers_content)
    napari_layers_content_layout.setContentsMargins(0, 0, 0, 0)
    napari_layers_content_layout.setSpacing(6)

    layer_list_widget = _borrow_layer_list_widget(viewer)
    if layer_list_widget is not None:
        napari_layers_content_layout.addWidget(layer_list_widget)

    panel.addTab(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.gui_config,
                widgets.display,
                CollapsibleSection('napari layers', napari_layers_content, expanded=False),
            ),
        ),
        'CONFIG',
    )

    container = QtWidgets.QWidget()
    container_layout = QtWidgets.QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(4)
    container_layout.addWidget(panel, 1)
    container_layout.addWidget(widgets.simulation.action_bar())

    return container


def build_main_window(
    viewer: napari.Viewer,
    controls_panel: QWidget,
    *,
    gallery_widget: QWidget | None = None,
    on_rotate_ccw: Callable[[], None] | None = None,
    on_rotate_cw: Callable[[], None] | None = None,
    on_undo: Callable[[], None] | None = None,
    on_redo: Callable[[], None] | None = None,
) -> QMainWindow:
    viewer_widget = take_viewer_widget(viewer)
    status_bar = QtWidgets.QStatusBar()
    status_bar.setSizeGripEnabled(False)

    main_window = AppMainWindow()
    main_window.setWindowTitle('SpektraFilm Plus')
    main_window.setWindowIcon(get_app_icon())
    total_width = DEFAULT_CONTROLS_PANEL_WIDTH + DEFAULT_VIEWER_SPLITTER_WIDTH
    if gallery_widget is not None:
        total_width += DEFAULT_GALLERY_WIDTH
    main_window.resize(total_width, 980)
    main_window.setFont(platform_default_font())
    main_window.setStyleSheet(APP_STYLE_SHEET)
    main_window.set_viewer_status_bar(status_bar)
    set_host_window(viewer, main_window)

    splitter = QtWidgets.QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(4)
    if gallery_widget is not None:
        gallery_widget.setMinimumWidth(220)
        gallery_widget.setMaximumWidth(450)
        splitter.addWidget(gallery_widget)
    viewer_panel = _build_viewer_panel(
        viewer_widget,
        status_bar,
        on_rotate_ccw=on_rotate_ccw,
        on_rotate_cw=on_rotate_cw,
        on_zoom_100=lambda: set_viewer_zoom_percent(viewer, 100.0),
        on_zoom_200=lambda: set_viewer_zoom_percent(viewer, 200.0),
        on_zoom_400=lambda: set_viewer_zoom_percent(viewer, 400.0),
        on_home_view=lambda: reset_viewer_camera(viewer),
        on_undo=on_undo,
        on_redo=on_redo,
    )
    viewer_panel.setMinimumWidth(450)
    main_window.viewer_panel = viewer_panel
    main_window.undo_button = getattr(viewer_panel, 'undo_button', None)
    main_window.redo_button = getattr(viewer_panel, 'redo_button', None)
    splitter.addWidget(viewer_panel)
    sidebar = _build_sidebar(controls_panel)
    sidebar.setMinimumWidth(380)
    sidebar.setMaximumWidth(520)
    splitter.addWidget(sidebar)
    if gallery_widget is not None:
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([DEFAULT_GALLERY_WIDTH, DEFAULT_VIEWER_SPLITTER_WIDTH, DEFAULT_CONTROLS_PANEL_WIDTH])
    else:
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([DEFAULT_VIEWER_SPLITTER_WIDTH, DEFAULT_CONTROLS_PANEL_WIDTH])

    central = QtWidgets.QWidget()
    central.setObjectName('appCentral')
    central_layout = QtWidgets.QHBoxLayout(central)
    central_layout.setContentsMargins(SIZE_APP_MARGIN, SIZE_APP_MARGIN, SIZE_APP_MARGIN, SIZE_APP_MARGIN)
    central_layout.addWidget(splitter, 1)

    main_window.setCentralWidget(central)
    _request_dark_title_bar(main_window)
    main_window.statusBar().showMessage('ready', 3000)
    return main_window


def dialog_parent(viewer: napari.Viewer) -> QWidget | None:
    return _host_window(viewer)


def set_status(viewer: napari.Viewer, message: str, *, timeout_ms: int = 5000) -> None:
    host_window = cast(Any, _host_window(viewer))
    if host_window is None:
        return
    status_bar = host_window.statusBar() if hasattr(host_window, 'statusBar') else None
    if status_bar is not None:
        status_bar.showMessage(message, timeout_ms)


def show_viewer_window(viewer: napari.Viewer) -> None:
    host_window = _host_window(viewer)
    if host_window is not None and hasattr(host_window, 'show'):
        host_window.show()

    app = QtWidgets.QApplication.instance()
    if app is None:
        return

    active_window = app.activeWindow()
    if active_window is not None:
        active_window.showMaximized()
        return

    for window in reversed(app.topLevelWidgets()):
        if window.isVisible():
            window.showMaximized()
            return


def _find_viewport_container(viewer: napari.Viewer | None) -> ViewportContainer | None:
    if viewer is None:
        return None
    host_window = _host_window(viewer)
    if host_window is not None:
        viewer_panel = getattr(host_window, 'viewer_panel', None)
        if viewer_panel is not None:
            container = getattr(viewer_panel, 'viewport_container', None)
            if container is not None and isinstance(container, ViewportContainer):
                return container
        find_child = getattr(host_window, 'findChild', None)
        if callable(find_child):
            found = find_child(ViewportContainer)
            if found is not None:
                return found
    return None


def show_viewport_overlay(viewer: napari.Viewer | None, title: str, detail: str = "") -> None:
    container = _find_viewport_container(viewer)
    if container is not None:
        container.show_loading(title, detail)


def hide_viewport_overlay(viewer: napari.Viewer | None) -> None:
    container = _find_viewport_container(viewer)
    if container is not None:
        container.hide_loading()

