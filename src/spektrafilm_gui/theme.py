from __future__ import annotations

import importlib
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui import theme_palette, theme_styles

DEFAULT_ACCENT_COLOR = '#ee9470'


def get_accent_color() -> str:
    settings = QtCore.QSettings('spektrafilm', 'spektrafilm')
    val = settings.value('ui/accent_color', DEFAULT_ACCENT_COLOR)
    return str(val) if val else DEFAULT_ACCENT_COLOR


def set_accent_color(color_hex: str) -> None:
    settings = QtCore.QSettings('spektrafilm', 'spektrafilm')
    settings.setValue('ui/accent_color', color_hex)
    theme_palette.ACCENT_COLOR_TEXT = color_hex
    theme_palette.TAB_TEXT_ACCENT = color_hex
    theme_palette.CHECKED_BG = color_hex
    theme_palette.TEXT_ACCENT = color_hex
    theme_palette.BOOL_EDITOR_CHECKED_ENABLED = color_hex
    theme_palette.BOOL_EDITOR_BORDER_CHECKED = color_hex


def build_app_stylesheet(accent_color: str | None = None) -> str:
    accent = accent_color or get_accent_color()
    set_accent_color(accent)
    importlib.reload(theme_styles)
    global APP_STYLE_SHEET
    APP_STYLE_SHEET = theme_styles.join_style_sections(
        theme_styles.WINDOW_STYLE,
        theme_styles.TAB_STYLE,
        theme_styles.CONTROL_STYLE,
        theme_styles.CHROME_STYLE,
        getattr(theme_styles, 'SECTION_STYLE', ''),
    )
    return APP_STYLE_SHEET


def apply_accent_color(color_hex: str) -> None:
    new_stylesheet = build_app_stylesheet(color_hex)
    try:
        from spektrafilm_gui.icons import section_header_icon
        section_header_icon.cache_clear()
    except Exception:
        pass

    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.setStyleSheet(new_stylesheet)
        for window in app.topLevelWidgets():
            if hasattr(window, 'setStyleSheet') and not isinstance(window, QtWidgets.QDialog):
                window.setStyleSheet(new_stylesheet)

        # Refresh all custom-drawn widgets across the application
        for widget in app.allWidgets():
            widget.update()
            if hasattr(widget, '_apply_header_icon') and hasattr(widget, '_title_button'):
                widget._apply_header_icon(widget._title_button.text())
            if hasattr(widget, '_update_ui_state') and hasattr(widget, 'list_widget'):
                widget._update_ui_state()
                if widget.list_widget.count() > 0:
                    auto_item = widget.list_widget.item(0)
                    if auto_item:
                        auto_item.setForeground(QtGui.QBrush(QtGui.QColor(color_hex)))
            if hasattr(widget, '_update_swatch_styles') and hasattr(widget, '_buttons'):
                widget._current_color = color_hex
                widget._update_swatch_styles()


APP_STYLE_SHEET = build_app_stylesheet(get_accent_color())


def resolve_theme_qcolor(color_spec: str) -> QtGui.QColor:
    return QtGui.QColor(color_spec)


def resolve_theme_color_name(color_spec: str) -> str:
    return resolve_theme_qcolor(color_spec).name()
