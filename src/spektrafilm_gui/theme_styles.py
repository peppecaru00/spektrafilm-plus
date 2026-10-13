from pathlib import Path

from spektrafilm_gui.theme_palette import (
    ACCENT_COLOR_TEXT,
    BASE_BG,
    BUTTON_ACCENT_BG,
    BUTTON_ACCENT_BG_HOVER,
    BUTTON_ACCENT_BG_PRESSED,
    CHECKED_BG,
    CONTROL_BG,
    CONTROL_BG_HOVER,
    CONTROL_BG_PRESSED,
    CONTROL_MENU_BG,
    CONTROL_SELECTION_BG,
    DISABLED_BG,
    FONT_WEIGHT_BOLD,
    FONT_WEIGHT_SEMIBOLD,
    GRAY_1,
    GRAY_3,
    GRAY_4,
    HEADER_BG,
    PANEL_BG,
    PANEL_BG_ACTIVE,
    RADIUS_BUTTON,
    RADIUS_CARD,
    RADIUS_COMPACT,
    RADIUS_CONTROL,
    RADIUS_INDICATOR,
    RADIUS_PANEL,
    RADIUS_POPUP,
    RADIUS_SCROLLBAR,
    RADIUS_TAB,
    SCROLLBAR_HANDLE_BG,
    SIZE_BUTTON_MIN_HEIGHT,
    SIZE_BUTTON_PADDING,
    SIZE_CHECKBOX_INDICATOR,
    SIZE_CHECKBOX_SPACING,
    SIZE_COMBO_DROPDOWN_WIDTH,
    SIZE_COMPACT_BUTTON_MIN_HEIGHT,
    SIZE_CONTROL_MIN_HEIGHT,
    SIZE_CONTROL_PADDING,
    SIZE_FOOTER_MIN_HEIGHT,
    SIZE_FORM_SPACING,
    SIZE_HEADER_PADDING,
    SIZE_SCROLLBAR_HANDLE_MIN,
    SIZE_SPLITTER_HANDLE_MARGIN_LEFT,
    SIZE_SPLITTER_HANDLE_WIDTH,
    SIZE_SPIN_BUTTON_WIDTH,
    SIZE_TAB_MARGIN_RIGHT,
    SIZE_TAB_PADDING,
    SIZE_TAB_STRIP_OFFSET,
    SIZE_TOOLBUTTON_PADDING,
    SIZE_TOOLBUTTON_POPUP_PADDING_RIGHT,
    SPLITTER_BG,
    STATUS_BG,
    TAB_SELECTED_BG,
    TAB_TEXT_ACCENT,
    TEXT_ACCENT,
    TEXT_BRIGHT,
    TEXT_CONTROL,
    TEXT_HEADER,
    TEXT_HEADER_SECTION,
    TEXT_HI,
    TEXT_PRIMARY,
    TEXT_SELECTION_BG,
    TEXT_SOFT,
    TEXT_STATUS,
)

CHECKMARK_ICON_PATH = (Path(__file__).parent / "assets" / "checkmark.png").resolve().as_posix()


def join_style_sections(*sections: str) -> str:
    return '\n\n'.join(section.strip() for section in sections if section.strip())


WINDOW_STYLE = f"""
QMainWindow {{
    background: {BASE_BG};
    color: {TEXT_PRIMARY};
}}

QWidget#appCentral {{
    background: {BASE_BG};
}}

QWidget {{
    background: {BASE_BG};
    color: {TEXT_PRIMARY};
}}

QFrame#sidebarPanel,
QFrame#viewerPanel,
QWidget#galleryFilmstrip {{
    background: {PANEL_BG};
    border: none;
    border-radius: {RADIUS_PANEL};
}}

QFrame#viewportContainer {{
    background: {PANEL_BG};
    border: none;
    border-radius: {RADIUS_CARD};
}}

QFrame {{
    background: {PANEL_BG};
}}

QLabel#sidebarEyebrow {{
    color: {TEXT_HEADER};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    letter-spacing: 0.08em;
    text-transform: lowercase;
}}

"""

TAB_STYLE = f"""
QTabWidget,
QTabWidget#controlsTabWidget {{
    background: {BASE_BG};
}}

QTabWidget::pane,
QTabWidget#controlsTabWidget::pane {{
    border: none;
    border-radius: {RADIUS_CARD};
    background: {BASE_BG};
    top: {SIZE_TAB_STRIP_OFFSET};
}}

QTabWidget::tab-bar {{
    alignment: left;
}}

QTabBar {{
    background: {BASE_BG};
}}

QTabWidget#controlsTabWidget > QWidget {{
    background: {BASE_BG};
}}

QTabWidget#controlsTabWidget QStackedWidget > QWidget {{
    background: {CONTROL_BG};
    border-radius: {RADIUS_CARD};
}}

QTabBar::tab {{
    background: {BASE_BG};
    color: {TEXT_SOFT};
    border: none;
    border-radius: {RADIUS_TAB};
    font-weight: {FONT_WEIGHT_BOLD};
    padding: {SIZE_TAB_PADDING};
    margin-right: {SIZE_TAB_MARGIN_RIGHT};
    min-height: {SIZE_CONTROL_MIN_HEIGHT};
}}

QTabBar::tab:selected {{
    background: {TAB_SELECTED_BG};
    color: {TAB_TEXT_ACCENT};
    font-weight: {FONT_WEIGHT_BOLD};
}}

QTabBar::tab:hover {{
    background: {PANEL_BG_ACTIVE};
}}

QToolButton[role="sectionToggle"] {{
    color: {TEXT_ACCENT};
    background: transparent;
    padding: 0;
    min-height: 0;
}}

QToolButton[role="sectionToggle"]:hover,
QToolButton[role="sectionToggle"]:pressed {{
    background: transparent;
}}

QPushButton[role="accentAction"] {{
    background: {BUTTON_ACCENT_BG};
    color: {TEXT_BRIGHT};
    font-weight: {FONT_WEIGHT_BOLD};
    border-radius: {RADIUS_BUTTON};
}}

QPushButton[role="accentAction"]:hover {{
    background: {BUTTON_ACCENT_BG_HOVER};
    font-weight: {FONT_WEIGHT_BOLD};
}}

QPushButton[role="accentAction"]:pressed {{
    background: {BUTTON_ACCENT_BG_PRESSED};
    font-weight: {FONT_WEIGHT_BOLD};
}}
"""

CONTROL_STYLE = f"""
QPushButton,
QToolButton,
QComboBox,
QLineEdit,
QAbstractSpinBox,
QTextEdit,
QPlainTextEdit,
QListView,
QTreeView,
QScrollArea,
QGroupBox,
QMenu {{
    background: {CONTROL_BG};
    color: {TEXT_CONTROL};
    border: none;
    outline: none;
}}

QPushButton,
QToolButton {{
    border-radius: {RADIUS_BUTTON};
}}

QPushButton,
QToolButton,
QComboBox,
QLineEdit,
QAbstractSpinBox {{
    min-height: {SIZE_CONTROL_MIN_HEIGHT};
    padding: {SIZE_CONTROL_PADDING};
}}

QPushButton {{
    min-height: {SIZE_BUTTON_MIN_HEIGHT};
    padding: {SIZE_BUTTON_PADDING};
    text-align: center;
    border-radius: {RADIUS_BUTTON};
}}

QPushButton[role="compactAction"] {{
    min-height: {SIZE_COMPACT_BUTTON_MIN_HEIGHT};
    padding: {SIZE_CONTROL_PADDING};
    border-radius: {RADIUS_COMPACT};
}}

QComboBox,
QLineEdit,
QAbstractSpinBox,
QTextEdit,
QPlainTextEdit {{
    border-radius: {RADIUS_CONTROL};
}}

QGroupBox {{
    border-radius: {RADIUS_CARD};
}}

QPushButton:hover,
QToolButton:hover,
QComboBox:hover,
QLineEdit:hover,
QAbstractSpinBox:hover {{
    background: {CONTROL_BG_HOVER};
}}

QPushButton:pressed,
QToolButton:pressed,
QComboBox:editable,
QComboBox:on {{
    background: {CONTROL_BG_PRESSED};
}}

QPushButton:focus,
QToolButton:focus,
QComboBox:focus,
QLineEdit:focus,
QAbstractSpinBox:focus,
QCheckBox:focus,
QTabBar::tab:focus {{
    border: none;
    outline: none;
}}

QLineEdit,
QAbstractSpinBox,
QTextEdit,
QPlainTextEdit {{
    selection-background-color: {TEXT_SELECTION_BG};
    selection-color: {TEXT_BRIGHT};
}}

QComboBox::drop-down {{
    border: none;
    width: {SIZE_COMBO_DROPDOWN_WIDTH};
    background: transparent;
}}

QAbstractSpinBox::up-button,
QAbstractSpinBox::down-button {{
    background: transparent;
    border: none;
    width: {SIZE_SPIN_BUTTON_WIDTH};
    subcontrol-origin: border;
}}

QAbstractSpinBox::up-button:hover,
QAbstractSpinBox::down-button:hover,
QAbstractSpinBox::up-button:pressed,
QAbstractSpinBox::down-button:pressed {{
    background: transparent;
    border: none;
}}

QComboBox QAbstractItemView {{
    background: {CONTROL_MENU_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {GRAY_3};
    border-radius: {RADIUS_POPUP};
    padding: 3px;
    selection-background-color: {CONTROL_BG};
    selection-color: {TEXT_PRIMARY};
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    border-radius: {RADIUS_CONTROL};
    padding: 3px 6px;
}}

QComboBox QAbstractItemView::item:selected {{
    background: {CONTROL_BG};
    border-left: 2px solid {TEXT_ACCENT};
}}

QMenu {{
    background: {CONTROL_MENU_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {GRAY_3};
    border-radius: {RADIUS_POPUP};
    padding: 4px;
}}

QMenu::item {{
    border-radius: {RADIUS_CONTROL};
    padding: 4px 18px 4px 14px;
}}

QMenu::item:selected {{
    background: {CONTROL_BG};
    color: {TEXT_PRIMARY};
    border-left: 2px solid {TEXT_ACCENT};
}}

QToolTip {{
    background-color: {GRAY_1};
    color: {TEXT_HI};
    border: 1px solid {GRAY_3};
    border-radius: {RADIUS_POPUP};
    padding: 4px 8px;
    font-size: 11px;
}}

QAbstractItemView {{
    alternate-background-color: {CONTROL_BG_HOVER};
    selection-background-color: {CONTROL_SELECTION_BG};
    selection-color: {TEXT_BRIGHT};
    border: none;
    outline: none;
}}

QAbstractItemView::item:selected {{
    background: {CONTROL_BG};
    border-left: 2px solid {TEXT_ACCENT};
}}

QHeaderView::section {{
    background: {HEADER_BG};
    color: {TEXT_HEADER_SECTION};
    border: none;
    padding: {SIZE_HEADER_PADDING};
}}

QCheckBox {{
    spacing: {SIZE_CHECKBOX_SPACING};
}}

QCheckBox::indicator {{
    width: {SIZE_CHECKBOX_INDICATOR};
    height: {SIZE_CHECKBOX_INDICATOR};
    border: 1px solid {GRAY_4};
    border-radius: {RADIUS_INDICATOR};
    background: {CONTROL_BG};
}}

QCheckBox::indicator:hover {{
    border-color: {ACCENT_COLOR_TEXT};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT_COLOR_TEXT};
    border: 1px solid {ACCENT_COLOR_TEXT};
    image: url('{CHECKMARK_ICON_PATH}');
}}

QCheckBox::indicator:checked:disabled {{
    background-color: {DISABLED_BG};
    border-color: {DISABLED_BG};
    image: url('{CHECKMARK_ICON_PATH}');
}}

QCheckBox::indicator:unchecked:disabled {{
    background: {DISABLED_BG};
    border-color: {DISABLED_BG};
}}

QRadioButton {{
    spacing: 6px;
    color: {TEXT_PRIMARY};
}}

QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {GRAY_4};
    border-radius: 7px;
    background: {CONTROL_BG};
}}

QRadioButton::indicator:hover {{
    border-color: {ACCENT_COLOR_TEXT};
}}

QRadioButton::indicator:checked {{
    background-color: {ACCENT_COLOR_TEXT};
    border: 1px solid {ACCENT_COLOR_TEXT};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {GRAY_3};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT_COLOR_TEXT};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {TEXT_HI};
    border: 1px solid {GRAY_4};
    width: 12px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 6px;
}}

QSlider::handle:horizontal:hover {{
    background: #ffffff;
    border-color: {ACCENT_COLOR_TEXT};
}}

QProgressBar {{
    background-color: {GRAY_3};
    border: none;
    border-radius: {RADIUS_INDICATOR};
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {ACCENT_COLOR_TEXT};
    border-radius: {RADIUS_INDICATOR};
}}

QToolButton {{
    background: {PANEL_BG};
    padding: {SIZE_TOOLBUTTON_PADDING};
    border-radius: {RADIUS_BUTTON};
}}

QToolButton:hover,
QToolButton:checked,
QToolButton:pressed {{
    background: {PANEL_BG};
}}

QToolButton[popupMode="1"] {{
    padding-right: {SIZE_TOOLBUTTON_POPUP_PADDING_RIGHT};
}}

QFormLayout {{
    spacing: {SIZE_FORM_SPACING};
}}
"""

CHROME_STYLE = f"""
QSplitter::handle {{
    background: {SPLITTER_BG};
    width: {SIZE_SPLITTER_HANDLE_WIDTH};
    border-radius: 1px;
}}

QSplitter::handle:hover {{
    background: {BUTTON_ACCENT_BG};
}}

QSplitter::handle:horizontal {{
    margin: 0 0 0 {SIZE_SPLITTER_HANDLE_MARGIN_LEFT};
}}

QSplitter::handle:vertical {{
    height: {SIZE_SPLITTER_HANDLE_WIDTH};
    margin: 1px 0 1px 0;
}}

QScrollBar:vertical,
QScrollBar:horizontal {{
    background: {PANEL_BG};
    border: none;
    margin: 0;
}}

QScrollBar::handle:vertical,
QScrollBar::handle:horizontal {{
    background: {SCROLLBAR_HANDLE_BG};
    border: none;
    border-radius: {RADIUS_SCROLLBAR};
    min-height: {SIZE_SCROLLBAR_HANDLE_MIN};
    min-width: {SIZE_SCROLLBAR_HANDLE_MIN};
}}

QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{
    background: {GRAY_4};
}}

QScrollBar::add-line,
QScrollBar::sub-line,
QScrollBar::add-page,
QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}

QFrame,
QGroupBox,
QDockWidget,
QToolBox,
QLabel,
QWidget[class="qt-scrollarea-viewport"] {{
    border: none;
}}

QStatusBar {{
    background: {STATUS_BG};
    color: {TEXT_STATUS};
    border: none;
    border-radius: {RADIUS_CONTROL};
    min-height: {SIZE_FOOTER_MIN_HEIGHT};
}}

QStatusBar::item {{
    border: none;
}}
"""