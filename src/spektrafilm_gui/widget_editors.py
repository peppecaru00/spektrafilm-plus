from __future__ import annotations

from functools import lru_cache
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

QPointF = getattr(QtCore, 'QPointF')
QRect = getattr(QtCore, 'QRect')
QSize = getattr(QtCore, 'QSize')
QStyle = QtWidgets.QStyle
QStyleOptionComboBox = QtWidgets.QStyleOptionComboBox
QStyledItemDelegate = QtWidgets.QStyledItemDelegate
QStylePainter = QtWidgets.QStylePainter

from spektrafilm.profiles.io import load_profile
from spektrafilm_gui.theme_palette import (
    TEXT_SOFT,
    TEXT_MAIN,
    TEXT_HI,
    RADIUS_CONTROL,
    RADIUS_COMPACT,
    GRAY_4,
    GRAY_3,
    GRAY_2,
    GRAY_1,
    ACCENT_COLOR_TEXT,
    ACCENT_COLOR_TEXT_SECONDARY,
    BOOL_EDITOR_BORDER_CHECKED,
    BOOL_EDITOR_BORDER_UNCHECKED,
    BOOL_EDITOR_CHECKED_DISABLED,
    BOOL_EDITOR_CHECKED_ENABLED,
    BOOL_EDITOR_CHECKMARK,
    BOOL_EDITOR_FILL_DISABLED,
    BOOL_EDITOR_FILL_ENABLED,
    BOOL_EDITOR_HOVER_BG,
    ACCENT_PRESETS,
)
from spektrafilm_gui.theme import resolve_theme_qcolor, get_accent_color, apply_accent_color


@lru_cache(maxsize=None)
def _profile_use_text_and_color(profile_name: str) -> tuple[str, str]:
    if not profile_name:
        return '', ACCENT_COLOR_TEXT
    try:
        profile = load_profile(profile_name)
    except (FileNotFoundError, TypeError, ValueError):
        return '', ACCENT_COLOR_TEXT
    if profile.is_cine:
        return 'cine', ACCENT_COLOR_TEXT_SECONDARY
    if profile.is_still:
        return 'still', ACCENT_COLOR_TEXT
    return '', ACCENT_COLOR_TEXT


def _draw_profile_display_text(
    painter: QtGui.QPainter,
    rect: QtCore.QRect,
    *,
    profile_name: str,
    base_color: QtGui.QColor,
    font: QtGui.QFont,
) -> None:
    profile_use, use_color = _profile_use_text_and_color(profile_name)
    if not profile_use:
        painter.setPen(base_color)
        painter.setFont(font)
        painter.drawText(rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, profile_name)
        return

    font_metrics = QtGui.QFontMetrics(font)
    prefix_text = f'{profile_use} '
    slash_text = '/ '
    prefix_width = font_metrics.horizontalAdvance(prefix_text)
    slash_width = font_metrics.horizontalAdvance(slash_text)

    painter.setFont(font)
    prefix_rect = QRect(rect.x(), rect.y(), prefix_width, rect.height())
    slash_rect = QRect(rect.x() + prefix_width, rect.y(), slash_width, rect.height())
    value_rect = QRect(rect.x() + prefix_width + slash_width, rect.y(), max(0, rect.width() - prefix_width - slash_width), rect.height())

    painter.setPen(QtGui.QColor(use_color))
    painter.drawText(prefix_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft | QtCore.Qt.TextSingleLine, prefix_text)
    painter.setPen(base_color)
    painter.drawText(slash_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft | QtCore.Qt.TextSingleLine, slash_text)
    painter.drawText(value_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft | QtCore.Qt.TextSingleLine, profile_name)


class ProfileEnumItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:  # noqa: N802 - Qt API name
        display_value = index.data(QtCore.Qt.DisplayRole) or ''
        style_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style_option.text = ''

        style = style_option.widget.style() if style_option.widget is not None else QtWidgets.QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, style_option, painter, style_option.widget)

        text_color = style_option.palette.color(
            QtGui.QPalette.HighlightedText if style_option.state & QStyle.State_Selected else QtGui.QPalette.Text,
        )
        text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, style_option, style_option.widget)
        _draw_profile_display_text(
            painter,
            text_rect.adjusted(0, 0, -4, 0),
            profile_name=display_value,
            base_color=text_color,
            font=style_option.font,
        )


class ProfileEnumEditor(QtWidgets.QComboBox):
    def __init__(self, values: list[str]):
        super().__init__()
        self.addItems(values)
        self.setItemDelegate(ProfileEnumItemDelegate(self))
        if self.count() > 0 and self.currentIndex() < 0:
            self.setCurrentIndex(0)

    @staticmethod
    def display_text_for_value(value: str) -> str:
        profile_use, _color = _profile_use_text_and_color(value)
        if not profile_use:
            return value
        return f'{profile_use} / {value}'

    @property
    def value(self) -> str:
        return self.currentText()

    @value.setter
    def value(self, value: str) -> None:
        index = self.findText(value)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            raise ValueError(f"{value!r} is not a valid option")

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        current_value = self.currentText() or (self.itemText(0) if self.count() > 0 else '')
        option.currentText = ''
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)
        text_rect = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self)
        _draw_profile_display_text(
            painter,
            text_rect.adjusted(1, 0, -1, 0),
            profile_name=current_value,
            base_color=option.palette.color(QtGui.QPalette.Text),
            font=self.font(),
        )


class FloatEditor(QtWidgets.QDoubleSpinBox):
    def __init__(self, *, decimals: int = 2, minimum: float = -1_000_000.0, maximum: float = 1_000_000.0):
        super().__init__()
        self.setDecimals(decimals)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setKeyboardTracking(False)
        self.setFixedHeight(24)

    @property
    def value(self) -> float:
        return super().value()

    @value.setter
    def value(self, value: float) -> None:
        self.setValue(value)


class IntEditor(QtWidgets.QSpinBox):
    def __init__(self, *, minimum: int = -1_000_000, maximum: int = 1_000_000):
        super().__init__()
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setKeyboardTracking(False)
        self.setFixedHeight(24)

    @property
    def value(self) -> int:
        return super().value()

    @value.setter
    def value(self, value: int) -> None:
        self.setValue(value)


class BoolEditor(QtWidgets.QCheckBox):
    def __init__(self):
        super().__init__()
        self.setText('')
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(24)

    @property
    def value(self) -> bool:
        return self.isChecked()

    @value.setter
    def value(self, value: bool) -> None:
        self.setChecked(value)

    def sizeHint(self):  # noqa: N802 - Qt API name
        return QSize(24, 24)

    def minimumSizeHint(self):  # noqa: N802 - Qt API name
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            indicator_rect = self._indicator_rect()
            active_accent = get_accent_color()
            is_enabled = self.isEnabled()
            is_hovered = is_enabled and self.underMouse()
            fill_color = resolve_theme_qcolor(
                BOOL_EDITOR_HOVER_BG if is_hovered else BOOL_EDITOR_FILL_ENABLED if is_enabled else BOOL_EDITOR_FILL_DISABLED,
            )
            checked_color = resolve_theme_qcolor(
                BOOL_EDITOR_HOVER_BG if is_hovered else active_accent if is_enabled else BOOL_EDITOR_CHECKED_DISABLED,
            )
            border_color = resolve_theme_qcolor(active_accent if self.isChecked() else BOOL_EDITOR_BORDER_UNCHECKED)

            painter.setPen(QtGui.QPen(border_color, 1))
            painter.setBrush(checked_color if self.isChecked() else fill_color)
            painter.drawRoundedRect(indicator_rect, 4, 4)

            if self.isChecked():
                self._draw_check_mark(painter, indicator_rect)
        finally:
            painter.end()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().leaveEvent(event)
        self.update()

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 - Qt API name
        super().changeEvent(event)
        self.update()

    def _indicator_rect(self):
        return QRect(1, max(1, (self.height() - 14) // 2), 14, 14)

    @staticmethod
    def _draw_check_mark(painter: QtGui.QPainter, indicator_rect) -> None:
        painter.setPen(
            QtGui.QPen(resolve_theme_qcolor(BOOL_EDITOR_CHECKMARK), 2.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin),
        )
        painter.drawLine(
            QPointF(indicator_rect.left() + 3.0, indicator_rect.center().y() + 0.5),
            QPointF(indicator_rect.left() + 6.0, indicator_rect.bottom() - 3.5),
        )
        painter.drawLine(
            QPointF(indicator_rect.left() + 6.0, indicator_rect.bottom() - 3.5),
            QPointF(indicator_rect.right() - 2.5, indicator_rect.top() + 3.5),
        )


class EnumEditor(QtWidgets.QComboBox):
    def __init__(self, values: list[str]):
        super().__init__()
        self.addItems(values)

    @property
    def value(self) -> str:
        return self.currentText()

    @value.setter
    def value(self, value: str) -> None:
        index = self.findText(value)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            raise ValueError(f"{value!r} is not a valid option")


class TupleEditor(QtWidgets.QWidget):
    def __init__(self, editors: list[QtWidgets.QWidget]):
        super().__init__()
        self._editors = editors
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for editor in self._editors:
            editor.setMinimumWidth(56)
            editor.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            layout.addWidget(editor)
        self.setLayout(layout)

    @property
    def editors(self) -> tuple[QtWidgets.QWidget, ...]:
        return tuple(self._editors)

    @property
    def value(self) -> tuple[Any, ...]:
        return tuple(editor.value for editor in self._editors)

    @value.setter
    def value(self, value: tuple[Any, ...]) -> None:
        for editor, component in zip(self._editors, value, strict=True):
            editor.value = component

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt API name
        super().setToolTip(text)
        for editor in self._editors:
            editor.setToolTip(text)


class FloatTupleEditor(TupleEditor):
    def __init__(self, length: int, *, decimals: int = 2, minimum: float = -1_000_000.0, maximum: float = 1_000_000.0):
        super().__init__([FloatEditor(decimals=decimals, minimum=minimum, maximum=maximum) for _ in range(length)])


class IntTupleEditor(TupleEditor):
    def __init__(self, length: int, *, minimum: int = -1_000_000, maximum: int = 1_000_000):
        super().__init__([IntEditor(minimum=minimum, maximum=maximum) for _ in range(length)])


class SliderFloatEditor(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(self, *, minimum: float = 0.0, maximum: float = 100.0, step: float = 1.0, decimals: int = 0, suffix: str = ''):
        super().__init__()
        self._minimum = minimum
        self._maximum = maximum
        self._step = step
        self._decimals = decimals
        self._suffix = suffix

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(0, self._n_steps())
        self._slider.setFixedHeight(24)

        self._label = QtWidgets.QLabel()
        self._label.setFixedWidth(52)
        self._label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._label)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._update_label()

    def _n_steps(self) -> int:
        if self._step <= 0:
            return 0
        return int(round((self._maximum - self._minimum) / self._step))

    def _on_slider_changed(self, _tick: int) -> None:
        self._update_label()
        self.valueChanged.emit(self.value)

    def _update_label(self) -> None:
        self._label.setText(f'{self.value:.{self._decimals}f}{self._suffix}')

    @property
    def value(self) -> float:
        return self._minimum + self._slider.value() * self._step

    @value.setter
    def value(self, v: float) -> None:
        tick = int(round((v - self._minimum) / self._step))
        tick = max(0, min(self._slider.maximum(), tick))
        was_blocked = self._slider.blockSignals(True)
        self._slider.setValue(tick)
        self._slider.blockSignals(was_blocked)
        self._update_label()

    def setMinimum(self, v: float) -> None:
        self._minimum = v
        self._slider.setRange(0, self._n_steps())

    def setMaximum(self, v: float) -> None:
        self._maximum = v
        self._slider.setRange(0, self._n_steps())

    def setSingleStep(self, v: float) -> None:
        self._step = v
        self._slider.setRange(0, self._n_steps())

    def setToolTip(self, text: str) -> None:
        super().setToolTip(text)
        self._slider.setToolTip(text)
        self._label.setToolTip(text)

class LensSelectorWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(str)
    currentTextChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._value: str = "Auto"
        self._detected_summary: str | None = None
        self._is_updating = False
        self._picker_expanded = False
        self.setObjectName("lensSelector")
        self.setProperty("role", "lensSelector")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Header with status label, toggle button, and Reset to Auto button
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self.status_label = QtWidgets.QLabel("Auto: (Auto-detection active)")
        self.status_label.setStyleSheet(f"color: {TEXT_SOFT}; font-size: 11px;")
        self.status_label.setWordWrap(True)
        header_layout.addWidget(self.status_label, 1)

        self.toggle_button = QtWidgets.QPushButton("Browse Lenses ▾")
        self.toggle_button.setToolTip("Browse and manually select a lens profile")
        self.toggle_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.toggle_button.setStyleSheet(f"""
            QPushButton {{
                background: {GRAY_2};
                color: {TEXT_MAIN};
                border: 1px solid {GRAY_3};
                border-radius: {RADIUS_COMPACT};
                padding: 1px 8px;
                font-size: 11px;
                min-height: 18px;
            }}
            QPushButton:hover {{
                background: {GRAY_3};
                color: {TEXT_HI};
            }}
            QPushButton:disabled {{
                color: {GRAY_4};
                border-color: {GRAY_2};
            }}
        """)
        self.toggle_button.clicked.connect(self._toggle_picker)
        header_layout.addWidget(self.toggle_button, 0)

        self.reset_button = QtWidgets.QPushButton("Auto \u21ba")
        self.reset_button.setToolTip("Reset to automatic lens detection from RAW EXIF")
        self.reset_button.setStyleSheet(f"""
            QPushButton {{
                background: {GRAY_2};
                color: {TEXT_MAIN};
                border: 1px solid {GRAY_3};
                border-radius: {RADIUS_COMPACT};
                padding: 1px 6px;
                font-size: 11px;
                min-height: 18px;
            }}
            QPushButton:hover {{
                background: {GRAY_3};
                color: {TEXT_HI};
            }}
            QPushButton:disabled {{
                color: {GRAY_4};
                border-color: {GRAY_2};
            }}
        """)
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        header_layout.addWidget(self.reset_button, 0)
        layout.addLayout(header_layout)

        # Collapsible picker container (search edit + list widget)
        self.picker_container = QtWidgets.QWidget()
        picker_layout = QtWidgets.QVBoxLayout(self.picker_container)
        picker_layout.setContentsMargins(0, 2, 0, 0)
        picker_layout.setSpacing(4)

        # Filter / Search Edit
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search lens (e.g. Canon 50mm, Sony FE)...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {GRAY_1};
                color: {TEXT_HI};
                border: 1px solid {GRAY_3};
                border-radius: {RADIUS_CONTROL};
                padding: 2px 6px;
                font-size: 11px;
                min-height: 20px;
            }}
            QLineEdit:focus {{
                border-color: {ACCENT_COLOR_TEXT};
            }}
        """)
        self.search_edit.textChanged.connect(self._on_search_changed)
        picker_layout.addWidget(self.search_edit)

        # Scrollable list
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.list_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_widget.setMinimumHeight(130)
        self.list_widget.setMaximumHeight(160)
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background: {GRAY_1};
                color: {TEXT_MAIN};
                border: 1px solid {GRAY_3};
                border-radius: {RADIUS_CONTROL};
                padding: 2px;
                font-size: 11px;
            }}
            QListWidget::item {{
                padding: 2px 4px;
                border-radius: 3px;
            }}
            QListWidget::item:hover {{
                background: {GRAY_2};
            }}
            QListWidget::item:selected {{
                background: {GRAY_3};
                color: {TEXT_HI};
                border-left: 2px solid {ACCENT_COLOR_TEXT};
            }}
        """)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        picker_layout.addWidget(self.list_widget)

        self.picker_container.setVisible(False)
        layout.addWidget(self.picker_container)

        self._populate_lenses()

    def _update_auto_item_text(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._detected_summary:
            item.setText(f"Auto (Detected: {self._detected_summary})")
        else:
            item.setText("Auto (Detected)")

    def _populate_lenses(self) -> None:
        from spektrafilm.utils.raw_file_processor import get_available_lenses

        self.list_widget.clear()

        # Item 0: Auto
        auto_item = QtWidgets.QListWidgetItem()
        auto_item.setData(QtCore.Qt.UserRole, "Auto")
        self._update_auto_item_text(auto_item)
        font = auto_item.font()
        font.setBold(True)
        auto_item.setFont(font)
        auto_item.setForeground(QtGui.QBrush(QtGui.QColor(get_accent_color())))
        self.list_widget.addItem(auto_item)

        # Lenses from lensfun
        lenses = get_available_lenses()
        for lens_name in lenses:
            item = QtWidgets.QListWidgetItem(lens_name)
            item.setData(QtCore.Qt.UserRole, lens_name)
            self.list_widget.addItem(item)

        self._select_item_by_value(self._value)

    def _select_item_by_value(self, val: str) -> None:
        self._is_updating = True
        try:
            target_val = val if val else "Auto"
            matched_item = None
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                item_val = item.data(QtCore.Qt.UserRole)
                if item_val == target_val:
                    matched_item = item
                    break

            if matched_item is not None:
                self.list_widget.setCurrentItem(matched_item)
                matched_item.setHidden(False)
                self.list_widget.scrollToItem(matched_item, QtWidgets.QAbstractItemView.EnsureVisible)
            else:
                # If custom value not in list, select Auto
                auto_item = self.list_widget.item(0)
                if auto_item is not None:
                    self.list_widget.setCurrentItem(auto_item)

            self._update_ui_state()
        finally:
            self._is_updating = False

    def _on_selection_changed(self) -> None:
        if self._is_updating:
            return
        selected = self.list_widget.selectedItems()
        if not selected:
            return
        item = selected[0]
        val = item.data(QtCore.Qt.UserRole) or "Auto"
        if val != self._value:
            self._value = val
            self._update_ui_state()
            self.valueChanged.emit(self._value)
            self.currentTextChanged.emit(self._value)

    def _on_reset_clicked(self) -> None:
        self.value = "Auto"
        self.valueChanged.emit(self._value)
        self.currentTextChanged.emit(self._value)

    def _on_search_changed(self, text: str) -> None:
        query = text.strip().lower()
        words = query.split()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            val = item.data(QtCore.Qt.UserRole)
            if val == "Auto":
                if not words:
                    item.setHidden(False)
                else:
                    auto_text = (item.text() + " auto").lower()
                    item.setHidden(not all(w in auto_text for w in words))
            else:
                txt = item.text().lower()
                item.setHidden(not all(w in txt for w in words))

    def _update_ui_state(self) -> None:
        is_auto = (self._value == "Auto")
        self.reset_button.setEnabled(not is_auto and self.isEnabled())
        if is_auto:
            if self._detected_summary:
                self.status_label.setText(f"Auto: {self._detected_summary}")
                self.status_label.setStyleSheet(f"color: {get_accent_color()}; font-size: 11px;")
            else:
                self.status_label.setText("Auto: (Auto-detection active)")
                self.status_label.setStyleSheet(f"color: {TEXT_SOFT}; font-size: 11px;")
        else:
            self.status_label.setText(f"Manual: {self._value}")
            self.status_label.setStyleSheet(f"color: {TEXT_HI}; font-weight: bold; font-size: 11px;")

    def set_detected_lens(self, lens_summary: str | None) -> None:
        """Update the detected lens summary for the Auto item and status label."""
        self._detected_summary = lens_summary.strip() if lens_summary else None
        if self.list_widget.count() > 0:
            auto_item = self.list_widget.item(0)
            if auto_item and auto_item.data(QtCore.Qt.UserRole) == "Auto":
                self._update_auto_item_text(auto_item)
        self._update_ui_state()

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, v: str) -> None:
        new_val = v if v else "Auto"
        if new_val != self._value:
            self._value = new_val
            self._select_item_by_value(self._value)
        else:
            self._select_item_by_value(self._value)

    @property
    def is_picker_expanded(self) -> bool:
        return self._picker_expanded

    def set_picker_expanded(self, expanded: bool) -> None:
        self._picker_expanded = bool(expanded)
        self.picker_container.setVisible(self._picker_expanded)
        self.toggle_button.setText("Close ▴" if self._picker_expanded else "Browse Lenses ▾")

    def _toggle_picker(self) -> None:
        self.set_picker_expanded(not self._picker_expanded)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.status_label.setEnabled(enabled)
        self.toggle_button.setEnabled(enabled)
        self.search_edit.setEnabled(enabled)
        self.list_widget.setEnabled(enabled)
        self.reset_button.setEnabled(enabled and self._value != "Auto")


class AccentColorSelectorWidget(QtWidgets.QWidget):
    colorChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._current_color = get_accent_color()
        self._buttons: dict[str, QtWidgets.QPushButton] = {}

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 4, 0, 4)
        main_layout.setSpacing(6)

        # Header / Label row
        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        label = QtWidgets.QLabel("UI Accent Color")
        label.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 11px; font-weight: bold;")
        header_row.addWidget(label, 0)

        self.color_preview = QtWidgets.QLabel()
        self.color_preview.setFixedSize(14, 14)
        header_row.addWidget(self.color_preview, 0)

        self.color_name_lbl = QtWidgets.QLabel(self._get_color_display_name(self._current_color))
        self.color_name_lbl.setStyleSheet(f"color: {TEXT_SOFT}; font-size: 10px;")
        header_row.addWidget(self.color_name_lbl, 1)

        main_layout.addLayout(header_row)

        # Swatch bar row
        swatch_row = QtWidgets.QHBoxLayout()
        swatch_row.setContentsMargins(0, 4, 0, 4)
        swatch_row.setSpacing(6)
        swatch_row.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        for name, hex_code in ACCENT_PRESETS:
            btn = QtWidgets.QPushButton()
            btn.setProperty("role", "colorSwatch")
            btn.setFixedSize(22, 22)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            btn.setToolTip(f"{name} ({hex_code})")
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, c=hex_code: self.select_color(c))
            self._buttons[hex_code.lower()] = btn
            swatch_row.addWidget(btn)

        # Custom color button
        self.custom_btn = QtWidgets.QPushButton("Custom...")
        self.custom_btn.setToolTip("Pick a custom accent color from the color dialog")
        self.custom_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.custom_btn.setFixedHeight(22)
        self.custom_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.custom_btn.setStyleSheet(f"""
            QPushButton {{
                background: {GRAY_2};
                color: {TEXT_MAIN};
                border: 1px solid {GRAY_3};
                border-radius: {RADIUS_COMPACT};
                padding: 1px 8px;
                font-size: 10px;
                min-height: 20px;
                max-height: 22px;
            }}
            QPushButton:hover {{
                background: {GRAY_3};
                color: {TEXT_HI};
            }}
        """)
        self.custom_btn.clicked.connect(self._on_custom_clicked)
        swatch_row.addWidget(self.custom_btn)
        swatch_row.addStretch(1)

        main_layout.addLayout(swatch_row)
        self._update_swatch_styles()

    def _get_color_display_name(self, hex_code: str) -> str:
        for name, h in ACCENT_PRESETS:
            if h.lower() == hex_code.lower():
                return f"{name} ({hex_code.upper()})"
        return f"Custom ({hex_code.upper()})"

    def _update_swatch_styles(self) -> None:
        curr = self._current_color.lower()
        for hex_code, btn in self._buttons.items():
            is_active = (hex_code == curr)
            border = "2px solid #ffffff" if is_active else "1px solid rgba(255, 255, 255, 0.25)"
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {hex_code};
                    border: {border};
                    border-radius: 11px;
                    min-height: 22px;
                    max-height: 22px;
                    min-width: 22px;
                    max-width: 22px;
                    padding: 0px;
                    margin: 0px;
                }}
                QPushButton:hover {{
                    border: 2px solid rgba(255, 255, 255, 0.8);
                }}
            """)
        self.color_preview.setStyleSheet(
            f"background-color: {self._current_color}; border-radius: 7px; border: 1px solid rgba(255,255,255,0.4); "
            "min-width: 14px; max-width: 14px; min-height: 14px; max-height: 14px; padding: 0px; margin: 0px;"
        )
        self.color_name_lbl.setText(self._get_color_display_name(self._current_color))

    @property
    def current_color(self) -> str:
        return self._current_color

    def select_color(self, hex_code: str) -> None:
        if not hex_code:
            return
        self._current_color = hex_code
        self._update_swatch_styles()
        apply_accent_color(hex_code)
        self.colorChanged.emit(hex_code)

    def _on_custom_clicked(self) -> None:
        initial = QtGui.QColor(self._current_color)
        chosen = QtWidgets.QColorDialog.getColor(initial, self, "Select Accent Color")
        if chosen.isValid():
            self.select_color(chosen.name())
