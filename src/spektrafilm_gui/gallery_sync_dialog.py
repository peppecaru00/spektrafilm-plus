from __future__ import annotations

import logging
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from spektrafilm_gui.gallery_params import SingleParamDef, all_param_defs, get_param_def
from spektrafilm_gui.state import GuiState
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


class SyncParameterDialog(QtWidgets.QDialog):
    """Dialog for choosing one or more active parameters to copy or sync across gallery photos."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        active_state: GuiState,
        active_filename: str,
        selected_count: int,
        total_count: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sync / Copy Parameters")
        self.setMinimumSize(560, 520)
        self.resize(620, 560)

        self._active_state = active_state
        self._active_filename = active_filename
        self._selected_count = selected_count
        self._total_count = total_count
        self._is_copy_only = False
        self._updating_checks = False

        self._build_ui()
        self._populate_tree()
        self._update_buttons()

    def _build_ui(self) -> None:
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {GRAY_0};
                color: {TEXT_MAIN};
            }}
            QLabel {{
                color: {TEXT_MAIN};
            }}
            QGroupBox {{
                background-color: {GRAY_1};
                border: 1px solid {GRAY_3};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: {FONT_WEIGHT_SEMIBOLD};
                color: {TEXT_HI};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QTreeWidget {{
                background-color: {GRAY_1};
                border: 1px solid {GRAY_3};
                border-radius: 8px;
                color: {TEXT_MAIN};
                alternate-background-color: {GRAY_2};
            }}
            QTreeWidget::item:selected {{
                background-color: {GRAY_3};
                color: {ACCENT_COLOR_TEXT};
                font-weight: {FONT_WEIGHT_BOLD};
            }}
            QTreeWidget::indicator {{
                width: 14px;
                height: 14px;
            }}
            QHeaderView::section {{
                background-color: {GRAY_2};
                color: {TEXT_HI};
                padding: 4px;
                border: 1px solid {GRAY_3};
                font-weight: {FONT_WEIGHT_SEMIBOLD};
            }}
            QLineEdit {{
                background-color: {GRAY_1};
                border: 1px solid {GRAY_3};
                border-radius: 6px;
                padding: 4px 8px;
                color: {TEXT_MAIN};
            }}
            QRadioButton {{
                color: {TEXT_MAIN};
                spacing: 6px;
            }}
            QRadioButton::indicator {{
                width: 14px;
                height: 14px;
            }}
            QPushButton {{
                background-color: {GRAY_2};
                color: {TEXT_MAIN};
                border: 1px solid {GRAY_4};
                border-radius: 6px;
                padding: 5px 12px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {GRAY_3};
                color: {TEXT_HI};
            }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Header banner
        header = QtWidgets.QFrame()
        header.setStyleSheet(f"background-color: {GRAY_1}; border: 1px solid {GRAY_3}; border-radius: 8px; padding: 6px;")
        h_layout = QtWidgets.QVBoxLayout(header)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(4)

        src_lbl = QtWidgets.QLabel(f"Source Photo: <b>{self._active_filename}</b>")
        src_lbl.setStyleSheet(f"color: {ACCENT_COLOR_TEXT}; font-size: 12px;")
        h_layout.addWidget(src_lbl)

        info_lbl = QtWidgets.QLabel(
            "Select which active parameters to copy or paste across photos. Check the boxes you wish to sync."
        )
        info_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        h_layout.addWidget(info_lbl)

        layout.addWidget(header)

        # Quick check buttons & Search filter row
        tools_row = QtWidgets.QHBoxLayout()
        tools_row.setSpacing(6)

        self.check_all_btn = QtWidgets.QPushButton("☑ Check All")
        self.check_all_btn.setToolTip("Check all parameters")
        self.check_all_btn.setFixedHeight(26)
        self.check_all_btn.clicked.connect(self._check_all)
        tools_row.addWidget(self.check_all_btn)

        self.check_none_btn = QtWidgets.QPushButton("☐ Check None")
        self.check_none_btn.setToolTip("Uncheck all parameters")
        self.check_none_btn.setFixedHeight(26)
        self.check_none_btn.clicked.connect(self._check_none)
        tools_row.addWidget(self.check_none_btn)

        tools_row.addSpacing(10)

        search_lbl = QtWidgets.QLabel("🔍 Search:")
        search_lbl.setStyleSheet(f"color: {TEXT_HI}; font-size: 11px;")
        tools_row.addWidget(search_lbl)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setFixedHeight(26)
        self.search_edit.setPlaceholderText("Filter parameters by name or category...")
        self.search_edit.textChanged.connect(self._filter_items)
        tools_row.addWidget(self.search_edit, 1)

        layout.addLayout(tools_row)

        # Parameter Tree
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Parameter", "Current Value"])
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(lambda item, col: self._on_double_clicked(item))
        layout.addWidget(self.tree, 1)

        # Scope selection
        scope_group = QtWidgets.QGroupBox("Target Photos")
        scope_layout = QtWidgets.QHBoxLayout(scope_group)
        scope_layout.setContentsMargins(12, 10, 12, 10)
        scope_layout.setSpacing(16)

        sel_text = f"Selected photos ({self._selected_count} photo{'s' if self._selected_count != 1 else ''})"
        self.radio_selected = QtWidgets.QRadioButton(sel_text)
        all_text = f"All photos in filmstrip ({self._total_count} total)"
        self.radio_all = QtWidgets.QRadioButton(all_text)

        if self._selected_count > 0:
            self.radio_selected.setChecked(True)
        else:
            self.radio_all.setChecked(True)
            self.radio_selected.setEnabled(False)

        scope_layout.addWidget(self.radio_selected)
        scope_layout.addWidget(self.radio_all)
        scope_layout.addStretch(1)
        layout.addWidget(scope_group)

        # Action button row
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self.copy_btn = QtWidgets.QPushButton("📋 Copy Parameters")
        self.copy_btn.setToolTip("Copy checked parameters to clipboard so they can be pasted anytime")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._on_copy_clicked)
        btn_row.addWidget(self.copy_btn)

        btn_row.addStretch(1)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.apply_btn = QtWidgets.QPushButton("🚀 Paste to Photos")
        self.apply_btn.setToolTip("Immediately apply checked parameters to target photos and update sidecars")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR_TEXT};
                color: #000;
                font-weight: {FONT_WEIGHT_BOLD};
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
            }}
            QPushButton:hover {{
                background-color: #fca987;
            }}
            QPushButton:disabled {{
                background-color: {GRAY_3};
                color: {TEXT_DIM};
            }}
        """)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        btn_row.addWidget(self.apply_btn)

        layout.addLayout(btn_row)

    def _populate_tree(self) -> None:
        self._updating_checks = True
        self.tree.clear()
        categories: dict[str, QtWidgets.QTreeWidgetItem] = {}

        for p_def in all_param_defs():
            cat = p_def.category
            if cat not in categories:
                cat_item = QtWidgets.QTreeWidgetItem(self.tree, [cat, ""])
                cat_item.setFlags(cat_item.flags() | Qt.ItemIsUserCheckable)
                cat_item.setCheckState(0, Qt.Checked)
                cat_item.setExpanded(True)
                categories[cat] = cat_item
            else:
                cat_item = categories[cat]

            val_str = p_def.get_display(self._active_state)
            item = QtWidgets.QTreeWidgetItem(cat_item, [p_def.name, val_str])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked)
            item.setData(0, Qt.UserRole, p_def.id)
            item.setToolTip(0, p_def.description)
            item.setToolTip(1, val_str)

        # Select first parameter by default
        if categories:
            first_cat = next(iter(categories.values()))
            if first_cat.childCount() > 0:
                first_param = first_cat.child(0)
                self.tree.setCurrentItem(first_param)

        self._updating_checks = False

    def _filter_items(self, text: str) -> None:
        search = text.strip().lower()
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat_item = root.child(i)
            cat_matches = search in cat_item.text(0).lower()
            any_child_visible = False
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                name = child.text(0).lower()
                val = child.text(1).lower()
                matches = search in name or search in val or cat_matches
                child.setHidden(not matches)
                if matches:
                    any_child_visible = True
            cat_item.setHidden(not any_child_visible and not cat_matches)

    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if self._updating_checks or column != 0:
            return

        self._updating_checks = True
        try:
            parent = item.parent()
            if parent is None:
                # Category clicked: update all child items to match category state
                state = item.checkState(0)
                for i in range(item.childCount()):
                    item.child(i).setCheckState(0, state)
            else:
                # Child clicked: update parent category check state
                checked_children = sum(1 for i in range(parent.childCount()) if parent.child(i).checkState(0) == Qt.Checked)
                if checked_children == parent.childCount():
                    parent.setCheckState(0, Qt.Checked)
                elif checked_children == 0:
                    parent.setCheckState(0, Qt.Unchecked)
                else:
                    parent.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self._updating_checks = False

        self._update_buttons()

    def _check_all(self) -> None:
        self._updating_checks = True
        try:
            root = self.tree.invisibleRootItem()
            for i in range(root.childCount()):
                cat = root.child(i)
                cat.setCheckState(0, Qt.Checked)
                for j in range(cat.childCount()):
                    cat.child(j).setCheckState(0, Qt.Checked)
        finally:
            self._updating_checks = False
        self._update_buttons()

    def _check_none(self) -> None:
        self._updating_checks = True
        try:
            root = self.tree.invisibleRootItem()
            for i in range(root.childCount()):
                cat = root.child(i)
                cat.setCheckState(0, Qt.Unchecked)
                for j in range(cat.childCount()):
                    cat.child(j).setCheckState(0, Qt.Unchecked)
        finally:
            self._updating_checks = False
        self._update_buttons()

    def _get_checked_param_ids(self) -> list[str]:
        ids: list[str] = []
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat = root.child(i)
            for j in range(cat.childCount()):
                child = cat.child(j)
                if child.checkState(0) == Qt.Checked:
                    param_id = child.data(0, Qt.UserRole)
                    if param_id:
                        ids.append(param_id)
        return ids

    def _update_buttons(self) -> None:
        checked_ids = self._get_checked_param_ids()
        count = len(checked_ids)

        if count > 0:
            self.copy_btn.setEnabled(True)
            self.apply_btn.setEnabled(True)
            self.copy_btn.setText(f"📋 Copy Parameters ({count})")
            self.apply_btn.setText(f"🚀 Paste to Photos ({count})")
        else:
            self.copy_btn.setEnabled(False)
            self.apply_btn.setEnabled(False)
            self.copy_btn.setText("📋 Copy Parameters")
            self.apply_btn.setText("🚀 Paste to Photos")

    def _on_selection_changed(self) -> None:
        self._update_buttons()

    def _on_double_clicked(self, item: QtWidgets.QTreeWidgetItem) -> None:
        if item is not None and item.data(0, Qt.UserRole) is not None:
            # Quick 1-click single parameter sync: isolate this parameter and accept
            self._updating_checks = True
            try:
                root = self.tree.invisibleRootItem()
                for i in range(root.childCount()):
                    cat = root.child(i)
                    cat.setCheckState(0, Qt.Unchecked)
                    for j in range(cat.childCount()):
                        cat.child(j).setCheckState(0, Qt.Unchecked)
                item.setCheckState(0, Qt.Checked)
                if item.parent() is not None:
                    item.parent().setCheckState(0, Qt.PartiallyChecked)
            finally:
                self._updating_checks = False
            self._on_apply_clicked()

    def _on_copy_clicked(self) -> None:
        self._is_copy_only = True
        self.accept()

    def _on_apply_clicked(self) -> None:
        self._is_copy_only = False
        self.accept()

    def get_checked_params_dict(self) -> dict[str, Any]:
        """Return a dictionary of {param_id: value} for all checked parameters from active state."""
        checked_ids = self._get_checked_param_ids()
        res: dict[str, Any] = {}
        for param_id in checked_ids:
            p_def = get_param_def(param_id)
            if p_def is not None:
                res[param_id] = p_def.extract(self._active_state)
        return res

    def get_selected_param_id(self) -> str | None:
        """Return the parameter ID of the currently focused item or the first checked parameter."""
        item = self.tree.currentItem()
        if item is not None and item.data(0, Qt.UserRole) is not None:
            return item.data(0, Qt.UserRole)
        checked_ids = self._get_checked_param_ids()
        if checked_ids:
            return checked_ids[0]
        return None

    def get_selected_param_ids(self) -> list[str]:
        """Return the list of all checked parameter IDs."""
        checked_ids = self._get_checked_param_ids()
        if checked_ids:
            return checked_ids
        single = self.get_selected_param_id()
        return [single] if single else []

    def get_apply_scope(self) -> str:
        """Return 'selected' or 'all' based on radio button choice."""
        if self.radio_all.isChecked():
            return "all"
        return "selected"

    def was_copy_only(self) -> bool:
        """Return True if the user clicked 'Copy Parameters' rather than 'Paste to Photos'."""
        return self._is_copy_only
