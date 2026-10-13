from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from qtpy import QtCore, QtWidgets

QComboBox = QtWidgets.QComboBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QPushButton = QtWidgets.QPushButton
QMenu = QtWidgets.QMenu
QSizePolicy = QtWidgets.QSizePolicy
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget
Qt = QtCore.Qt
Signal = QtCore.Signal

from spektrafilm_gui.param_manifest import (
    CAMERA_EXPOSURE_BORROWED_FIELDS,
    CROP_PANEL_FIELDS,
    DISPLAY_PANEL_FIELDS,
    GroupManifest,
    INPUT_IMAGE_FIELDS,
    INPUT_PANEL_FIELDS,
    ParamSpec,
    PRECISION_PANEL_FIELDS,
    SIMULATION_ENLARGER_PANEL_FIELDS,
    SIMULATION_EXPOSURE_PANEL_FIELDS,
    SIMULATION_FIELDS,
    SIMULATION_OUTPUT_PANEL_FIELDS,
    SIMULATION_PROFILE_PANEL_FIELDS,
    SIMULATION_SPECIAL_BORROWED_FIELDS,
    SPECTRAL_PANEL_FIELDS,
    SPECIAL_FIELDS,
    TUNE_PANEL_FIELDS,
)
from spektrafilm_gui.state import (
    DisplayState,
    InputImageState,
    LoadRawState,
    PROJECT_DEFAULT_GUI_STATE,
    SelectionState,
    SimulationState,
    SpecialState,
    clone_state_section,
)
from spektrafilm.utils.raw_file_processor import read_raw_white_balance
from spektrafilm_gui.curve_editor import YRGBCurveEditor
from spektrafilm_gui.persistence import load_dialog_dir, save_dialog_dir
from spektrafilm_gui.theme_palette import (
    ACCENT_COLOR_TEXT,
    GRAY_1,
    GRAY_3,
    SIZE_FOOTER_ITEM_SPACING,
    SIZE_SECTION_STACK_SPACING,
    TEXT_MAIN,
)
from spektrafilm_gui.options import RawHighlightMode, RawWhiteBalance
from spektrafilm_gui.widget_editors import BoolEditor, EnumEditor, FloatEditor, FloatTupleEditor, IntEditor, IntTupleEditor, ProfileEnumEditor
from spektrafilm_gui.widget_primitives import CollapsibleSection, normalize_ui_text as _normalize_ui_text


LOAD_RAW_FIELDS = (
    ParamSpec(
        'white_balance',
        label='White balance',
        tooltip='Select white balance settings, if custom you can tune temperature and tint',
        enum=RawWhiteBalance,
    ),
    ParamSpec(
        'temperature',
        label='Temperature',
        tooltip='Temperature in Kelvin for the custom whitebalance, not used for the other white balance settings',
        min=1000,
        step=100,
    ),
    ParamSpec(
        'tint',
        label='Tint',
        tooltip='Tint value for the custom white balance, not used for the other white balance settings',
        min=0,
        step=0.01,
    ),
    ParamSpec(
        'raw_exposure_ev',
        label='Raw exposure',
        tooltip='Pre-demosaic raw exposure shift in EV stops (-4.0 to +4.0). Darkening pulls unclipped sensor highlights into recoverable range',
        min=-4.0,
        max=4.0,
        step=0.1,
    ),
    ParamSpec(
        'highlight_recovery',
        label='Highlight recovery',
        tooltip='Highlight preservation and recovery strength (0.0 = linear, 1.0 = maximum preservation and roll-off)',
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    ParamSpec(
        'highlight_mode',
        label='Highlight mode',
        tooltip='Algorithm to reconstruct clipped highlights from unclipped channels (reconstruct, blend, clip, ignore)',
        enum=RawHighlightMode,
    ),
    ParamSpec('lens_correction', label='Lens correction', tooltip='Apply lens corrections'),
    ParamSpec(
        'chroma_denoise',
        label='Chroma denoise',
        tooltip='Apply edge-preserving guided chrominance denoising to reduce sensor color noise',
    ),
    ParamSpec(
        'chroma_denoise_strength',
        label='Denoise strength',
        tooltip='Strength of chrominance denoising (0.0 = off, 1.0 = standard, up to 2.0 = strong)',
        min=0.0,
        max=2.0,
        step=0.1,
    ),
)

_LOAD_RAW_FIELD_SPECS = {spec.leaf: spec for spec in LOAD_RAW_FIELDS}
_SECTION_FIELD_SPECS = {
    'load_raw': _LOAD_RAW_FIELD_SPECS,
    'simulation': {spec.leaf: spec for spec in SIMULATION_FIELDS},
}
_AUXILIARY_FIELD_SPECS = {
    'scan_for_print': ParamSpec(
        'scan_for_print',
        label='Scan for print',
        tooltip='Scan the image for print, ie white and black correction of the scanner are active, and glare is deactivated.',
    ),
}
_SIMULATION_ACTION_BUTTON_SPECS = {
    'preview': {
        'text': 'PREVIEW',
        'tooltip': 'run the simulation on a small preview and deactivates grain, halation, blurs, unsharp mask (diffusion filters are active)',
        'preserve_case': True,
    },
    'scan': {
        'text': 'SCAN',
        'tooltip': 'Run the full simulation on the full-resolution input',
        'preserve_case': True,
    },
    'save': {
        'text': 'SAVE',
        'tooltip': 'Save the current output layer to an image file',
        'preserve_case': True,
    },
}


def _enum_values(enum_cls):
    return [member.value for member in enum_cls]


def _field_spec(section_name: str, field_name: str) -> ParamSpec | None:
    return _SECTION_FIELD_SPECS.get(section_name, {}).get(field_name)


def _build_collapsible_form_section(
    title: str,
    form: QFormLayout,
    *,
    expanded: bool,
) -> QVBoxLayout:
    content = QWidget()
    content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    content.setLayout(form)

    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(CollapsibleSection(title, content, expanded=expanded))
    return root


def _new_form_layout() -> QFormLayout:
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
    return form


def _add_form_rows(form: QFormLayout, rows: list[tuple[str | QLabel, QWidget]]) -> None:
    for label, widget in rows:
        if isinstance(label, str):
            label = _normalize_ui_text(label)
        form.addRow(label, widget)


def _build_linked_form_section(
    title: str,
    rows: list[tuple[str | QLabel, QWidget]],
    *,
    expanded: bool,
) -> QVBoxLayout:
    form = _new_form_layout()
    _add_form_rows(form, rows)
    return _build_collapsible_form_section(title, form, expanded=expanded)


def _build_button(
    text: str,
    callback: Any,
    *,
    tooltip: str | None = None,
    preserve_case: bool = False,
    role: str | None = None,
) -> QPushButton:
    button = QPushButton(text if preserve_case else _normalize_ui_text(text))
    if role is not None:
        button.setProperty('role', role)
    if tooltip:
        button.setToolTip(tooltip)
    button.clicked.connect(callback)
    return button


def _build_widget_label(section_name: str, field_name: str) -> QLabel:
    spec = _field_spec(section_name, field_name)
    label_text = (spec.label if spec is not None else None) or _format_label(field_name)
    label = QLabel(_normalize_ui_text(label_text))
    if spec is not None and spec.tooltip:
        label.setToolTip(spec.tooltip)
    return label


def _build_auxiliary_label(name: str) -> QLabel:
    spec = _AUXILIARY_FIELD_SPECS.get(name)
    label_text = (spec.label if spec is not None else None) or name.replace("_", " ")
    label = QLabel(_normalize_ui_text(label_text))
    if spec is not None and spec.tooltip:
        label.setToolTip(spec.tooltip)
    return label


def _build_button_row(*widgets: QWidget, stretch: int | None = None, spacing: int = 6) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        if stretch is None:
            row.addWidget(widget)
        else:
            row.addWidget(widget, stretch)
    return row


def _build_vertical_container(*items: QHBoxLayout | QFormLayout | QWidget, spacing: int = 6) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    layout.setAlignment(Qt.AlignTop)
    for item in items:
        if isinstance(item, QWidget):
            layout.addWidget(item)
        else:
            layout.addLayout(item)
    return container


def _set_single_collapsible_layout(widget: QWidget, title: str, content: QWidget, *, expanded: bool = True) -> None:
    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(CollapsibleSection(_normalize_ui_text(title), content, expanded=expanded))
    widget.setLayout(root)


def _format_label(field_name: str) -> str:
    return _normalize_ui_text(field_name.replace("_", " "))


def _apply_numeric_attr(widget: QWidget, method_name: str, value: float | int) -> None:
    method = getattr(widget, method_name, None)
    if callable(method):
        method(value)
        return
    editors = getattr(widget, '_editors', None)
    if editors is not None:
        for editor in editors:
            getattr(editor, method_name)(value)


def _editor_from_param_spec(annotation: Any, spec: ParamSpec, *, label: str) -> QWidget:
    """Build an editor for a ParamSpec given the field's runtime annotation.

    Shared by the group-bound (ParamsGroupSection) and path-bound (input_image)
    sections so the type -> editor mapping lives in one place.
    """
    decimals = 2 if spec.decimals is None else spec.decimals
    if spec.enum is not None:
        if spec.leaf in {'film_stock', 'print_paper'}:
            editor = ProfileEnumEditor([member.value for member in spec.enum])
        else:
            editor = EnumEditor([member.value for member in spec.enum])
    elif annotation is bool:
        editor = BoolEditor()
    elif annotation is int:
        editor = IntEditor()
    elif annotation is float:
        editor = FloatEditor(decimals=decimals)
    elif get_origin(annotation) is tuple:
        element_types = get_args(annotation)
        if element_types and all(element_type is int for element_type in element_types):
            editor = IntTupleEditor(len(element_types))
        else:
            editor = FloatTupleEditor(len(element_types), decimals=decimals)
    else:
        raise TypeError(f"Unsupported field type for {label}: {annotation!r}")
    if spec.min is not None:
        _apply_numeric_attr(editor, 'setMinimum', spec.min)
    if spec.max is not None:
        _apply_numeric_attr(editor, 'setMaximum', spec.max)
    if spec.step is not None:
        _apply_numeric_attr(editor, 'setSingleStep', spec.step)
    return editor


def _path_annotation(root_cls: type, path: str) -> Any:
    """Resolve the type annotation of a dotted path on a dataclass tree."""
    cls = root_cls
    annotation: Any = root_cls
    for part in path.split('.'):
        annotation = get_type_hints(cls)[part]
        cls = annotation
    return annotation


def _read_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def _write_path(root: Any, path: str, value: Any) -> None:
    parts = path.split('.')
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _build_path_panel(
    title: str,
    specs: tuple[ParamSpec, ...],
    editors: dict[str, QWidget],
    *,
    expanded: bool,
) -> QVBoxLayout:
    """Lay out a collapsible panel from shared, path-bound editors (by leaf)."""
    form = _new_form_layout()
    for spec in specs:
        editor = editors[spec.leaf]
        label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
        if spec.tooltip:
            label.setToolTip(spec.tooltip)
            editor.setToolTip(spec.tooltip)
        form.addRow(label, editor)
    return _build_collapsible_form_section(title, form, expanded=expanded)


class InputImageSection(QWidget):
    """Owner of the input_image cluster (io.* + settings.*), rendered across
    three panels on two tabs (Input here; Crop and Spectral embed this section's
    editors). Path-bound: each editor binds to a dotted path on InputImageState.
    Marked _is_params_group so auto-preview wiring and profile-sync drive it via
    _editors / set_state, exactly like ParamsGroupSection.
    """

    _is_params_group = True

    def __init__(self, filepicker_section: 'FilePickerSection'):
        super().__init__()
        self._filepicker_section = filepicker_section
        self._source: InputImageState | None = None
        self._specs = {spec.leaf: spec for spec in INPUT_IMAGE_FIELDS}
        self._editors: dict[str, QWidget] = {}
        for spec in INPUT_IMAGE_FIELDS:
            editor = _editor_from_param_spec(
                _path_annotation(InputImageState, spec.path), spec, label=spec.path,
            )
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
        self.setLayout(_build_path_panel('Input', INPUT_PANEL_FIELDS, self._editors, expanded=False))

    def set_state(self, state: InputImageState) -> None:
        self._source = state
        for spec in self._specs.values():
            self._editors[spec.leaf].value = _read_path(state, spec.path)

    def get_state(self) -> InputImageState:
        state = clone_state_section(self._source) if self._source is not None else InputImageState()
        for spec in self._specs.values():
            _write_path(state, spec.path, self._editors[spec.leaf].value)
        return state


class LoadRawSection(QWidget):
    load_requested = Signal(str)
    SECTION_NAME = 'load_raw'
    TITLE = 'Import Raw'
    _TYPE_HINTS = get_type_hints(LoadRawState)
    _STATE_FIELD_NAMES = tuple(_TYPE_HINTS)

    def __init__(self):
        super().__init__()
        self._source: LoadRawState | None = None
        self._detected_wb: tuple[float, float] | None = None
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText(_normalize_ui_text('No raw selected'))
        self.reprocess_button = _build_button(
            'reprocess raw',
            self._reprocess_raw,
            tooltip='Reprocess the RAW image applying the modified import settings (exposure, highlight recovery, white balance, etc.)',
            role='compactAction',
        )
        self.reprocess_button.setEnabled(False)
        self.convert_dng_button = _build_button(
            'Convert to DNG',
            self._convert_to_dng,
            tooltip='Convert the selected RAW file into a standard DNG-RAW using Adobe DNG Converter',
            role='compactAction',
        )
        self.convert_dng_button.setEnabled(False)
        self._build_ui()

    def _build_ui(self) -> None:
        for field_name in self._STATE_FIELD_NAMES:
            spec = _LOAD_RAW_FIELD_SPECS[field_name]
            editor = _editor_from_param_spec(self._TYPE_HINTS[field_name], spec, label=f'{self.SECTION_NAME}.{field_name}')
            setattr(self, field_name, editor)

        self.white_balance.currentTextChanged.connect(self._on_white_balance_mode_changed)

        form = _new_form_layout()
        browse_button = _build_button(
            'Select file',
            self._choose_file,
            tooltip='Load and process a raw file using rawpy, output colorspace and cctf as defined in current input widget state',
            role='compactAction',
        )
        form.addRow(_build_vertical_container(_build_button_row(self.file_path, browse_button, spacing=4), spacing=0))
        for field_name in self._STATE_FIELD_NAMES:
            form.addRow(_build_widget_label(self.SECTION_NAME, field_name), getattr(self, field_name))
        form.addRow(_build_button_row(self.reprocess_button, self.convert_dng_button, spacing=4))

        self.setLayout(_build_collapsible_form_section(self.TITLE, form, expanded=False))

    def _on_white_balance_mode_changed(self, mode: str) -> None:
        if mode == 'as_shot':
            if self._detected_wb is not None:
                self.temperature.value = self._detected_wb[0]
                self.tint.value = self._detected_wb[1]
            elif self.file_path.text().strip() and Path(self.file_path.text().strip()).is_file():
                try:
                    wb = read_raw_white_balance(self.file_path.text().strip())
                    self._detected_wb = wb
                    self.temperature.value = wb[0]
                    self.tint.value = wb[1]
                except Exception:
                    pass
        elif mode == 'daylight':
            self.temperature.value = 5500.0
            self.tint.value = 1.0
        elif mode == 'tungsten':
            self.temperature.value = 3200.0
            self.tint.value = 1.0

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, _normalize_ui_text('Select input raw'), load_dialog_dir('raw_input'))
        if not path:
            return
        save_dialog_dir('raw_input', str(Path(path).parent))
        self.set_path(path)
        self.load_requested.emit(path)

    def _reprocess_raw(self) -> None:
        path = self.file_path.text().strip()
        if not path:
            return
        self.load_requested.emit(path)

    def set_state(self, state: LoadRawState) -> None:
        self._source = state
        for field_name in self._STATE_FIELD_NAMES:
            getattr(self, field_name).value = getattr(state, field_name)
        if state.white_balance == 'as_shot':
            if self._detected_wb is not None:
                self.temperature.value = self._detected_wb[0]
                self.tint.value = self._detected_wb[1]
            elif self.file_path.text().strip() and Path(self.file_path.text().strip()).is_file():
                try:
                    wb = read_raw_white_balance(self.file_path.text().strip())
                    self._detected_wb = wb
                    self.temperature.value = wb[0]
                    self.tint.value = wb[1]
                except Exception:
                    pass

    def get_state(self) -> LoadRawState:
        state = clone_state_section(self._source) if self._source is not None else clone_state_section(PROJECT_DEFAULT_GUI_STATE.gui_only.load_raw)
        for field_name in self._STATE_FIELD_NAMES:
            setattr(state, field_name, getattr(self, field_name).value)
        return state

    def set_path(self, path: str) -> None:
        path_changed = self.file_path.text() != path
        self.file_path.setText(path)
        self.reprocess_button.setEnabled(bool(path.strip()))
        ext = Path(path).suffix.lower() if path.strip() else ""
        from spektrafilm_gui.gallery_thumbnails import RAW_EXTENSIONS
        self.convert_dng_button.setEnabled(bool(path.strip()) and ext in RAW_EXTENSIONS and ext != '.dng')
        if path_changed or self._detected_wb is None:
            if path.strip() and Path(path).is_file():
                try:
                    temp, tint = read_raw_white_balance(path)
                    self._detected_wb = (temp, tint)
                    if self.white_balance.value == 'as_shot':
                        self.temperature.value = temp
                        self.tint.value = tint
                except Exception:
                    pass
            else:
                self._detected_wb = None

    def _convert_to_dng(self) -> None:
        path = self.file_path.text().strip()
        if not path or not Path(path).is_file():
            return
        from spektrafilm.utils.raw_file_processor import convert_raw_to_dng
        try:
            dng_path = convert_raw_to_dng(path)
            self.set_path(str(dng_path))
            self.load_requested.emit(str(dng_path))
        except Exception as exc:
            from qtpy.QtWidgets import QMessageBox
            from spektrafilm_gui.napari_layout import dialog_parent
            QMessageBox.critical(dialog_parent(self), 'Convert to DNG', f'Failed to convert RAW to DNG:\n\n{exc}')


class PreviewCropSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_path_panel('Crop and upscale', CROP_PANEL_FIELDS, input_image_section._editors, expanded=False),
        )


class ParamsGroupSection(QWidget):
    """A GUI panel bound directly to a runtime parameter group.

    Driven by a :class:`GroupManifest` keyed on ``RuntimePhotoParams``
    paths. Builds one editor per declared field (type inferred from the
    group dataclass), reads and writes the runtime group object directly
    via ``get_state`` / ``set_state``, and passes through any group
    fields the manifest does not declare. This replaces the per-panel
    mirror dataclass + dedicated presentation table + mapper block trio
    with a single declaration.
    """

    # Marks this as a path-bound group section so profile-sync drives it
    # through set_state (applying unit transforms) rather than assigning
    # raw runtime values field-by-field.
    _is_params_group = True

    def __init__(self, manifest: GroupManifest):
        super().__init__()
        self._manifest = manifest
        self._group_cls = manifest.group_cls
        self._type_hints = get_type_hints(manifest.group_cls)
        self._source: Any = None
        self._editors: dict[str, QWidget] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        form = _new_form_layout()
        for spec in self._manifest.fields:
            editor = self._build_editor(spec)
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
        # Build editors for every field (so the whole group stays editable,
        # persisted, and auto-preview wired) but only lay out panel_fields;
        # any remaining fields are displayed by a section that borrows them.
        for spec in self._manifest.panel_fields or self._manifest.fields:
            editor = self._editors[spec.leaf]
            label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
                editor.setToolTip(spec.tooltip)
            form.addRow(label, editor)

        content = QWidget()
        content.setLayout(form)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(
            CollapsibleSection(self._manifest.title, content, expanded=not self._manifest.collapsed_by_default),
        )
        self.setLayout(root)

    def _build_editor(self, spec: ParamSpec) -> QWidget:
        annotation = self._type_hints[spec.leaf]
        return _editor_from_param_spec(annotation, spec, label=f"{self._group_cls.__name__}.{spec.leaf}")

    def set_state(self, group: Any) -> None:
        self._source = group
        for leaf, editor in self._editors.items():
            editor.value = getattr(group, leaf)

    def get_state(self) -> Any:
        base = self._source if self._source is not None else self._group_cls()
        overrides = {leaf: editor.value for leaf, editor in self._editors.items()}
        return replace(base, **overrides)


class ChemistrySection(ParamsGroupSection):
    """Chemistry section combining an interactive YRGB curve editor with the fine-tuning parameter sliders."""

    def __init__(self, manifest: GroupManifest):
        self._syncing_curve = False
        self.curve_editor: YRGBCurveEditor = YRGBCurveEditor()
        super().__init__(manifest)
        self.set_state(self._group_cls())

    def _build_ui(self) -> None:
        for spec in self._manifest.fields:
            editor = self._build_editor(spec)
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)

        # Wire chemistry sliders to update the curve editor live
        for leaf, ed in self._editors.items():
            if leaf == 'active':
                continue
            sig = getattr(ed, 'valueChanged', None)
            if sig is not None and hasattr(sig, 'connect'):
                sig.connect(self._on_slider_changed)

        # Wire curve editor interaction to update the sliders live
        self.curve_editor.chemistry_changed.connect(self._on_curve_editor_changed)

        # Active checkbox row at the top
        top_form = _new_form_layout()
        if 'active' in self._editors:
            active_spec = next((s for s in self._manifest.fields if s.leaf == 'active'), None)
            active_label = (
                QLabel(_normalize_ui_text(active_spec.label or _format_label('active')))
                if active_spec
                else QLabel(_normalize_ui_text('active'))
            )
            if active_spec and active_spec.tooltip:
                active_label.setToolTip(active_spec.tooltip)
            top_form.addRow(active_label, self._editors['active'])

        # Sliders form below the curve editor
        sliders_form = _new_form_layout()
        for spec in self._manifest.panel_fields or self._manifest.fields:
            if spec.leaf == 'active':
                continue
            editor = self._editors[spec.leaf]
            label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
                editor.setToolTip(spec.tooltip)
            sliders_form.addRow(label, editor)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        content_layout.addLayout(top_form)
        content_layout.addWidget(self.curve_editor)
        content_layout.addLayout(sliders_form)

        content = QWidget()
        content.setLayout(content_layout)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(
            CollapsibleSection(self._manifest.title, content, expanded=not self._manifest.collapsed_by_default),
        )
        self.setLayout(root)

    def _on_slider_changed(self, *_args: Any) -> None:
        if self._syncing_curve or self.curve_editor is None:
            return
        self._syncing_curve = True
        try:
            self.curve_editor.set_chemistry_params(self.get_state())
        finally:
            self._syncing_curve = False

    def _on_curve_editor_changed(self, updates: dict[str, float]) -> None:
        if self._syncing_curve:
            return
        self._syncing_curve = True
        try:
            for leaf, val in updates.items():
                ed = self._editors.get(leaf)
                if ed is not None:
                    ed.value = val
        finally:
            self._syncing_curve = False

    def set_state(self, group: Any) -> None:
        super().set_state(group)
        if self.curve_editor is not None:
            self._syncing_curve = True
            try:
                self.curve_editor.set_chemistry_params(group)
            finally:
                self._syncing_curve = False

    def set_histogram_image(self, image: Any) -> None:
        """Forward image data to the curve editor to update background live histogram."""
        if hasattr(self, "curve_editor") and hasattr(self.curve_editor, "set_histogram_image"):
            self.curve_editor.set_histogram_image(image)


class SpecialSection(QWidget):
    _is_params_group = True

    def __init__(self, simulation_section: 'SimulationSection'):
        super().__init__()
        self._simulation_section = simulation_section
        self._source: SpecialState | None = None
        self._specs = {spec.leaf: spec for spec in SPECIAL_FIELDS}
        self._editors: dict[str, QWidget] = {}
        form = _new_form_layout()
        borrowed = SIMULATION_SPECIAL_BORROWED_FIELDS[0]
        borrowed_label = QLabel(_normalize_ui_text(borrowed.label or _format_label(borrowed.leaf)))
        if borrowed.tooltip:
            borrowed_label.setToolTip(borrowed.tooltip)
        form.addRow(borrowed_label, simulation_section.print_illuminant)
        for spec in SPECIAL_FIELDS:
            editor = _editor_from_param_spec(_path_annotation(SpecialState, spec.path), spec, label=spec.path)
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
            if spec.leaf == 'film_gamma_factor':
                continue
            label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
                editor.setToolTip(spec.tooltip)
            form.addRow(label, editor)
        self.setLayout(_build_collapsible_form_section('Experimental', form, expanded=False))

    def set_state(self, state: SpecialState) -> None:
        self._source = state
        for spec in self._specs.values():
            self._editors[spec.leaf].value = _read_path(state, spec.path)

    def get_state(self) -> SpecialState:
        state = clone_state_section(self._source) if self._source is not None else SpecialState(
            film_channel_swap=(0, 1, 2),
            print_channel_swap=(0, 1, 2),
        )
        for spec in self._specs.values():
            _write_path(state, spec.path, self._editors[spec.leaf].value)
        return state


class SpectralUpsamplingSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_path_panel('Spectral upsampling', SPECTRAL_PANEL_FIELDS, input_image_section._editors, expanded=False),
        )


class TuneSection(QWidget):
    def __init__(self, special_section: SpecialSection):
        super().__init__()
        self.setLayout(
            _build_path_panel('Tune', TUNE_PANEL_FIELDS, special_section._editors, expanded=True),
        )


class ComputePrecisionSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_path_panel('Compute precision', PRECISION_PANEL_FIELDS, input_image_section._editors, expanded=True),
        )


class FilePickerSection(QWidget):
    load_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText(_normalize_ui_text('No image selected'))

        browse_button = _build_button('Select file', self._choose_file, role='compactAction')
        content = _build_vertical_container(_build_button_row(self.file_path, browse_button, spacing=4), spacing=6)
        _set_single_collapsible_layout(self, 'Import RGB', content, expanded=False)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, _normalize_ui_text('Select input image'), load_dialog_dir('rgb_input'))
        if not path:
            return
        save_dialog_dir('rgb_input', str(Path(path).parent))
        self.file_path.setText(path)
        self.load_requested.emit(path)

    def set_path(self, path: str) -> None:
        self.file_path.setText(path)


class GuiConfigSection(QWidget):
    save_current_as_default_requested = Signal()
    save_current_to_file_requested = Signal()
    load_from_file_requested = Signal()
    restore_factory_default_requested = Signal()

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        self.save_current_as_default_button = _build_button(
            'Save current as default',
            self.save_current_as_default_requested.emit,
        )
        self.save_current_to_file_button = _build_button(
            'Save current to file',
            self.save_current_to_file_requested.emit,
        )
        self.load_from_file_button = _build_button('Load from file', self.load_from_file_requested.emit)
        self.restore_factory_default_button = _build_button(
            'Restore factory default',
            self.restore_factory_default_requested.emit,
        )

        content = _build_vertical_container(
            _build_button_row(self.save_current_as_default_button, self.save_current_to_file_button),
            _build_button_row(self.load_from_file_button, self.restore_factory_default_button),
        )
        _set_single_collapsible_layout(self, 'GUI parameters', content, expanded=True)


class DisplaySection(QWidget):
    _is_params_group = True
    _skip_auto_preview_leaves = {'preview_max_size', 'output_interpolation'}
    update_preview_requested = Signal()

    def __init__(self):
        super().__init__()
        self._source: DisplayState | None = None
        self._specs = {spec.leaf: spec for spec in DISPLAY_PANEL_FIELDS}
        self._editors: dict[str, QWidget] = {}
        self.update_preview_button = _build_button(
            'update',
            self.update_preview_requested.emit,
            preserve_case=True,
            role='compactAction',
        )
        self.update_preview_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        compute_form = _new_form_layout()
        display_form = _new_form_layout()

        from spektrafilm.cuda.device import is_cuda_available, get_device_info
        if is_cuda_available():
            info = get_device_info()
            dev_name = info.get("device_name", "CUDA GPU")
            mem_gb = info.get("total_memory_gb", 0.0)
            self._hw_status_text = f"{dev_name} ({mem_gb:.1f} GB)"
            self.hardware_label = QLabel(self._hw_status_text)
            self.hardware_label.setStyleSheet("color: #4ade80; font-weight: 500;")
            self.hardware_label.setToolTip(
                f"NVIDIA CUDA acceleration ACTIVE\nDevice: {dev_name}\nVRAM: {mem_gb:.1f} GB\n\n"
                "Note: In Windows Task Manager, CUDA compute usage appears under the 'Compute_0' or 'CUDA' engine graph, "
                "not the default '3D' graph."
            )
        else:
            info = get_device_info()
            err = info.get("error")
            self._hw_status_text = "CPU (multi-core)"
            self.hardware_label = QLabel(self._hw_status_text)
            self.hardware_label.setStyleSheet("color: #facc15; font-weight: 500;")
            tooltip = "GPU acceleration not available (using multi-core CPU)"
            if err:
                tooltip += f"\nReason: {err}"
            self.hardware_label.setToolTip(tooltip)


        for spec in DISPLAY_PANEL_FIELDS:
            editor = _editor_from_param_spec(_path_annotation(DisplayState, spec.path), spec, label=spec.path)
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
            label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
                editor.setToolTip(spec.tooltip)
            widget: QWidget = editor
            if spec.leaf == 'preview_max_size':
                widget = _build_vertical_container(_build_button_row(editor, self.update_preview_button, spacing=4), spacing=0)

            if spec.leaf == 'device':
                compute_form.addRow(label, widget)
                hw_lbl = QLabel(_normalize_ui_text("detected gpu"))
                hw_lbl.setToolTip("Hardware acceleration device detected on this system")
                compute_form.addRow(hw_lbl, self.hardware_label)
            else:
                display_form.addRow(label, widget)

        compute_content = QWidget()
        compute_content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        compute_content.setLayout(compute_form)
        self.compute_section = CollapsibleSection(_normalize_ui_text('Compute device'), compute_content, expanded=True)

        display_content = QWidget()
        display_content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        display_content.setLayout(display_form)
        self.display_section = CollapsibleSection(_normalize_ui_text('Display'), display_content, expanded=True)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(int(SIZE_SECTION_STACK_SPACING.removesuffix('px')))
        root.addWidget(self.compute_section)
        root.addWidget(self.display_section)
        self.setLayout(root)

    def set_state(self, state: DisplayState) -> None:
        self._source = state
        for spec in self._specs.values():
            self._editors[spec.leaf].value = _read_path(state, spec.path)

    def get_state(self) -> DisplayState:
        state = clone_state_section(self._source) if self._source is not None else DisplayState(
            use_display_transform=True,
            gray_18_canvas=True,
            white_padding=0.03,
            output_interpolation='spline36',
        )
        for spec in self._specs.values():
            _write_path(state, spec.path, self._editors[spec.leaf].value)
        return state


class SimulationSection(QWidget):
    _is_params_group = True
    _skip_auto_preview_leaves = {'auto_preview', 'scan_film'}
    preview_requested = Signal()
    scan_requested = Signal()
    save_requested = Signal((str,), ())
    _glare_section: 'ParamsGroupSection | None'
    _scanner_section: 'ParamsGroupSection | None'
    _scan_for_print_restore_state: dict[str, object] | None

    def __init__(self):
        super().__init__()
        self._source: SimulationState | None = None
        self._specs = {spec.leaf: spec for spec in SIMULATION_FIELDS}
        self._editors: dict[str, QWidget] = {}
        self._glare_section = None
        self._scanner_section = None
        self._scan_for_print_restore_state = None
        for spec in SIMULATION_FIELDS:
            editor = _editor_from_param_spec(_path_annotation(SimulationState, spec.path), spec, label=spec.path)
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
        self.bottom_auto_preview = self.auto_preview
        self.bottom_scan_film = self.scan_film
        self.bottom_scan_for_print = BoolEditor()
        scan_for_print_spec = _AUXILIARY_FIELD_SPECS['scan_for_print']
        if scan_for_print_spec.tooltip:
            self.bottom_scan_for_print.setToolTip(scan_for_print_spec.tooltip)
        self.bottom_scan_for_print.toggled.connect(self._apply_scan_for_print_mode)
        preview_button_spec = _SIMULATION_ACTION_BUTTON_SPECS['preview']
        self.preview_button = _build_button(
            preview_button_spec['text'],
            self.preview_requested.emit,
            tooltip=preview_button_spec['tooltip'],
            preserve_case=preview_button_spec['preserve_case'],
            role='accentAction',
        )
        scan_button_spec = _SIMULATION_ACTION_BUTTON_SPECS['scan']
        self.scan_button = _build_button(
            scan_button_spec['text'],
            self.scan_requested.emit,
            tooltip=scan_button_spec['tooltip'],
            preserve_case=scan_button_spec['preserve_case'],
            role='accentAction',
        )
        save_button_spec = _SIMULATION_ACTION_BUTTON_SPECS['save']
        self.save_button = QPushButton(save_button_spec['text'])
        self.save_button.setProperty('role', 'accentAction')
        if save_button_spec.get('tooltip'):
            self.save_button.setToolTip(save_button_spec['tooltip'])

        self.save_menu = QMenu(self.save_button)
        self.save_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {GRAY_1};
                color: {TEXT_MAIN};
                border: 1px solid {GRAY_3};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 4px 18px 4px 14px;
                border-radius: 5px;
            }}
            QMenu::item:selected {{
                background-color: {ACCENT_COLOR_TEXT};
                color: #000000;
            }}
        """)
        for fmt, label in [
            ('tif', 'TIF (.tif)'),
            ('png', 'PNG (.png)'),
            ('jpg', 'JPG (.jpg)'),
            ('webp', 'WEBP (.webp)'),
        ]:
            action = self.save_menu.addAction(label)
            action.triggered.connect(lambda checked=False, f=fmt: self.save_requested.emit(f))

        self.save_button.setMenu(self.save_menu)
        self.save_button.clicked.connect(self.save_button.showMenu)

        scan_film_row = QHBoxLayout()
        scan_film_row.setContentsMargins(0, 0, 0, 0)
        scan_film_row.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_widget_label('simulation', 'auto_preview'))
        scan_film_row.addWidget(self.bottom_auto_preview)
        scan_film_row.addSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_widget_label('simulation', 'scan_film'))
        scan_film_row.addWidget(self.bottom_scan_film)
        scan_film_row.addSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_auxiliary_label('scan_for_print'))
        scan_film_row.addWidget(self.bottom_scan_for_print)
        scan_film_row.addStretch(1)

        action_buttons = QWidget()
        action_buttons.setLayout(
            _build_button_row(
                self.preview_button,
                self.scan_button,
                self.save_button,
                stretch=1,
                spacing=SIZE_FOOTER_ITEM_SPACING,
            ),
        )

        self.bottom_bar = QWidget()
        bottom_bar_layout = QVBoxLayout(self.bottom_bar)
        bottom_bar_layout.setContentsMargins(0, 0, 0, 0)
        bottom_bar_layout.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        bottom_bar_layout.addLayout(scan_film_row)
        bottom_bar_layout.addWidget(action_buttons)
        self.setLayout(_build_path_panel('Profiles', SIMULATION_PROFILE_PANEL_FIELDS, self._editors, expanded=True))

    def set_state(self, state: SimulationState) -> None:
        self._source = state
        for spec in self._specs.values():
            self._editors[spec.leaf].value = _read_path(state, spec.path)

    def get_state(self) -> SimulationState:
        state = clone_state_section(self._source) if self._source is not None else SimulationState(
            selection=SelectionState(film_stock='', print_paper=''),
        )
        for spec in self._specs.values():
            _write_path(state, spec.path, self._editors[spec.leaf].value)
        return state

    def action_bar(self) -> QWidget:
        return self.bottom_bar

    def set_auto_preview_value(self, value: bool) -> None:
        self.bottom_auto_preview.setChecked(value)

    def auto_preview_value(self) -> bool:
        return self.bottom_auto_preview.isChecked()

    def set_scan_film_value(self, value: bool) -> None:
        self.bottom_scan_film.setChecked(value)

    def scan_film_value(self) -> bool:
        return self.bottom_scan_film.isChecked()

    def bind_scan_for_print_sections(
        self,
        *,
        glare: 'ParamsGroupSection',
        scanner: 'ParamsGroupSection',
    ) -> None:
        self._glare_section = glare
        self._scanner_section = scanner

    def reset_scan_for_print_value(self) -> None:
        was_blocked = self.bottom_scan_for_print.blockSignals(True)
        self.bottom_scan_for_print.setChecked(False)
        self.bottom_scan_for_print.blockSignals(was_blocked)
        self._scan_for_print_restore_state = None

    def _apply_scan_for_print_mode(self, active: bool) -> None:
        scanner = self._scanner_section
        if scanner is None:
            return
        if active:
            if self._scan_for_print_restore_state is None:
                self._scan_for_print_restore_state = {
                    'white_correction': scanner.white_correction.value,
                    'black_correction': scanner.black_correction.value,
                    'glare_active': None if self._glare_section is None else self._glare_section.active.value,
                }
            scanner.white_correction.value = True
            scanner.black_correction.value = True
            if self._glare_section is not None:
                self._glare_section.active.value = False
            return

        restore_state = self._scan_for_print_restore_state
        if restore_state is None:
            return
        scanner.white_correction.value = restore_state['white_correction']
        scanner.black_correction.value = restore_state['black_correction']
        glare_active = restore_state['glare_active']
        if self._glare_section is not None and glare_active is not None:
            self._glare_section.active.value = glare_active
        self._scan_for_print_restore_state = None


class OutputSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_path_panel('Output', SIMULATION_OUTPUT_PANEL_FIELDS, simulation_section._editors, expanded=False),
        )


class ExposureControlSection(QWidget):
    def __init__(self, simulation_section: SimulationSection, camera_section: 'ParamsGroupSection'):
        super().__init__()
        # auto_exposure / exposure_compensation_ev are CameraParams fields
        # owned by camera_section; the print-exposure fields are owned by
        # simulation_section. Render the camera pair on top, then the print
        # controls, from a merged editor lookup (leaves do not collide).
        editors = {**camera_section._editors, **simulation_section._editors}
        fields = CAMERA_EXPOSURE_BORROWED_FIELDS + SIMULATION_EXPOSURE_PANEL_FIELDS
        self.setLayout(_build_path_panel('Exposure control', fields, editors, expanded=True))


class EnlargerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_path_panel('Enlarger', SIMULATION_ENLARGER_PANEL_FIELDS, simulation_section._editors, expanded=True),
        )