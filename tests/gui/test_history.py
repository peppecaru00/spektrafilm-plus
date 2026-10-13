from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qtpy import QtCore, QtWidgets

from spektrafilm_gui.history_manager import HistoryManager
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state
from spektrafilm_gui.widgets import create_widget_bundle
from spektrafilm_gui.controller import GuiController
from spektrafilm_gui.app import build_main_window_for_app, connect_auto_preview_signals

_q_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_history_manager_initial_state() -> None:
    mgr = HistoryManager(max_entries=10)
    assert not mgr.can_undo
    assert not mgr.can_redo
    assert mgr.undo_count == 0
    assert mgr.redo_count == 0
    assert mgr.current_state() is None
    assert mgr.undo() is None
    assert mgr.redo() is None


def test_history_manager_reset_with_initial_state() -> None:
    mgr = HistoryManager(max_entries=10)
    s0 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    mgr.reset(s0)

    assert not mgr.can_undo
    assert not mgr.can_redo
    assert mgr.undo_count == 0
    assert mgr.current_state() == s0


def test_history_manager_push_undo_redo_flow() -> None:
    mgr = HistoryManager(max_entries=10)
    s0 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    mgr.reset(s0)

    undo_signals: list[bool] = []
    redo_signals: list[bool] = []
    history_signals: list[int] = []

    mgr.can_undo_changed.connect(undo_signals.append)
    mgr.can_redo_changed.connect(redo_signals.append)
    mgr.history_changed.connect(lambda: history_signals.append(1))

    # Push state 1 (modify grain)
    s1 = clone_gui_state(s0)
    s1.grain.blur = 0.2
    assert mgr.push(s1) is True
    assert mgr.can_undo is True
    assert mgr.can_redo is False
    assert mgr.undo_count == 1
    assert undo_signals == [True]
    assert redo_signals == []
    assert len(history_signals) == 1

    # Push identical state 1 (should be ignored)
    assert mgr.push(s1) is False
    assert mgr.undo_count == 1
    assert len(history_signals) == 1

    # Push state 2 (modify halation)
    s2 = clone_gui_state(s1)
    s2.halation.scatter_amount = 0.5
    assert mgr.push(s2) is True
    assert mgr.undo_count == 2
    assert mgr.can_undo is True

    # Undo back to s1
    restored1 = mgr.undo()
    assert restored1 is not None
    assert restored1.halation.scatter_amount == s1.halation.scatter_amount
    assert restored1.grain.blur == s1.grain.blur
    assert mgr.can_undo is True
    assert mgr.can_redo is True
    assert mgr.undo_count == 1
    assert mgr.redo_count == 1
    assert redo_signals == [True]

    # Undo back to s0
    restored0 = mgr.undo()
    assert restored0 is not None
    assert restored0.grain.blur == s0.grain.blur
    assert mgr.can_undo is False
    assert mgr.can_redo is True
    assert mgr.undo_count == 0
    assert mgr.redo_count == 2
    assert undo_signals == [True, False]

    # Redo back to s1
    redone1 = mgr.redo()
    assert redone1 is not None
    assert redone1.grain.blur == s1.grain.blur
    assert mgr.can_undo is True
    assert mgr.can_redo is True
    assert mgr.undo_count == 1
    assert mgr.redo_count == 1

    # New branch: pushing state 3 after partial redo clears redo stack
    s3 = clone_gui_state(redone1)
    s3.grain.blur = 0.99
    assert mgr.push(s3) is True
    assert mgr.can_undo is True
    assert mgr.can_redo is False
    assert mgr.redo_count == 0


def test_history_manager_max_entries_eviction() -> None:
    mgr = HistoryManager(max_entries=3)
    s0 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    mgr.reset(s0)

    s1 = clone_gui_state(s0)
    s1.grain.blur = 0.1
    mgr.push(s1)

    s2 = clone_gui_state(s0)
    s2.grain.blur = 0.2
    mgr.push(s2)

    # Currently 3 items: [s0, s1, s2]
    assert mgr.undo_count == 2

    # Pushing 4th item evicts s0: [s1, s2, s3]
    s3 = clone_gui_state(s0)
    s3.grain.blur = 0.3
    mgr.push(s3)

    assert mgr.undo_count == 2
    u1 = mgr.undo()
    assert u1.grain.blur == 0.2
    u2 = mgr.undo()
    assert u2.grain.blur == 0.1
    assert mgr.undo() is None  # s0 was evicted


def test_history_manager_deep_clone_isolation() -> None:
    mgr = HistoryManager(max_entries=10)
    s0 = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    s0.grain.blur = 0.5
    mgr.reset(s0)

    # Mutate external object
    s0.grain.blur = 0.99
    assert mgr.current_state().grain.blur == 0.5

    # Mutate returned state from undo
    s1 = clone_gui_state(s0)
    s1.grain.blur = 0.7
    mgr.push(s1)
    restored = mgr.undo()
    restored.grain.blur = 0.123
    assert mgr.current_state().grain.blur == 0.5


from spektrafilm_gui.state_bridge import apply_gui_state


def _create_controller_with_widgets() -> tuple[GuiController, Any]:
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    viewer = SimpleNamespace(
        window=SimpleNamespace(qt_viewer=SimpleNamespace()),
        layers=SimpleNamespace(events=SimpleNamespace(connect=lambda *_: None)),
    )
    controller = GuiController(viewer=viewer, widgets=widgets)
    return controller, widgets


def test_controller_history_integration() -> None:
    controller, widgets = _create_controller_with_widgets()

    assert not controller.can_undo
    assert not controller.can_redo

    # Change a slider (e.g. grain blur)
    widgets.grain.blur.value = 0.12
    controller.request_auto_preview()

    # Before debounce timer fires:
    assert controller._history_record_timer.isActive()

    # Fast forward debounce timer by triggering timeout manually
    controller._history_record_timer.stop()
    controller._record_debounced_history_state()

    assert controller.can_undo is True
    assert not controller.can_redo

    # Undo the slider change
    assert controller.undo() is True
    assert widgets.grain.blur.value != 0.12
    assert not controller.can_undo
    assert controller.can_redo is True

    # Redo the slider change
    assert controller.redo() is True
    assert abs(widgets.grain.blur.value - 0.12) < 1e-4
    assert controller.can_undo is True
    assert not controller.can_redo


def test_controller_fast_undo_flushes_pending_debounce() -> None:
    controller, widgets = _create_controller_with_widgets()

    initial_blur = widgets.grain.blur.value

    # Change slider
    widgets.grain.blur.value = initial_blur + 0.25
    controller.request_auto_preview()
    assert controller._history_record_timer.isActive()

    # Immediate undo before timer times out
    assert controller.undo() is True
    assert not controller._history_record_timer.isActive()
    assert abs(widgets.grain.blur.value - initial_blur) < 1e-4


def test_controller_immediate_discrete_history_recording() -> None:
    controller, widgets = _create_controller_with_widgets()

    # Initial state
    assert not controller.can_undo

    # Changing profile defaults triggers immediate record
    widgets.simulation.film_stock.setCurrentText("Portra 400")
    widgets.simulation.print_paper.setCurrentText("Endura Premier")
    controller.apply_profile_defaults("Portra 400")

    assert controller.can_undo is True
    assert controller.undo() is True


def test_chemistry_curve_knot_undo_redo() -> None:
    controller, widgets = _create_controller_with_widgets()
    connect_auto_preview_signals(controller, widgets)

    initial_gamma = widgets.chemistry.gamma_factor.value

    # Modify curve editor / chemistry
    widgets.chemistry.gamma_factor.value = initial_gamma + 0.35
    controller.request_auto_preview()
    controller._history_record_timer.stop()
    controller._record_debounced_history_state()

    assert controller.can_undo is True
    controller.undo()
    assert abs(widgets.chemistry.gamma_factor.value - initial_gamma) < 1e-4

    controller.redo()
    assert abs(widgets.chemistry.gamma_factor.value - (initial_gamma + 0.35)) < 1e-4


def test_viewer_panel_undo_redo_buttons() -> None:
    from spektrafilm_gui.napari_layout import _build_viewer_panel

    viewer_widget = QtWidgets.QWidget()
    status_bar = QtWidgets.QStatusBar()

    undo_called = False
    redo_called = False

    def on_undo():
        nonlocal undo_called
        undo_called = True

    def on_redo():
        nonlocal redo_called
        redo_called = True

    panel = _build_viewer_panel(
        viewer_widget,
        status_bar,
        on_undo=on_undo,
        on_redo=on_redo,
    )

    assert hasattr(panel, 'undo_button')
    assert hasattr(panel, 'redo_button')
    assert panel.undo_button.text() == '↺ undo'
    assert panel.redo_button.text() == '↻ redo'
    assert not panel.undo_button.isEnabled()
    assert not panel.redo_button.isEnabled()

    panel.undo_button.setEnabled(True)
    panel.undo_button.click()
    assert undo_called is True

    panel.redo_button.setEnabled(True)
    panel.redo_button.click()
    assert redo_called is True
