from __future__ import annotations

from typing import TYPE_CHECKING
from qtpy.QtCore import QObject, Signal

from spektrafilm_gui.state import GuiState, clone_gui_state


class HistoryManager(QObject):
    """Manages an undo/redo stack of GuiState snapshots."""

    can_undo_changed = Signal(bool)
    can_redo_changed = Signal(bool)
    history_changed = Signal()

    def __init__(self, max_entries: int = 60, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._max_entries = max_entries
        self._undo_stack: list[GuiState] = []
        self._redo_stack: list[GuiState] = []

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 1

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_count(self) -> int:
        """Number of undo steps available."""
        return max(0, len(self._undo_stack) - 1)

    @property
    def redo_count(self) -> int:
        """Number of redo steps available."""
        return len(self._redo_stack)

    def current_state(self) -> GuiState | None:
        if not self._undo_stack:
            return None
        return self._undo_stack[-1]

    def reset(self, initial_state: GuiState | None = None) -> None:
        """Clear all undo/redo history and optionally initialize with an initial state."""
        prev_can_undo = self.can_undo
        prev_can_redo = self.can_redo
        self._undo_stack.clear()
        self._redo_stack.clear()
        if initial_state is not None:
            self._undo_stack.append(clone_gui_state(initial_state))
        self._notify_changes(prev_can_undo, prev_can_redo)

    def push(self, state: GuiState) -> bool:
        """Record a new state snapshot.

        Returns True if the state was recorded, or False if the state is identical
        to the top of the stack.
        """
        cloned = clone_gui_state(state)
        if self._undo_stack and self._undo_stack[-1] == cloned:
            return False

        prev_can_undo = self.can_undo
        prev_can_redo = self.can_redo

        self._undo_stack.append(cloned)
        if len(self._undo_stack) > self._max_entries:
            del self._undo_stack[0 : len(self._undo_stack) - self._max_entries]

        self._redo_stack.clear()
        self._notify_changes(prev_can_undo, prev_can_redo)
        return True

    def undo(self) -> GuiState | None:
        """Pop the current state onto redo stack and return the previous state to restore."""
        if not self.can_undo:
            return None

        prev_can_undo = self.can_undo
        prev_can_redo = self.can_redo

        current = self._undo_stack.pop()
        self._redo_stack.append(current)

        target = self._undo_stack[-1]
        self._notify_changes(prev_can_undo, prev_can_redo)
        return clone_gui_state(target)

    def redo(self) -> GuiState | None:
        """Pop the redo stack onto undo stack and return the redone state to restore."""
        if not self.can_redo:
            return None

        prev_can_undo = self.can_undo
        prev_can_redo = self.can_redo

        target = self._redo_stack.pop()
        self._undo_stack.append(target)
        self._notify_changes(prev_can_undo, prev_can_redo)
        return clone_gui_state(target)

    def _notify_changes(self, prev_can_undo: bool, prev_can_redo: bool) -> None:
        new_can_undo = self.can_undo
        new_can_redo = self.can_redo
        if new_can_undo != prev_can_undo:
            self.can_undo_changed.emit(new_can_undo)
        if new_can_redo != prev_can_redo:
            self.can_redo_changed.emit(new_can_redo)
        self.history_changed.emit()
