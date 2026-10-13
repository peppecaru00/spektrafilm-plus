from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qtpy import QtCore
from qtpy.QtCore import QObject, Signal

from spektrafilm_gui.gallery_sidecar import has_sidecar
from spektrafilm_gui.state import GuiState, clone_gui_state

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = frozenset({
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".nef",
    ".orf",
    ".raf",
    ".rw2",
})

RGB_EXTENSIONS = frozenset({
    ".exr",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
})

SUPPORTED_IMAGE_EXTENSIONS = RAW_EXTENSIONS | RGB_EXTENSIONS


@dataclass(frozen=True, slots=True)
class GalleryItem:
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def is_raw(self) -> bool:
        return self.extension in RAW_EXTENSIONS

    @property
    def has_settings(self) -> bool:
        return has_sidecar(self.path)


class GalleryModel(QObject):
    items_changed = Signal()
    current_changed = Signal(int, object)  # (current_index, GalleryItem or None)
    selection_changed = Signal()           # emitted when multi-selection set changes
    item_status_updated = Signal(int)       # (index)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[GalleryItem] = []
        self._current_index: int = -1
        self._selected_indices: set[int] = set()
        self._clipboard_state: GuiState | None = None
        self._clipboard_param_id: str | None = None
        self._clipboard_param_value: Any = None
        self._clipboard_params_dict: dict[str, Any] = {}

    @property
    def items(self) -> list[GalleryItem]:
        return list(self._items)

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def current_item(self) -> GalleryItem | None:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index]
        return None

    def item_at(self, index: int) -> GalleryItem | None:
        """Return the GalleryItem at the given index, or None if out of bounds."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    @property
    def selected_indices(self) -> frozenset[int]:
        """Return the current set of multi-selected photo indices."""
        return frozenset(self._selected_indices)

    @property
    def selected_count(self) -> int:
        """Return the number of currently multi-selected photos."""
        return len(self._selected_indices)

    @property
    def clipboard_state(self) -> GuiState | None:
        return clone_gui_state(self._clipboard_state) if self._clipboard_state is not None else None

    @property
    def has_param_clipboard(self) -> bool:
        """Return True if any parameters are currently stored in the parameter clipboard."""
        return bool(self._clipboard_params_dict) or (self._clipboard_param_id is not None)

    @property
    def param_clipboard_id(self) -> str | None:
        if self._clipboard_param_id is not None:
            return self._clipboard_param_id
        if len(self._clipboard_params_dict) == 1:
            return next(iter(self._clipboard_params_dict.keys()))
        return "__multi__" if len(self._clipboard_params_dict) > 1 else None

    @property
    def param_clipboard_value(self) -> Any:
        if self._clipboard_param_value is not None:
            return self._clipboard_param_value
        if len(self._clipboard_params_dict) == 1:
            return next(iter(self._clipboard_params_dict.values()))
        return self._clipboard_params_dict if self._clipboard_params_dict else None

    def is_selected(self, index: int) -> bool:
        """Return True if the given photo index is part of the multi-selection."""
        return index in self._selected_indices

    def set_selected(self, index: int, selected: bool) -> None:
        """Set or unset the selection state for a specific photo index."""
        if not (0 <= index < len(self._items)):
            return
        if selected and index not in self._selected_indices:
            self._selected_indices.add(index)
            self.selection_changed.emit()
        elif not selected and index in self._selected_indices:
            self._selected_indices.remove(index)
            self.selection_changed.emit()

    def toggle_selection(self, index: int) -> bool:
        """Toggle the selection state for a photo index and return the new state."""
        if not (0 <= index < len(self._items)):
            return False
        if index in self._selected_indices:
            self._selected_indices.remove(index)
            res = False
        else:
            self._selected_indices.add(index)
            res = True
        self.selection_changed.emit()
        return res

    def select_all(self) -> None:
        """Select all photos in the gallery."""
        if not self._items:
            return
        self._selected_indices = set(range(len(self._items)))
        self.selection_changed.emit()

    def deselect_all(self) -> None:
        """Clear all photo selections."""
        if self._selected_indices:
            self._selected_indices.clear()
            self.selection_changed.emit()

    def select_range(self, from_index: int, to_index: int, extend: bool = False) -> None:
        """Select a contiguous range of photos between from_index and to_index."""
        if not self._items:
            return
        start = max(0, min(from_index, to_index))
        end = min(len(self._items) - 1, max(from_index, to_index))
        if not extend:
            self._selected_indices.clear()
        for i in range(start, end + 1):
            self._selected_indices.add(i)
        self.selection_changed.emit()

    def get_target_indices(self, fallback_to_current: bool = True) -> list[int]:
        """Return the list of target indices: multi-selected photos if any, or active photo."""
        if self._selected_indices:
            return sorted(self._selected_indices)
        if fallback_to_current and 0 <= self._current_index < len(self._items):
            return [self._current_index]
        return []

    def clear(self) -> None:
        self._items.clear()
        self._current_index = -1
        self._selected_indices.clear()
        self.items_changed.emit()
        self.current_changed.emit(-1, None)
        self.selection_changed.emit()

    def load_folder(self, folder: str | Path, recursive: bool = False) -> int:
        """Scan a directory for all supported RAW and RGB images, naturally sorted."""
        root = Path(folder)
        if not root.is_dir():
            return 0

        found_paths: list[Path] = []
        seen: set[str] = set()

        file_iter = root.rglob("*") if recursive else root.iterdir()
        for p in file_iter:
            if not p.is_file():
                continue
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                canon = str(p.resolve()).lower()
                if canon not in seen:
                    seen.add(canon)
                    found_paths.append(p)

        # If both a .dng and another raw format (e.g. .nef) exist for the same photo, prefer .dng
        dng_stems = {
            (p.parent, p.stem.lower())
            for p in found_paths
            if p.suffix.lower() == ".dng"
        }
        filtered_paths = [
            p for p in found_paths
            if p.suffix.lower() == ".dng" or (p.parent, p.stem.lower()) not in dng_stems or p.suffix.lower() not in RAW_EXTENSIONS
        ]

        # Sort naturally by filename
        filtered_paths.sort(key=lambda p: (p.name.lower(), str(p)))

        self._items = [GalleryItem(path=p) for p in filtered_paths]
        self._selected_indices.clear()
        if self._items:
            self._current_index = 0
        else:
            self._current_index = -1

        self.items_changed.emit()
        self.current_changed.emit(self._current_index, self.current_item)
        self.selection_changed.emit()
        return len(self._items)

    def add_files(self, paths: Sequence[str | Path]) -> int:
        """Add specific image files to the gallery."""
        added = 0
        existing_paths = {str(item.path.resolve()).lower() for item in self._items}

        for path in paths:
            p = Path(path)
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            canon = str(p.resolve()).lower()
            if canon not in existing_paths:
                existing_paths.add(canon)
                self._items.append(GalleryItem(path=p))
                added += 1

        if added > 0:
            # Deduplicate if both .dng and another raw format exist with the same stem
            dng_stems = {
                (item.path.parent, item.path.stem.lower())
                for item in self._items
                if item.path.suffix.lower() == ".dng"
            }
            if dng_stems:
                self._items = [
                    item for item in self._items
                    if item.path.suffix.lower() == ".dng"
                    or (item.path.parent, item.path.stem.lower()) not in dng_stems
                    or item.path.suffix.lower() not in RAW_EXTENSIONS
                ]

            # Preserve current path across sort
            curr_path = self.current_item.path if self.current_item else None
            # Sort full list naturally
            self._items.sort(key=lambda it: (it.path.name.lower(), str(it.path)))
            if curr_path is not None:
                for idx, it in enumerate(self._items):
                    if it.path == curr_path:
                        self._current_index = idx
                        break
            elif self._current_index == -1 and self._items:
                self._current_index = 0
            self.items_changed.emit()
            self.current_changed.emit(self._current_index, self.current_item)

        return added

    def replace_item_path(self, old_path: Path | str, new_path: Path | str) -> bool:
        """Replace an item's path (e.g. when converted to DNG) preserving its index."""
        old_res = Path(old_path).resolve()
        new_p = Path(new_path)
        new_res = new_p.resolve()

        found_idx = -1
        existing_new_idx = -1
        for idx, item in enumerate(self._items):
            item_res = item.path.resolve()
            if item_res == old_res:
                found_idx = idx
            elif item_res == new_res:
                existing_new_idx = idx

        if found_idx == -1 and existing_new_idx == -1:
            return False

        if found_idx != -1 and existing_new_idx != -1:
            self.remove_item(found_idx)
            return True

        if found_idx != -1:
            self._items[found_idx] = GalleryItem(path=new_p)
            self.item_status_updated.emit(found_idx)
            return True

        return False

    def select_index(self, index: int) -> bool:
        """Select a photo by index."""
        if index < 0 or index >= len(self._items):
            return False
        if index == self._current_index:
            return True

        self._current_index = index
        self.current_changed.emit(self._current_index, self.current_item)
        return True

    def select_path(self, path: str | Path) -> bool:
        """Select a photo by file path."""
        target = Path(path).resolve()
        for idx, item in enumerate(self._items):
            if item.path.resolve() == target:
                return self.select_index(idx)
        return False

    def next_photo(self) -> bool:
        """Navigate to the next photo in the gallery."""
        if self._current_index + 1 < len(self._items):
            return self.select_index(self._current_index + 1)
        return False

    def previous_photo(self) -> bool:
        """Navigate to the previous photo in the gallery."""
        if self._current_index - 1 >= 0:
            return self.select_index(self._current_index - 1)
        return False

    def remove_items(self, indices: Sequence[int]) -> list[Path]:
        """Remove multiple photos by index from the gallery queue without deleting files on disk."""
        targets = sorted({i for i in indices if 0 <= i < len(self._items)})
        if not targets:
            return []

        curr_path = self.current_item.path if self.current_item is not None else None
        curr_was_removed = self._current_index in targets

        # Determine fallback photo if current photo is being removed
        remaining_candidates = [i for i in range(len(self._items)) if i not in targets]
        if curr_was_removed:
            if remaining_candidates:
                after = [i for i in remaining_candidates if i >= self._current_index]
                if after:
                    fallback_item = self._items[after[0]]
                else:
                    fallback_item = self._items[remaining_candidates[-1]]
                fallback_path = fallback_item.path
            else:
                fallback_path = None
        else:
            fallback_path = curr_path

        removed_paths: list[Path] = []
        for idx in reversed(targets):
            item = self._items.pop(idx)
            removed_paths.append(item.path)

        removed_paths.reverse()

        # Update selected indices: remove deleted, shift surviving indices
        new_selected = set()
        for s in self._selected_indices:
            if s not in targets:
                shift = sum(1 for t in targets if t < s)
                new_selected.add(s - shift)
        self._selected_indices = new_selected

        # Update current index
        if not self._items:
            self._current_index = -1
        elif fallback_path is not None:
            new_idx = next((i for i, it in enumerate(self._items) if it.path == fallback_path), 0)
            self._current_index = new_idx
        else:
            self._current_index = min(max(0, self._current_index), len(self._items) - 1)

        self.items_changed.emit()
        self.selection_changed.emit()

        if curr_was_removed or not self._items:
            self.current_changed.emit(self._current_index, self.current_item)

        logger.info("Removed %d photo(s) from gallery queue (disk files preserved).", len(removed_paths))
        return removed_paths

    def remove_item(self, index: int) -> bool:
        """Remove a photo from the gallery queue without deleting the file on disk."""
        return bool(self.remove_items([index]))

    def remove_current_photo(self) -> bool:
        """Remove the currently active photo from the gallery queue."""
        return self.remove_item(self._current_index)

    def remove_path(self, path: str | Path) -> bool:
        """Remove a photo specified by path from the gallery queue."""
        target = Path(path).resolve()
        for idx, item in enumerate(self._items):
            if item.path.resolve() == target:
                return self.remove_item(idx)
        return False

    def notify_item_saved(self, path_or_index: Path | str | int) -> None:
        """Notify the model that a photo's sidecar was updated so UI badges refresh."""
        if isinstance(path_or_index, int):
            idx = path_or_index
        else:
            target = Path(path_or_index).resolve()
            idx = next((i for i, item in enumerate(self._items) if item.path.resolve() == target), -1)

        if 0 <= idx < len(self._items):
            self.item_status_updated.emit(idx)

    def copy_look(self, state: GuiState) -> None:
        """Copy a photo's complete GuiState to the gallery clipboard."""
        self._clipboard_state = clone_gui_state(state)
        logger.info("Copied look to gallery clipboard.")

    def paste_look(self) -> GuiState | None:
        """Return a copy of the clipboard GuiState if available."""
        if self._clipboard_state is None:
            return None
        return clone_gui_state(self._clipboard_state)

    def copy_params_dict(self, params_dict: dict[str, Any]) -> None:
        """Copy a dictionary of {param_id: value} to the gallery parameter clipboard."""
        self._clipboard_params_dict = dict(params_dict)
        if len(params_dict) == 1:
            self._clipboard_param_id, self._clipboard_param_value = next(iter(params_dict.items()))
        elif len(params_dict) > 1:
            self._clipboard_param_id = "__multi__"
            self._clipboard_param_value = self._clipboard_params_dict
        else:
            self._clipboard_param_id = None
            self._clipboard_param_value = None
        logger.info("Copied %d parameter(s) to gallery parameter clipboard.", len(params_dict))

    def get_params_clipboard_dict(self) -> dict[str, Any]:
        """Return a copy of the dictionary of parameters stored in the clipboard."""
        if self._clipboard_params_dict:
            return dict(self._clipboard_params_dict)
        if self._clipboard_param_id is not None and self._clipboard_param_id != "__multi__":
            return {self._clipboard_param_id: self._clipboard_param_value}
        return {}

    def copy_single_param(self, param_id: str, value: Any) -> None:
        """Copy a single parameter value to the gallery parameter clipboard."""
        self.copy_params_dict({param_id: value})

    def get_single_param_clipboard(self) -> tuple[str, Any] | None:
        """Return (param_id, value) from the single parameter clipboard if set."""
        if self._clipboard_param_id is not None and self._clipboard_param_id != "__multi__":
            return (self._clipboard_param_id, self._clipboard_param_value)
        if len(self._clipboard_params_dict) == 1:
            return next(iter(self._clipboard_params_dict.items()))
        return None

    def clear_param_clipboard(self) -> None:
        """Clear the parameter clipboard."""
        self._clipboard_param_id = None
        self._clipboard_param_value = None
        self._clipboard_params_dict.clear()
