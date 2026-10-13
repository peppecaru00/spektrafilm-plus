from __future__ import annotations

from functools import lru_cache

from qtpy import QtGui

from spektrafilm_gui.theme_palette import ACCENT_COLOR_TEXT

try:
    import pyconify
except ImportError:  # pragma: no cover - exercised in runtime fallback only
    pyconify = None


HEADER_ICON_SIZE = 16

_SECTION_HEADER_ICONS = {
    'import rgb': 'tabler:photo-plus',
    'import raw': 'tabler:photo-cog',
    'input': 'tabler:arrow-big-down-lines',
    'camera': 'tabler:camera',
    'profiles': 'tabler:toilet-paper',
    'exposure control': 'tabler:exposure',
    'enlarger': 'tabler:building-lighthouse',
    'scanner': 'tabler:scan',
    'preview and crop': 'tabler:crop',
    'crop and upscale': 'tabler:crop',
    'output': 'tabler:arrow-big-down-lines',
    'grain': 'tabler:grain',
    'halation': 'tabler:time-duration-0',
    'couplers': 'tabler:chart-sankey',
    'chemistry': 'tabler:flask',
    'glare': 'tabler:background',
    'preflash': 'tabler:sparkles',
    'diffusion': 'tabler:artboard',
    'spectral upsampling': 'tabler:prism-light',
    'input gamut compress': 'tabler:triangle-off',
    'output gamut compress': 'tabler:triangles',
    'tune': 'tabler:stroke-curved',
    'experimental': 'tabler:flask',
    'gui parameters': 'tabler:adjustments-horizontal',
    'compute device': 'tabler:cpu',
    'display': 'tabler:skew-x',
    'napari layers': 'tabler:stack-2',
}


def section_header_icon_name(title: str) -> str | None:
    return _SECTION_HEADER_ICONS.get(title.strip().lower())


@lru_cache(maxsize=None)
def section_header_icon(title: str, size: int = HEADER_ICON_SIZE) -> QtGui.QIcon:
    icon_name = section_header_icon_name(title)
    if icon_name is None or pyconify is None:
        return QtGui.QIcon()

    try:
        path = pyconify.svg_path(icon_name, color=ACCENT_COLOR_TEXT, width=size, height=size)
    except (OSError, TypeError, ValueError):
        return QtGui.QIcon()

    return QtGui.QIcon(str(path))


@lru_cache(maxsize=1)
def get_app_icon() -> QtGui.QIcon:
    """Return the application icon from packaged assets."""
    import sys
    from pathlib import Path

    candidates: list[Path] = [
        Path(__file__).resolve().parent / "assets" / "spektrafilm.ico",
        Path(__file__).resolve().parent / "assets" / "spektrafilm.png",
    ]
    if getattr(sys, "frozen", False):
        bundle_dir = Path(getattr(sys, "_MEIPASS", sys.executable)).resolve()
        candidates.extend([
            bundle_dir / "spektrafilm.ico",
            bundle_dir / "spektrafilm_gui" / "assets" / "spektrafilm.ico",
            bundle_dir / "spektrafilm_gui" / "assets" / "spektrafilm.png",
        ])
    for cand in candidates:
        if cand.is_file():
            icon = QtGui.QIcon(str(cand))
            if not icon.isNull():
                return icon
    return QtGui.QIcon()