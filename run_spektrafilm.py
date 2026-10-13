"""Launcher script for SpektraFilm standalone application and PyInstaller freezing."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# In frozen mode, configure DLL paths and environment
if getattr(sys, "frozen", False):
    bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    if bundle_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bundle_dir + os.pathsep + os.environ.get("PATH", "")
    # Add DLL subdirectories if present
    for sub in ["nvidia", os.path.join("OpenImageIO", "bin"), "exiv2", "rawpy", "lensfunpy", "shiboken6", "PySide6"]:
        sub_path = os.path.join(bundle_dir, sub)
        if os.path.isdir(sub_path):
            os.environ["PATH"] = sub_path + os.pathsep + os.environ.get("PATH", "")
else:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _launch() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("IanoLiotta.SpektraFilmPlus.0.1.0")
        except Exception:
            pass

    from qtpy import QtWidgets
    # Ensure QApplication exists before any QWidget or plugin discovery
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    app.setApplicationName("SpektraFilm Plus")
    app.setOrganizationName("IanoLiotta")
    app.setApplicationVersion("0.1.0")

    try:
        from spektrafilm_gui.icons import get_app_icon
        app_icon = get_app_icon()
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
    except Exception:
        pass

    from spektrafilm_gui.app import main
    main()



if __name__ == "__main__":
    try:
        _launch()
    except Exception:
        import traceback
        err_msg = traceback.format_exc()
        try:
            log_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
            with open(os.path.join(log_dir, "spektrafilm_crash.log"), "w", encoding="utf-8") as f:
                f.write(err_msg)
        except Exception:
            pass
        if sys.stderr is not None:
            try:
                sys.stderr.write(err_msg)
            except Exception:
                pass
        sys.exit(1)

