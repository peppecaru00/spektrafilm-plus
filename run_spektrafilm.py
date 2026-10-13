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
    if hasattr(os, "add_dll_directory") and os.path.isdir(bundle_dir):
        try:
            os.add_dll_directory(bundle_dir)
        except Exception:
            pass

    # Recursively discover and register all directories containing DLL files
    dll_dirs: set[str] = set()
    for root, dirs, files in os.walk(bundle_dir):
        if any(f.lower().endswith(".dll") for f in files):
            dll_dirs.add(root)

    for d in sorted(dll_dirs):
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

    # Point CUDA_PATH to bundled nvidia runtime directory if present
    cuda_runtime_dir = os.path.join(bundle_dir, "nvidia", "cuda_runtime")
    if os.path.isdir(cuda_runtime_dir):
        os.environ.setdefault("CUDA_PATH", cuda_runtime_dir)
else:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # In development mode on Windows, register any site-packages nvidia DLL directories
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        for sp in list(sys.path):
            nvidia_cand = os.path.join(sp, "nvidia")
            if os.path.isdir(nvidia_cand):
                for root, dirs, files in os.walk(nvidia_cand):
                    if any(f.lower().endswith(".dll") for f in files):
                        try:
                            os.add_dll_directory(root)
                        except Exception:
                            pass


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

