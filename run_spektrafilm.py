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
    # Add nvidia DLLs directory if present
    nvidia_bin = os.path.join(bundle_dir, "nvidia")
    if os.path.isdir(nvidia_bin):
        os.environ["PATH"] = nvidia_bin + os.pathsep + os.environ.get("PATH", "")
else:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from spektrafilm_gui.app import main

if __name__ == "__main__":
    main()
