import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller 6+ onedir: data files are in _internal/ (sys._MEIPASS)
    _BASE = Path(sys._MEIPASS)            # type: ignore[attr-defined]
    ROOT = Path(sys.executable).parent
    EXPORTS_DIR = _BASE / "exports"
    DATA_DIR    = _BASE / "wifakey_data"
else:
    ROOT = Path(__file__).parent.parent      # WiFaKey_252/
    EXPORTS_DIR = ROOT / "exports"
    DATA_DIR    = ROOT / "wifakey_module" / "data"
