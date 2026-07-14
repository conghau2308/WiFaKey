# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec — WiFaKey Native Authenticator
# Build:  pyinstaller WiFaKeyAuth.spec --noconfirm
# Output: dist\WiFaKeyAuth\WiFaKeyAuth.exe
#
from PyInstaller.utils.hooks import collect_all, collect_data_files

# ── Collect C-extension packages ────────────────────────────────────────────
ort_d, ort_b, ort_h = collect_all("onnxruntime")
cv2_d, cv2_b, cv2_h = collect_all("cv2")
ski_d, ski_b, ski_h = collect_all("skimage")
# ── Model & data files ───────────────────────────────────────────────────────
datas = [
    # ONNX models → exports/
    ("exports/det_10g.onnx",           "exports"),   # RetinaFace buffalo_l detector
    ("exports/anti-spoofing.onnx",     "exports"),
    ("exports/adaface_ir101.onnx",     "exports"),
    # NOTE: neural_ms_decoder.onnx is intentionally NOT bundled — LDPC decoding
    # now happens server-side (Authentication_Service) so the trained decoder
    # never ships inside the distributed app.
    # WiFaKey data arrays → wifakey_data/
    ("wifakey_module/data/M_matrix.npy",               "wifakey_data"),
    ("wifakey_module/data/binarization_intervals.npy",  "wifakey_data"),
    ("wifakey_module/data/BaseGraph",                  "wifakey_data/BaseGraph"),
    ("wifakey_module/data/BaseGraph_GM",               "wifakey_data/BaseGraph_GM"),
] + ort_d + cv2_d + ski_d

binaries = ort_b + cv2_b + ski_b

# ── Hidden imports ───────────────────────────────────────────────────────────
hiddenimports = [
    "native_app",
    "native_app.pipeline",
    "native_app.pipeline.face_detector",
    "native_app.pipeline.face_aligner",
    "native_app.pipeline.liveness",
    "native_app.pipeline.embedding",
    "native_app.pipeline.wifakey",
    "native_app.ui.auth_window",
    "native_app.server.websocket_server",
    # websockets internals
    "websockets",
    "websockets.legacy",
    "websockets.legacy.server",
    "websockets.legacy.client",
    "websockets.server",
    "websockets.asyncio",
    "websockets.asyncio.server",
    # PySide6
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtNetwork",
    # skimage
    "skimage.transform",
    "skimage._shared",
    "skimage._shared.geometry",
    "skimage.transform._geometric",
    "skimage.transform._warps",
] + ort_h + cv2_h + ski_h

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    ["native_app/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tensorflow", "torch", "transformers", "insightface",
        "matplotlib", "galois", "sympy",
        "tkinter", "wx", "PyQt5", "PyQt6",
        "jupyter", "IPython", "notebook",
        "PIL",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ── Single-file exe — everything packed inside one WiFaKeyAuth.exe ────────────
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="WiFaKeyAuth",
    debug=False,
    strip=False,
    upx=False,      # disable UPX — avoids false-positive AV detections
    console=False,  # no console window (tray app)
    # icon="WiFaKeyAuth.ico",  # uncomment if you have an .ico file
)
