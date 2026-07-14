"""
Entry point: WiFaKeyAuth.exe [--port 7825] [--origin https://…]

Startup:
  1. Single-instance guard (port mutex)
  2. QApplication, tray icon (grey = loading, blue = ready)
  3. Load models in background thread
  4. WebSocket server on localhost:{port}
  5. Tray menu: status / startup toggle / quit
"""
import argparse
import socket
import sys
import winreg
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QAction
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from native_app import EXPORTS_DIR, DATA_DIR
from native_app.pipeline import AuthPipeline
from native_app.server.websocket_server import WiFaKeyWSServer, PendingAuth
from native_app.ui.auth_window import AuthWindow

_APP_NAME   = "WiFaKeyAuth"
_STARTUP_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


# ─────────────────────────────────────────────────────────────────────────────
# Windows startup registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _startup_enabled() -> bool:
    try:
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _STARTUP_KEY, 0, winreg.KEY_READ)
        winreg.QueryValueEx(k, _APP_NAME)
        winreg.CloseKey(k)
        return True
    except OSError:
        return False


def _set_startup(enable: bool) -> None:
    k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _STARTUP_KEY, 0, winreg.KEY_SET_VALUE)
    if enable:
        winreg.SetValueEx(k, _APP_NAME, 0, winreg.REG_SZ, sys.executable)
    else:
        try:
            winreg.DeleteValue(k, _APP_NAME)
        except OSError:
            pass
    winreg.CloseKey(k)


# ─────────────────────────────────────────────────────────────────────────────
# Single-instance guard
# ─────────────────────────────────────────────────────────────────────────────

def _already_running(port: int) -> bool:
    """Returns True if the WS port is already bound (another instance running)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind(("127.0.0.1", port))
        s.close()
        return False
    except OSError:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Tray icon (programmatic lock shape — no external file needed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_icon(colour: str) -> QIcon:
    pm = QPixmap(32, 32)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(colour)
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(8, 15, 16, 13, 2, 2)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(c, 3))
    p.drawArc(9, 6, 14, 14, 0, 180 * 16)
    p.setBrush(QColor("#1a1a2e"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(13, 19, 6, 6)
    p.end()
    return QIcon(pm)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

class _ModelLoader(QThread):
    done   = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.done.emit(AuthPipeline(EXPORTS_DIR, DATA_DIR))
        except Exception as exc:
            self.failed.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Auth launcher — WS thread → Qt main thread bridge
# ─────────────────────────────────────────────────────────────────────────────

class _AuthLauncher(QObject):
    _request_signal = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pipeline: AuthPipeline | None = None
        self._request_signal.connect(self._open_window, Qt.ConnectionType.QueuedConnection)
        # Giữ reference để Python GC không destroy window ngay khi _open_window return
        self._active_windows: list[AuthWindow] = []

    def set_pipeline(self, p: AuthPipeline):
        self._pipeline = p

    def request(self, pending: PendingAuth):
        self._request_signal.emit(pending)

    @Slot(object)
    def _open_window(self, pending: PendingAuth):
        if self._pipeline is None:
            pending.set_result({"ok": False, "code": "not_ready"})
            return

        win = AuthWindow(self._pipeline, pending.mode, pending.data)
        self._active_windows.append(win)

        def _on_result(r: dict):
            pending.set_result(r)
            if win in self._active_windows:
                self._active_windows.remove(win)

        win.result_ready.connect(_on_result)
        win.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WiFaKey Native Authenticator")
    parser.add_argument("--port",   type=int, default=7825)
    parser.add_argument("--origin", action="append", dest="origins", default=[])
    args = parser.parse_args()

    # ── Single instance ──
    if _already_running(args.port):
        sys.exit(0)

    app = QApplication(sys.argv)
    app.setApplicationName(_APP_NAME)
    app.setQuitOnLastWindowClosed(False)

    # ── Tray icon ──
    tray = QSystemTrayIcon(app)
    tray.setIcon(_make_icon("#666666"))
    tray.setToolTip("WiFaKey Auth — Loading…")
    tray.setVisible(True)

    # ── Tray menu ──
    menu = QMenu()

    status_action = QAction("⏳  Loading models…")
    status_action.setEnabled(False)
    menu.addAction(status_action)
    menu.addSeparator()

    startup_action = QAction("Khởi động cùng Windows")
    startup_action.setCheckable(True)
    startup_action.setChecked(_startup_enabled())

    def _toggle_startup(checked: bool):
        _set_startup(checked)

    startup_action.triggered.connect(_toggle_startup)
    menu.addAction(startup_action)
    menu.addSeparator()

    quit_action = QAction("Thoát")
    quit_action.triggered.connect(app.quit)
    menu.addAction(quit_action)

    tray.setContextMenu(menu)

    # ── WebSocket server ──
    ws_server = WiFaKeyWSServer(
        port=args.port,
        allowed_origins=set(args.origins) if args.origins else set(),
    )
    launcher = _AuthLauncher()
    ws_server._on_auth = launcher.request
    ws_server.start_in_thread()

    # ── Load models ──
    loader = _ModelLoader()

    def _on_loaded(pipeline: AuthPipeline):
        launcher.set_pipeline(pipeline)
        ws_server.set_ready(True)
        tray.setIcon(_make_icon("#4a9eff"))
        tray.setToolTip(f"WiFaKey Auth  ·  ws://127.0.0.1:{args.port}")
        status_action.setText("🔵  Ready")
        tray.showMessage("WiFaKey Auth", "Authenticator đã sẵn sàng.",
                         QSystemTrayIcon.MessageIcon.Information, 3000)

    def _on_failed(msg: str):
        tray.setIcon(_make_icon("#f44336"))
        tray.setToolTip(f"WiFaKey Auth — Lỗi")
        status_action.setText(f"❌  {msg[:60]}")
        tray.showMessage("WiFaKey Auth — Lỗi", msg,
                         QSystemTrayIcon.MessageIcon.Critical, 8000)

    loader.done.connect(_on_loaded)
    loader.failed.connect(_on_failed)
    loader.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
