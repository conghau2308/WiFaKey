import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QApplication,
)

from native_app.pipeline import AuthPipeline

_BG      = "#1a1a2e"
_SURFACE = "#16213e"
_ACCENT  = "#4a9eff"
_TEXT    = "#e0e0e0"
_MUTED   = "#888888"
_SUCCESS = "#4caf50"
_ERROR   = "#f44336"

# Log ra file — hữu ích khi console=False trong PyInstaller
_log_dir = Path(os.environ.get("LOCALAPPDATA", ".")) / "WiFaKeyAuth"
_log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(_log_dir / "auth.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("wifakey")


# ─────────────────────────────────────────────────────────────────────────────
# Camera worker — camera loop và pipeline chạy ở hai thread khác nhau
# ─────────────────────────────────────────────────────────────────────────────

class _CameraWorker(QThread):
    frame_ready   = Signal(object, object)   # (frame_bgr ndarray, face_info|None)
    status_update = Signal(str, str)         # (text, colour_hex)
    auth_done     = Signal(dict)

    _COOLDOWN          = 2.0    # giây giữa các lần thử pipeline
    _STABILITY_FRAMES  = 4      # số frame liên tiếp cần có quality face trước khi trigger
    _MIN_FACE_RATIO    = 0.05   # diện tích face / frame tối thiểu (5%)
    _EDGE_MARGIN       = 0.15   # face center phải cách mép ít nhất 15% chiều khung

    def __init__(self, pipeline: AuthPipeline, mode: str, request_data: dict):
        super().__init__()
        self._pipeline     = pipeline
        self._mode         = mode
        self._request_data = request_data
        self._active       = True
        self._processing   = False   # pipeline thread đang chạy

    def stop(self):
        self._active = False

    # ── Camera loop (QThread) ─────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.error("Camera unavailable")
            self.auth_done.emit({"ok": False, "code": "camera_unavailable",
                                 "message": "Không mở được camera"})
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        log.debug("Camera opened, starting loop")

        last_attempt  = 0.0
        stable_count  = 0      # số frame liên tiếp có quality face

        try:
            while self._active:
                ret, frame = cap.read()
                if not ret:
                    QThread.msleep(30)
                    continue

                frame = cv2.flip(frame, 1)
                face  = self._pipeline.detector.detect(frame)
                self.frame_ready.emit(frame, face)

                now       = time.time()
                quality   = self._face_quality_ok(frame, face)

                if quality:
                    stable_count += 1
                else:
                    stable_count = 0

                if quality and not self._processing and (now - last_attempt) > self._COOLDOWN:
                    if stable_count >= self._STABILITY_FRAMES:
                        # Khuôn mặt ổn định, quality tốt → trigger pipeline
                        last_attempt  = now
                        stable_count  = 0
                        self._processing = True
                        self.status_update.emit("Đang xác minh…", _ACCENT)
                        log.debug("Triggering pipeline (mode=%s, stable=%d)", self._mode, stable_count)
                        t = threading.Thread(
                            target=self._pipeline_thread,
                            args=(frame.copy(), face),
                            daemon=True,
                        )
                        t.start()
                    else:
                        remaining = self._STABILITY_FRAMES - stable_count
                        self.status_update.emit(f"Giữ yên mặt… ({remaining})", _TEXT)
                elif not self._processing:
                    if face and not quality:
                        x, y, w, h = face["bbox"]
                        fh2, fw2 = frame.shape[:2]
                        edge = int(min(fw2, fh2) * 0.02)
                        if x < edge or y < edge or (x + w) > (fw2 - edge) or (y + h) > (fh2 - edge):
                            self.status_update.emit("Lùi lại để khuôn mặt vào đầy khung…", _MUTED)
                        else:
                            self.status_update.emit("Di chuyển khuôn mặt vào giữa…", _MUTED)
                    elif not face:
                        self.status_update.emit("Nhìn thẳng vào camera…", _MUTED)

                QThread.msleep(33)   # ~30 fps
        finally:
            cap.release()
            log.debug("Camera released")

    # ── Face quality gate ────────────────────────────────────────────────────

    def _face_quality_ok(self, frame: np.ndarray, face: dict | None) -> bool:
        """
        Trả về True chỉ khi face đủ lớn, nằm gần giữa, và toàn bộ
        bounding box nằm trong khung hình (khuôn mặt không bị cắt).
        """
        if face is None:
            return False
        x, y, w, h = face["bbox"]
        fh, fw = frame.shape[:2]
        if fw == 0 or fh == 0:
            return False

        # 1. Face phải chiếm ít nhất MIN_FACE_RATIO diện tích frame
        if (w * h) / (fw * fh) < self._MIN_FACE_RATIO:
            return False

        # 2. Tâm face không được gần mép (tránh crop méo)
        cx, cy = x + w / 2, y + h / 2
        m = self._EDGE_MARGIN
        if not (m * fw < cx < (1 - m) * fw and m * fh < cy < (1 - m) * fh):
            return False

        # 3. Toàn bộ bounding box phải nằm trong khung hình
        # Dùng margin nhỏ 2% để đảm bảo không bị cắt sát mép
        edge = int(min(fw, fh) * 0.02)
        if x < edge or y < edge or (x + w) > (fw - edge) or (y + h) > (fh - edge):
            return False

        return True

    # ── Pipeline thread (daemon Python thread) ────────────────────────────────

    def _pipeline_thread(self, frame: np.ndarray, face: dict):
        try:
            result = self._run_pipeline(frame, face)
        except Exception as exc:
            log.exception("Unhandled exception in pipeline")
            result = {"ok": False, "code": "processing_error", "message": str(exc)}

        log.debug("Pipeline result: %s", result)
        self._processing = False

        if result["ok"] or result.get("code") == "spoof_detected":
            self._active = False
            self.auth_done.emit(result)
        else:
            # Lỗi không nghiêm trọng → hiện thông báo, cho phép thử lại
            msg = result.get("message", "Thử lại…")
            self.status_update.emit(f"⚠ {msg[:60]}", _ERROR)

    # ── Actual pipeline logic ─────────────────────────────────────────────────

    def _run_pipeline(self, frame: np.ndarray, face: dict) -> dict:
        live, score = self._pipeline.liveness.is_live(frame, face["bbox"])
        log.debug("Liveness score=%.3f threshold=%.3f live=%s", score, self._pipeline.liveness._logit_threshold, live)
        if not live:
            return {"ok": False, "code": "spoof_detected",
                    "message": f"Liveness score: {score:.3f}"}

        if self._mode == "enroll":
            return self._pipeline.run_enroll(frame, face)
        else:
            return self._pipeline.run_verify(
                frame, face,
                self._request_data["helper_data_b64"],
                self._request_data["mask_b64"],
            )


# ─────────────────────────────────────────────────────────────────────────────
# Auth window
# ─────────────────────────────────────────────────────────────────────────────

class AuthWindow(QWidget):
    result_ready = Signal(dict)

    _W, _H   = 400, 510
    _CAM_W, _CAM_H = 356, 267

    def __init__(self, pipeline: AuthPipeline, mode: str, request_data: dict):
        super().__init__()
        self._pipeline     = pipeline
        self._mode         = mode
        self._request_data = request_data
        self._worker: _CameraWorker | None = None
        self._dot_tick = 0

        self._build_ui()
        self._center()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._tick_dots)
        self._dot_timer.start(500)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        container = QWidget(objectName="container")
        container.setFixedSize(self._W, self._H)
        container.setStyleSheet(f"""
            QWidget#container {{
                background-color: {_BG};
                border-radius: 12px;
                border: 1px solid #2a2a4a;
            }}
        """)
        root.addWidget(container)

        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(22, 18, 22, 18)
        vbox.setSpacing(12)

        # Header
        header = QHBoxLayout()
        icon_lbl = QLabel("🔒")
        icon_lbl.setStyleSheet("font-size:14px;")
        title = QLabel("WiFaKey Auth")
        title.setStyleSheet(f"color:{_TEXT}; font-size:15px; font-weight:600;")
        close_btn = QPushButton("✕", objectName="close_btn")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(f"""
            QPushButton#close_btn {{ background:transparent; color:{_MUTED}; border:none; font-size:14px; }}
            QPushButton#close_btn:hover {{ color:{_TEXT}; }}
        """)
        close_btn.clicked.connect(self._cancel)
        header.addWidget(icon_lbl)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(close_btn)
        vbox.addLayout(header)

        action_txt = "Đăng ký khuôn mặt" if self._mode == "enroll" else "Xác thực khuôn mặt"
        subtitle = QLabel(action_txt)
        subtitle.setStyleSheet(f"color:{_MUTED}; font-size:12px;")
        vbox.addWidget(subtitle)

        # Camera preview
        self._cam_lbl = QLabel(objectName="cam_lbl")
        self._cam_lbl.setFixedSize(self._CAM_W, self._CAM_H)
        self._cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_lbl.setStyleSheet(f"""
            QLabel#cam_lbl {{
                background:{_SURFACE}; border-radius:8px;
                border:1px solid #2a2a4a; color:{_MUTED}; font-size:12px;
            }}
        """)
        self._cam_lbl.setText("Đang khởi động camera…")
        vbox.addWidget(self._cam_lbl, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Status
        self._status_lbl = QLabel("Nhìn thẳng vào camera…")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet(f"color:{_TEXT}; font-size:13px;")
        vbox.addWidget(self._status_lbl)

        # Dots
        self._dots_lbl = QLabel("● ● ●")
        self._dots_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dots_lbl.setStyleSheet(f"color:{_ACCENT}; font-size:10px; letter-spacing:4px;")
        vbox.addWidget(self._dots_lbl)

        vbox.addStretch()

        cancel_btn = QPushButton("Hủy")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{ background:transparent; color:{_MUTED}; border:1px solid #2a2a4a; border-radius:6px; font-size:13px; }}
            QPushButton:hover {{ background:#1e1e3a; color:{_TEXT}; }}
        """)
        cancel_btn.clicked.connect(self._cancel)
        vbox.addWidget(cancel_btn)

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self._W) // 2, (screen.height() - self._H) // 2)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def show(self):
        super().show()
        self._start_worker()

    def _start_worker(self):
        self._worker = _CameraWorker(self._pipeline, self._mode, self._request_data)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status_update.connect(self._on_status)
        self._worker.auth_done.connect(self._on_auth_done)
        self._worker.start()

    def _cancel(self):
        self._shutdown_worker()
        self._emit_and_close({"ok": False, "code": "cancelled"})

    def _shutdown_worker(self):
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

    # ── Slots ────────────────────────────────────────────────────────────────

    @Slot(object, object)
    def _on_frame(self, frame_bgr: np.ndarray, face_info):
        display = frame_bgr.copy()
        if face_info:
            x, y, w, h = face_info["bbox"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (74, 158, 255), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._CAM_W, self._CAM_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._cam_lbl.setPixmap(pix)

    @Slot(str, str)
    def _on_status(self, text: str, colour: str):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"color:{colour}; font-size:13px;")

    @Slot(dict)
    def _on_auth_done(self, result: dict):
        self._shutdown_worker()
        self._dot_timer.stop()
        if result.get("ok"):
            ok_msg = "✓  Lấy mẫu thành công, vui lòng chờ…"
            self._status_lbl.setText(ok_msg)
            self._status_lbl.setStyleSheet(f"color:{_SUCCESS}; font-size:14px; font-weight:600;")
            self._dots_lbl.hide()
        else:
            code = result.get("code", "")
            msgs = {
                "spoof_detected":    "Phát hiện khuôn mặt giả.",
                "camera_unavailable":"Không mở được camera.",
                "cancelled":         "Đã hủy.",
            }
            self._status_lbl.setText(msgs.get(code, result.get("message", "Thất bại.")))
            self._status_lbl.setStyleSheet(f"color:{_ERROR}; font-size:13px;")
            self._dots_lbl.hide()
        QTimer.singleShot(1800, lambda: self._emit_and_close(result))

    def _emit_and_close(self, result: dict):
        self.result_ready.emit(result)
        self.close()
        self.deleteLater()

    # ── Dots animation ───────────────────────────────────────────────────────

    def _tick_dots(self):
        self._dot_tick = (self._dot_tick + 1) % 4
        self._dots_lbl.setText(["●", "● ●", "● ● ●", "● ●"][self._dot_tick])

    # ── Draggable / paint ────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, "_drag_pos"):
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def closeEvent(self, event):
        self._shutdown_worker()
        super().closeEvent(event)

    def paintEvent(self, event):
        pass
