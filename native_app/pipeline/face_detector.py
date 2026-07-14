"""
Standalone RetinaFace detector — dùng det_10g.onnx trực tiếp qua onnxruntime.
Không dùng insightface package (tránh dependency chain phức tạp).
Logic post-processing ported từ insightface/model_zoo/retinaface.py.
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nms(bboxes: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w    = np.maximum(0.0, xx2 - xx1 + 1)
        h    = np.maximum(0.0, yy2 - yy1 + 1)
        iou  = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(iou <= thresh)[0] + 1]
    return np.array(keep, dtype=np.int32)


def _dist2bbox(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    return np.stack([
        anchors[:, 0] - deltas[:, 0],
        anchors[:, 1] - deltas[:, 1],
        anchors[:, 0] + deltas[:, 2],
        anchors[:, 1] + deltas[:, 3],
    ], axis=-1)


def _dist2kps(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    kps = []
    for i in range(0, deltas.shape[1], 2):
        kps.append(anchors[:, 0] + deltas[:, i])
        kps.append(anchors[:, 1] + deltas[:, i + 1])
    return np.stack(kps, axis=-1)           # (N, 10)


# ── Detector ──────────────────────────────────────────────────────────────────

class InsightFaceDetector:
    """
    RetinaFace face detector wrapping det_10g.onnx (InsightFace buffalo_l).
    Đúng pipeline gốc của wifakey_252 — không phụ thuộc insightface package.
    """
    _STRIDES      = [8, 16, 32]
    _NUM_ANCHORS  = 2
    _NMS_THRESH   = 0.4
    _DET_SIZE     = (640, 640)       # (W, H)

    def __init__(self, exports_dir: Path, confidence: float = 0.5):
        model_path = exports_dir / "det_10g.onnx"
        self._session   = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._in_name   = self._session.get_inputs()[0].name
        self._out_names = [o.name for o in self._session.get_outputs()]
        self._confidence = confidence
        self._use_kps   = len(self._out_names) >= 9
        self._anchors   = {}          # cache per stride

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(self, img_bgr: np.ndarray) -> dict | None:
        ih, iw = img_bgr.shape[:2]
        dw, dh = self._DET_SIZE

        # Scale-preserve resize + zero-pad (top-left)
        scale   = min(dw / iw, dh / ih)
        nw, nh  = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.zeros((dh, dw, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized

        # Preprocess: (img - 127.5) / 128  →  BCHW float32
        blob = (canvas.astype(np.float32) - 127.5) / 128.0
        blob = np.expand_dims(blob.transpose(2, 0, 1), 0)

        outputs = self._session.run(self._out_names, {self._in_name: blob})

        fmc            = 3            # score / bbox / kps each have fmc tensors
        all_scores     = []
        all_bboxes     = []
        all_kpss       = []

        for i, stride in enumerate(self._STRIDES):
            scores  = outputs[i].flatten()
            bboxes  = outputs[i + fmc].reshape(-1, 4) * stride
            anchors = self._get_anchors(dh, dw, stride)

            pos = np.where(scores >= self._confidence)[0]
            if len(pos) == 0:
                continue

            all_scores.append(scores[pos])
            all_bboxes.append(_dist2bbox(anchors, bboxes)[pos])

            if self._use_kps:
                kps_raw = outputs[i + fmc * 2].reshape(-1, 10) * stride
                kpss    = _dist2kps(anchors, kps_raw).reshape(-1, 5, 2)
                all_kpss.append(kpss[pos])

        if not all_scores:
            return None

        scores = np.concatenate(all_scores)
        bboxes = np.concatenate(all_bboxes)
        keep   = _nms(bboxes, scores, self._NMS_THRESH)

        if len(keep) == 0:
            return None

        scores = scores[keep]
        bboxes = bboxes[keep] / scale       # scale back to original image coords

        # Largest face by area
        areas     = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best      = int(np.argmax(areas))
        x1, y1, x2, y2 = bboxes[best].astype(int)

        if not self._use_kps or not all_kpss:
            return None

        kpss = np.concatenate(all_kpss)[keep] / scale
        kps  = kpss[best].astype(np.float32)   # (5, 2)

        return {
            "bbox":       (x1, y1, x2 - x1, y2 - y1),
            "keypoints":  kps,
            "confidence": float(scores[best]),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_anchors(self, det_h: int, det_w: int, stride: int) -> np.ndarray:
        key = (det_h, det_w, stride)
        if key not in self._anchors:
            fh, fw = det_h // stride, det_w // stride
            gx, gy = np.meshgrid(np.arange(fw), np.arange(fh))
            centers = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            # Repeat for _NUM_ANCHORS anchors per cell (interleaved)
            centers = np.repeat(centers, self._NUM_ANCHORS, axis=0)
            self._anchors[key] = centers
        return self._anchors[key]
