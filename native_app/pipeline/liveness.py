import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class LivenessChecker:
    def __init__(self, model_path: Path, threshold: float = 0.8, bbox_expand: float = 1.5):
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._size = int(inp.shape[2])  # square input (H == W)

        p = float(np.clip(threshold, 1e-6, 1 - 1e-6))
        self._logit_threshold = float(np.log(p / (1.0 - p)))
        self._expand = bbox_expand

    def is_live(self, img_bgr: np.ndarray, bbox_xywh: tuple) -> tuple[bool, float]:
        """Returns (is_live, score). score = real_logit - spoof_logit."""
        crop = self._crop(img_bgr, bbox_xywh)
        tensor = self._preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        score = float(logits[0]) - float(logits[1])
        return score >= self._logit_threshold, score

    def _crop(self, img: np.ndarray, bbox_xywh: tuple) -> np.ndarray:
        x, y, w, h = bbox_xywh
        size = int(max(w, h) * self._expand)
        cx, cy = x + w / 2, y + h / 2
        x0 = int(cx - size / 2)
        y0 = int(cy - size / 2)
        ih, iw = img.shape[:2]
        x1, y1 = min(iw, x0 + size), min(ih, y0 + size)
        x0c, y0c = max(0, x0), max(0, y0)
        if x1 <= x0c or y1 <= y0c:
            return np.zeros((size, size, 3), dtype=img.dtype)
        cropped = img[y0c:y1, x0c:x1]
        pad = (max(0, -y0), max(0, y0 + size - ih),
               max(0, -x0), max(0, x0 + size - iw))
        return cv2.copyMakeBorder(cropped, *pad, cv2.BORDER_REFLECT_101)

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        s = self._size
        oh, ow = rgb.shape[:2]
        r = s / max(oh, ow)
        nh, nw = int(oh * r), int(ow * r)
        interp = cv2.INTER_LANCZOS4 if r > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
        dh, dw = s - nh, s - nw
        padded = cv2.copyMakeBorder(
            resized, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
            cv2.BORDER_REFLECT_101,
        )
        return np.expand_dims(padded.transpose(2, 0, 1).astype(np.float32) / 255.0, 0)
