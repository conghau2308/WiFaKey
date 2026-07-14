import base64
import time
from pathlib import Path

import cv2
import numpy as np

from .face_detector import InsightFaceDetector
from .face_aligner import align_face
from .liveness import LivenessChecker
from .embedding import AdaFaceONNX
from .wifakey import WiFaKeyONNX


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


class AuthPipeline:
    def __init__(self, exports_dir: Path, data_dir: Path):
        print("[Pipeline] Loading InsightFace buffalo_l (det_10g.onnx)...")
        self.detector = InsightFaceDetector(exports_dir)

        print("[Pipeline] Loading anti-spoof...")
        self.liveness = LivenessChecker(exports_dir / "anti-spoofing.onnx")

        print("[Pipeline] Loading AdaFace (260 MB, may take a moment)...")
        self.embedding = AdaFaceONNX(exports_dir / "adaface_ir101.onnx")

        print("[Pipeline] Loading WiFaKey fuzzy-commitment module...")
        self.wifakey = WiFaKeyONNX(data_dir)

        print("[Pipeline] All models ready.")

    def run_enroll(self, frame_bgr: np.ndarray, face_info: dict) -> dict:
        aligned = align_face(frame_bgr, face_info["keypoints"])
        embedding = self.embedding.get_embedding(aligned)
        helper_data, mask, key_hash = self.wifakey.enroll(embedding)
        return {
            "ok": True,
            "action": "enroll_result",
            "helper_data_b64": _b64(helper_data.tobytes()),
            "mask_b64": _b64(mask.tobytes()),
            "key_hash_b64": _b64(key_hash),
        }

    def run_verify(
        self,
        frame_bgr: np.ndarray,
        face_info: dict,
        helper_data_b64: str,
        mask_b64: str,
    ) -> dict:
        aligned = align_face(frame_bgr, face_info["keypoints"])
        embedding = self.embedding.get_embedding(aligned)

        helper_data = np.frombuffer(_unb64(helper_data_b64), dtype=np.uint8)
        mask = np.frombuffer(_unb64(mask_b64), dtype=np.uint8)
        c_prime = self.wifakey.get_noisy_codeword(embedding, helper_data, mask)

        return {
            "ok": True,
            "action": "verify_result",
            "c_prime_b64": _b64(c_prime.tobytes()),
        }
