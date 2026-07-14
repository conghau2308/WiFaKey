import numpy as np
import onnxruntime as ort
from pathlib import Path


class AdaFaceONNX:
    def __init__(self, model_path: Path):
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def get_embedding(self, aligned_rgb_112: np.ndarray) -> np.ndarray:
        """Input: 112x112 uint8 RGB. Output: 512-dim L2-normalized float32."""
        if aligned_rgb_112.shape != (112, 112, 3):
            raise ValueError(f"Expected (112,112,3), got {aligned_rgb_112.shape}")

        # Normalize to [-1, 1]
        x = aligned_rgb_112.astype(np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        x = np.expand_dims(x.transpose(2, 0, 1), 0)  # (1, 3, 112, 112)

        feat = self._session.run(None, {self._input_name: x})[0].squeeze()

        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat
