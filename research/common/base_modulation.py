"""
Base interface cho bước Modulation (bit -> LLR).

Mọi phiên bản cải tiến (v0, v1, v2, ...) đều PHẢI kế thừa class này và
implement đúng signature của `modulate()`. Điều này đảm bảo:
  - pipeline_factory có thể hoán đổi (swap) version qua config, không sửa code gọi.
  - baseline (v0) và các bản cải tiến luôn tương thích 1-1 để so sánh công bằng.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModulation(ABC):
    """
    Input của mọi version:
        noisy_bits : np.ndarray, dtype uint8/int, shape (N,)
                     bit sau XOR (helper_data XOR b_selected), giá trị {0,1}
        context    : dict, optional - thông tin phụ trợ để tính độ tin cậy
                     (vd: khoảng cách tới ngưỡng binarization, thống kê enroll...)
                     v0 (hard-decision) sẽ bỏ qua context hoàn toàn.

    Output:
        llr : np.ndarray, dtype float32, shape giống noisy_bits
              Giá trị LLR đưa thẳng vào decoder Neural-MS hiện có
              (decoder gốc nhận float, không quan tâm LLR là hard hay soft).
    """

    name: str = "base"

    @abstractmethod
    def modulate(
        self, noisy_bits: np.ndarray, context: dict | None = None
    ) -> np.ndarray: ...

    def __call__(
        self, noisy_bits: np.ndarray, context: dict | None = None
    ) -> np.ndarray:
        out = self.modulate(noisy_bits, context)
        assert (
            out.shape == noisy_bits.shape
        ), f"[{self.name}] LLR output shape {out.shape} != input shape {noisy_bits.shape}"
        assert out.dtype in (
            np.float32,
            np.float64,
        ), f"[{self.name}] LLR output phải là float, nhận {out.dtype}"
        return out.astype(np.float32)
