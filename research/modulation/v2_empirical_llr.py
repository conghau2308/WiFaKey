"""
v2_empirical_llr — CẢI TIẾN #2: thay công thức tay (scale=60) bằng LLR
HIỆU CHỈNH THỰC NGHIỆM (empirical calibration), fit trực tiếp trên dữ liệu.

Bối cảnh (xem 18_reliability_calibration_curve.py và 19_empirical_llr_fit.py):
    v1_soft_distance_llr (scale=60) phóng đại một tín hiệu độ tin cậy chỉ
    thật sự mạnh ~1.31x (đo thô ban đầu) thành biên độ LLR khổng lồ, gây
    overconfidence và làm decoder thua cả hard-BPSK.

    Bước 2 (đo lại bằng 10 bin mịn hơn) cho ratio thật = 22-23x (không phải
    1.31x) — tín hiệu độ tin cậy CÓ predictive value tốt, chỉ là công thức
    tham số (scale=60) hiệu chỉnh sai biên độ hoàn toàn.

    Bước 3 fit một hàm hiệu chỉnh KHÔNG THAM SỐ (isotonic regression qua
    PAVA) trực tiếp margin -> p(lật bit) trên tune_train, validate trên
    tune_val (generalize tốt: Brier cải thiện 12.6%, sai số hiệu chỉnh TB
    chỉ 0.0033). Module này dùng bảng đó, KHÔNG còn scale/min_mag/max_mag
    tùy chỉnh tay nữa — LLR tự sinh biên độ đúng thực tế (~1-3, well-calibrated).

QUAN TRỌNG — margin đầu vào PHẢI lấy từ
`research/quantizer/v1_lssc_with_perbit_confidence.binarize_with_perbit_confidence`,
KHÔNG phải `v0_lssc_with_confidence` (confidence của v0 dùng chung cho cả
block 3 bit, khác định nghĩa margin per-bit đã dùng để fit bảng lookup —
xem docstring của v1_lssc_with_perbit_confidence.py).
"""

import os
import numpy as np
from research.common.base_modulation import BaseModulation

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# ADAPT nếu bạn di chuyển reliability_lookup.npz sang nơi khác (khuyến nghị
# copy vào wifakey_module/data/ hoặc research/modulation/data/ để không phụ
# thuộc thư mục experiments/ vốn có thể bị coi là output tạm thời).
_DEFAULT_LOOKUP_PATH = os.path.join(
    _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
)


class EmpiricalLLR(BaseModulation):
    name = "v2_empirical_llr"

    def __init__(
        self,
        lookup_path: str = _DEFAULT_LOOKUP_PATH,
        masked_mag: float = 1.5,
    ):
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(
                f"[{self.name}] không tìm thấy bảng lookup tại {lookup_path}. "
                f"Chạy 19_empirical_llr_fit.py trước, hoặc truyền lookup_path đúng."
            )
        data = np.load(lookup_path)
        self._margin_bp = data["margin_breakpoints"]
        self._p_bp = data["p_breakpoints"]
        self._eps = float(data["eps"])
        # Biên độ gán cho bit bị mask (mask_r=0). Giữ NGUYÊN logic đã kiểm
        # chứng ở v1_soft_distance_llr: y_noisy tại các vị trí này = helper_data,
        # biết trước TẤT ĐỊNH (không phụ thuộc embedding) -> margin không có ý
        # nghĩa gì ở đây, và gán độ tin cậy cao (như max_mag) từng gây FAR=39.2%
        # trong thực nghiệm trước. masked_mag=1.5 khớp quy ước hard-BPSK.
        self.masked_mag = masked_mag
        self.lookup_path = lookup_path

    def _margin_to_llr_magnitude(self, margin: np.ndarray) -> np.ndarray:
        p = np.interp(
            margin,
            self._margin_bp,
            self._p_bp,
            left=self._p_bp[0],
            right=self._p_bp[-1],
        )
        p = np.clip(p, self._eps, 0.5 - self._eps)
        return np.log((1.0 - p) / p).astype(np.float32)

    def modulate(
        self, noisy_bits: np.ndarray, context: dict | None = None
    ) -> np.ndarray:
        if context is None or "margin" not in context:
            raise ValueError(
                f"[{self.name}] cần context['margin'] (từ "
                f"research/quantizer/v1_lssc_with_perbit_confidence.py). "
                f"KHÔNG dùng context['distance'] của v0/v1 cũ — định nghĩa "
                f"margin khác nhau (xem docstring module này)."
            )

        margin = context["margin"]
        assert margin.shape == noisy_bits.shape, "margin và noisy_bits phải cùng shape"

        magnitude = self._margin_to_llr_magnitude(margin)

        mask = context.get("mask")
        if mask is not None:
            magnitude = np.where(mask.astype(bool), magnitude, self.masked_mag)

        sign = (
            2 * noisy_bits.astype(np.float32) - 1
        )  # 0 -> -1, 1 -> +1 (giữ quy ước gốc)

        return (sign * magnitude).astype(np.float32)
