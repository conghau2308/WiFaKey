"""
v1_adaptive_quantization.py

C.3 — Adaptive bit-allocation. Thay bước binarize gốc (1 bộ ngưỡng CHUNG
cho cả 512 chiều) bằng ngưỡng RIÊNG cho từng chiều (per_dim_intervals.npy,
sinh bởi compute_per_dim_intervals.py).

CÔ LẬP ĐÚNG 1 BIẾN: kế thừa trực tiếp SecureWiFaKeyHandler (v1) và CHỈ
override _binarize_full() -- phần selection (uniform random, an toàn) và
verify() giữ NGUYÊN 100% logic v1. Nhờ vậy, nếu GMR thay đổi, chắc chắn là
do bước lượng tử hoá, không lẫn với cơ chế chọn vị trí (đã học từ bài học
C.1 -- luôn cô lập biến trước khi kết luận).
"""

import os

import numpy as np

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler


def _adaptive_lssc_binary(
    projected_vector: np.ndarray, per_dim_intervals: np.ndarray
) -> np.ndarray:
    """Tái tạo logic lssc_binary gốc (thermometer code), nhưng dùng ngưỡng
    RIÊNG cho từng chiều thay vì 1 mảng interval chung.

    projected_vector : (n_dims,) -- giá trị đã qua M_matrix của 1 mẫu
    per_dim_intervals: (n_dims, n_thr) -- ngưỡng riêng cho từng chiều

    Trả về vector nhị phân dài n_dims * n_thr, đúng thứ tự block liên tiếp
    như lssc_binary gốc (giữ tương thích với full_binary_length hiện có).
    """
    n_dims, n_thr = per_dim_intervals.shape
    lkut = np.zeros((n_thr + 1, n_thr), dtype=np.uint8)
    for i in range(1, n_thr + 1):
        lkut[i, n_thr - i :] = 1

    out = np.zeros(n_dims * n_thr, dtype=np.uint8)
    for j in range(n_dims):
        val = projected_vector[j]
        thresholds = per_dim_intervals[j]
        where_greater = np.where(thresholds > val)[0]
        index = where_greater[0] if len(where_greater) != 0 else -1
        out[j * n_thr : (j + 1) * n_thr] = lkut[index]
    return out


class AdaptiveQuantizationWiFaKeyHandler(SecureWiFaKeyHandler):
    def __init__(self, *args, per_dim_intervals_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_dim_intervals = np.load(per_dim_intervals_path)

        n_dims, n_thr = self.per_dim_intervals.shape
        expected_full_len = n_dims * n_thr
        if expected_full_len != self.full_binary_length:
            raise ValueError(
                f"per_dim_intervals cho ra full_binary_length={expected_full_len}, "
                f"khác với self.full_binary_length={self.full_binary_length} hiện tại "
                f"(tính từ M_matrix/binarization_intervals gốc). Kiểm tra lại "
                f"--n-thr lúc chạy compute_per_dim_intervals.py có khớp n_thr gốc không."
            )

    def _binarize_full(self, feature_vector_float: np.ndarray) -> np.ndarray:
        """Override: dùng ngưỡng riêng theo chiều thay vì self.intervals chung."""
        projected = np.dot(feature_vector_float, self.M_matrix)
        return _adaptive_lssc_binary(projected, self.per_dim_intervals)
