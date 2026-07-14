"""
v0_lssc_with_confidence

QUAN TRỌNG: đây KHÔNG phải là cải tiến thuật toán binarization.
Đây là một bản re-implementation của `lssc_binary` gốc (wifakey_lib/utils.py),
sinh ra CHÍNH XÁC cùng chuỗi bit như bản gốc, nhưng đồng thời trả về thêm
một mảng "confidence" (khoảng cách từ giá trị liên tục tới ngưỡng gần nhất)
mà pipeline gốc không lưu lại.

Lý do cần file riêng thay vì sửa `utils.py`:
  - `utils.py` là code gốc, không được sửa (theo quy ước tổ chức project).
  - Soft-LLR (v1_soft_distance_llr) cần thông tin "độ tin cậy" per-bit,
    vốn bị bỏ đi trong bản gốc (gốc chỉ trả về 0/1 cuối cùng).

BẮT BUỘC: mỗi khi dùng file này, phải chạy test đối chiếu bit-for-bit với
`lssc_binary` gốc trên cùng input để đảm bảo re-implementation không lệch
(xem hàm `_selftest_against_original` bên dưới).
"""

import numpy as np


def binarize_with_confidence(projected: np.ndarray, intervals: np.ndarray):
    """
    Input:
        projected : (D,) float — embedding đã qua M_matrix
        intervals : (n_thr,) float, đã sort tăng dần — ngưỡng binarization

    Output:
        bits       : (D * n_thr,) uint8   — giống hệt output của lssc_binary gốc
        confidence : (D * n_thr,) float32 — khoảng cách |giá trị - ngưỡng gần nhất|
                     của chiều feature sinh ra block n_thr bit đó (broadcast).
                     Giá trị càng nhỏ = càng gần ranh giới quyết định = càng kém tin cậy.
    """
    n_thr = len(intervals)
    D = len(projected)
    bits = np.zeros(D * n_thr, dtype=np.uint8)
    confidence = np.zeros(D * n_thr, dtype=np.float32)

    # Cùng logic lkut như bản gốc: lkut[i, n_thr-i:] = 1
    lkut = np.zeros((n_thr + 1, n_thr), dtype=np.uint8)
    for i in range(1, n_thr + 1):
        lkut[i, n_thr - i :] = 1

    for d in range(D):
        val = projected[d]
        where_idx = np.where(intervals > val)[0]
        index = where_idx[0] if len(where_idx) != 0 else -1

        bits[d * n_thr : (d + 1) * n_thr] = lkut[index]

        # Khoảng cách tới ngưỡng gần nhất (bất kể ngưỡng nào) quyết định độ tin cậy
        dist_to_thresholds = np.abs(intervals - val)
        conf = float(np.min(dist_to_thresholds))
        confidence[d * n_thr : (d + 1) * n_thr] = conf

    return bits, confidence


def _selftest_against_original(lssc_binary_fn, n_features=50, n_thr=4, n_trials=20):
    """
    Chạy đối chiếu bit-for-bit với hàm lssc_binary gốc.
    Gọi hàm này trong unit test trước khi dùng module này cho bất kỳ experiment nào.

        from wifakey_module.wifakey_lib.utils import lssc_binary
        _selftest_against_original(lssc_binary)
    """
    rng = np.random.default_rng(0)
    for _ in range(n_trials):
        intervals = np.sort(rng.uniform(-1, 1, size=n_thr))
        projected = rng.uniform(-1.5, 1.5, size=n_features)

        bits_new, _ = binarize_with_confidence(projected, intervals)
        bits_orig = (
            lssc_binary_fn(projected.reshape(1, -1), interval=intervals)
            .flatten()
            .astype(np.uint8)
        )

        # Lưu ý: hàm gốc hard-code 512, chỉ so khớp phần chồng lấp thực tế
        overlap = min(len(bits_new), len(bits_orig))
        assert np.array_equal(
            bits_new[:overlap], bits_orig[:overlap]
        ), "MISMATCH: binarize_with_confidence lệch so với lssc_binary gốc!"
    print(f"Self-test PASSED: {n_trials} trials khớp bit-for-bit với bản gốc.")
