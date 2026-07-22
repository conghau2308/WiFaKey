"""
v1_lssc_with_perbit_confidence

QUAN TRỌNG — LÝ DO CẦN FILE NÀY THAY VÌ DÙNG v0_lssc_with_confidence:

`v0_lssc_with_confidence.binarize_with_confidence` trả về CONFIDENCE DÙNG
CHUNG cho cả block n_thr bit của một chiều embedding: confidence = khoảng
cách tới ngưỡng GẦN NHẤT trong số tất cả ngưỡng. Đây là một proxy hợp lý
cho "độ tin cậy tổng quát" nhưng KHÔNG khớp với margin đã dùng để fit bảng
hiệu chỉnh thực nghiệm (experiments/out_step3/reliability_lookup.npz,
xem 18_reliability_calibration_curve.py / 19_empirical_llr_fit.py).

Bảng đó được fit trên MARGIN RIÊNG CHO TỪNG BIT: mỗi bit trong block ứng
với ĐÚNG MỘT ngưỡng cụ thể (do cấu trúc thermometer-code đảo ngược của
lssc_binary gốc), không phải khoảng cách tới ngưỡng gần nhất. Nếu dùng
confidence của v0 (dùng chung cho cả block) tra vào bảng lookup fit trên
margin per-bit, 2/3 số bit trong mỗi block sẽ bị gán SAI LLR — vì confidence
của v0 phản ánh đúng cho bit "gần ngưỡng gần nhất nhất", còn 2 bit còn lại
trong cùng block có margin THẬT khác hẳn (có thể lớn hơn/nhỏ hơn nhiều) so
với giá trị confidence dùng chung đó.

File này sinh margin ĐÚNG PER-BIT, suy trực tiếp từ cấu trúc lkut gốc
(đã xác nhận bit-for-bit qua self-check ở bước 2):
    bit_j (block-local, dim-major, j = 0..n_thr-1)
        = 1  iff  v >= thr_sorted[n_thr-1-j]
    margin_j = |v - thr_sorted[n_thr-1-j]|

BẮT BUỘC: chạy `_selftest_against_original` trước khi dùng cho bất kỳ
experiment nào, để đảm bảo bit sinh ra khớp bit-for-bit với `lssc_binary`
gốc (nếu không khớp, margin per-bit cũng vô nghĩa).
"""

import numpy as np


def binarize_with_perbit_confidence(projected: np.ndarray, intervals: np.ndarray):
    """
    Input:
        projected : (D,) float — embedding đã qua M_matrix
        intervals : (n_thr,) float — ngưỡng binarization (KHÔNG cần sort sẵn,
                    hàm tự sort giống hệt cách lssc_binary gốc dùng)

    Output:
        bits   : (D * n_thr,) uint8   — giống hệt output của lssc_binary gốc
        margin : (D * n_thr,) float32 — margin RIÊNG cho từng bit (không
                 broadcast dùng chung trong block như v0), khớp đúng định
                 nghĩa đã dùng để fit reliability_lookup.npz.
    """
    thr = np.sort(np.asarray(intervals, dtype=np.float64).reshape(-1))
    rev_thr = thr[::-1]
    n_thr = thr.size

    v = np.asarray(projected, dtype=np.float64)
    D = v.shape[0]

    cmp = v[:, None] >= rev_thr[None, :]  # (D, n_thr)
    margin = np.abs(v[:, None] - rev_thr[None, :])  # (D, n_thr)

    bits = cmp.astype(np.uint8).reshape(-1)
    margin = margin.astype(np.float32).reshape(-1)

    assert bits.shape[0] == D * n_thr
    return bits, margin


def _selftest_against_original(lssc_binary_fn, n_features=50, n_thr=3, n_trials=20):
    """
    Đối chiếu bit-for-bit với hàm lssc_binary gốc.

    Gọi trước khi dùng module này cho bất kỳ experiment nào:

        from wifakey_module.wifakey_lib.utils import lssc_binary
        _selftest_against_original(lssc_binary)

    Lưu ý: dùng n_thr=3 mặc định vì đó là số ngưỡng thật của
    binarization_intervals.npy trong repo (đã xác nhận ở bước 2). Nếu repo
    đổi số ngưỡng, truyền n_thr tương ứng.
    """
    rng = np.random.default_rng(0)
    for _ in range(n_trials):
        intervals = np.sort(rng.uniform(-1, 1, size=n_thr))
        projected = rng.uniform(-1.5, 1.5, size=n_features)

        bits_new, margin_new = binarize_with_perbit_confidence(projected, intervals)
        bits_orig = (
            lssc_binary_fn(projected.reshape(1, -1), interval=intervals)
            .flatten()
            .astype(np.uint8)
        )

        overlap = min(len(bits_new), len(bits_orig))
        if not np.array_equal(bits_new[:overlap], bits_orig[:overlap]):
            raise AssertionError(
                "MISMATCH: binarize_with_perbit_confidence lệch so với lssc_binary gốc!\n"
                f"  bits_new[:12]  ={bits_new[:12].tolist()}\n"
                f"  bits_orig[:12] ={bits_orig[:12].tolist()}"
            )
        # margin phải luôn >= 0 và hữu hạn
        assert np.all(margin_new >= 0) and np.all(np.isfinite(margin_new))

    print(
        f"Self-test PASSED: {n_trials} trials khớp bit-for-bit với bản gốc "
        f"(n_thr={n_thr}), margin per-bit hợp lệ."
    )


if __name__ == "__main__":
    # Chạy self-test độc lập nếu gọi trực tiếp file này (không phụ thuộc
    # cấu trúc project root, chỉ để kiểm tra công thức toán học không phụ
    # thuộc vào bit-for-bit của lssc_binary thật — dùng bản sao logic
    # thermometer để test nhanh).
    def _reference_lssc_binary(embeddings_o, interval):
        n_thr = len(interval)
        lkut = np.zeros((n_thr + 1, n_thr), dtype=np.uint8)
        for i in range(1, n_thr + 1):
            lkut[i, n_thr - i :] = 1
        out = np.zeros((len(embeddings_o), embeddings_o.shape[1] * n_thr))
        for row in range(len(embeddings_o)):
            data = embeddings_o[row]
            for d in range(len(data)):
                where_idx = np.where(interval > data[d])[0]
                index = where_idx[0] if len(where_idx) else -1
                out[row, d * n_thr : (d + 1) * n_thr] = lkut[index]
        return out

    _selftest_against_original(_reference_lssc_binary)
