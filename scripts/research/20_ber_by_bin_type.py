"""
17_ber_by_bin_type.py

Margin trung bình bin trong (interior, bin1/bin2) nhỏ hơn bin ngoài
(exterior, bin0/bin3) khoảng 3.5 lần (đã đo ở script 16). Câu hỏi cần
trả lời: chênh lệch MARGIN này có thực sự dịch chuyển thành chênh lệch
BER (tỷ lệ lật bit thật giữa 2 lần chụp CÙNG người) đáng kể hay không?

Nếu BER(interior) >> BER(exterior) rõ rệt -> đáng đầu tư redesign ngưỡng
(margin-equalizing quantization) vì đây là nguyên nhân THẬT gây lỗi.
Nếu BER hai nhóm gần nhau -> decoder LDPC vốn đã khá "chịu đựng" được
chênh lệch reliability này, không cần đụng vào quantizer - nên tập trung
vào calibration LLR (Mức 2, symbol-level) thay vì redesign ngưỡng.

Cách chạy:
    python scripts/research/17_ber_by_bin_type.py
"""

import os
import sys
import csv
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)
PAIRS_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
)


def load_embedding(name, imagenum):
    return np.load(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def load_genuine_pairs(tier="tune", n=None):
    path = os.path.join(PAIRS_DIR, f"{tier}_genuine.csv")
    pairs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e1 = load_embedding(row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(row["name_verify"], row["imagenum_verify"])
            pairs.append((e1, e2))
    if n is not None:
        pairs = pairs[:n]
    return pairs


def lkut_thermometer(index, n_thr):
    """Tái hiện đúng bảng lkut trong lssc_binary gốc."""
    vec = np.zeros(n_thr, dtype=np.uint8)
    if index >= 1:
        vec[n_thr - index :] = 1
    return vec


def main():
    handler = WiFaKeyHandler()
    intervals = np.sort(np.asarray(handler.intervals).flatten())
    n_thr = len(intervals)

    pairs = load_genuine_pairs(tier="tune", n=300)
    print(f"Dùng {len(pairs)} cặp genuine (tập tune) để đo BER theo loại bin...\n")

    # interior bin = bin có CẢ 2 phía bị chặn bởi ngưỡng hữu hạn (bin 1..n_thr-1
    #                khi có n_thr+1 bin đánh số 0..n_thr)
    # exterior bin = bin 0 hoặc bin n_thr (mở về vô cực 1 phía)
    n_flip_interior, n_total_interior = 0, 0
    n_flip_exterior, n_total_exterior = 0, 0

    for emb_enroll, emb_verify in pairs:
        proj_e = np.dot(emb_enroll, handler.M_matrix)
        proj_v = np.dot(emb_verify, handler.M_matrix)

        bin_e = np.searchsorted(intervals, proj_e, side="left")  # 0..n_thr

        for d in range(len(proj_e)):
            idx_e = bin_e[d]
            is_interior = (idx_e != 0) and (idx_e != n_thr)

            bits_e = lkut_thermometer(idx_e, n_thr)
            idx_v = int(np.searchsorted(intervals, proj_v[d], side="left"))
            bits_v = lkut_thermometer(idx_v, n_thr)

            n_diff = int(np.sum(bits_e != bits_v))

            if is_interior:
                n_flip_interior += n_diff
                n_total_interior += n_thr
            else:
                n_flip_exterior += n_diff
                n_total_exterior += n_thr

    ber_interior = n_flip_interior / n_total_interior
    ber_exterior = n_flip_exterior / n_total_exterior

    print("=== BER (tỷ lệ lật bit thật giữa enroll/verify cùng người) ===")
    print(
        f"  Interior bin (margin hẹp ~0.0075): BER = {ber_interior:.4%}  "
        f"({n_flip_interior}/{n_total_interior} bit)"
    )
    print(
        f"  Exterior bin (margin rộng ~0.0265): BER = {ber_exterior:.4%}  "
        f"({n_flip_exterior}/{n_total_exterior} bit)"
    )
    print(
        f"  Tỷ lệ BER interior/exterior: {ber_interior / max(ber_exterior, 1e-9):.2f}x"
    )

    print("\n=== Kết luận ===")
    ratio = ber_interior / max(ber_exterior, 1e-9)
    if ratio > 2.0:
        print(f"❌ BER interior cao hơn exterior rõ rệt ({ratio:.1f}x) - margin hẹp")
        print("   THỰC SỰ gây lỗi nhiều hơn đáng kể. Đáng đầu tư redesign ngưỡng")
        print("   (margin-equalizing quantization) như đề xuất - đây là nguyên")
        print("   nhân gốc rễ, không chỉ là vấn đề calibration LLR.")
    elif ratio > 1.3:
        print(f"⚠️  BER interior cao hơn exterior vừa phải ({ratio:.1f}x) - có ảnh")
        print("   hưởng nhưng không cực đoan. Có thể thử calibration LLR theo")
        print("   loại bin (Mức 2, symbol-level) trước - rẻ hơn nhiều - trước khi")
        print("   quyết định redesign ngưỡng.")
    else:
        print(f"✅ BER 2 nhóm gần nhau ({ratio:.1f}x) - chênh lệch margin 3.5x KHÔNG")
        print("   dịch chuyển thành chênh lệch lỗi thật đáng kể. KHÔNG cần redesign")
        print("   quantizer - nên tập trung vào hướng khác (vd: calibration LLR")
        print("   Mức 2, hoặc xem lại κ/M_matrix).")


if __name__ == "__main__":
    main()
