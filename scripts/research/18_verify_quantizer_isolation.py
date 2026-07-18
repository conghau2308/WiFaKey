"""
15_verify_quantizer_isolation.py

Xác nhận binarize_with_confidence (dùng bởi v1/soft_llr) cho ra CHÍNH XÁC
cùng 'bits' như baseline_quantizer (dùng lssc_binary gốc trực tiếp, dùng
bởi v0/baseline) - trên TOÀN BỘ embedding thật trong tập select, không
chỉ vài mẫu giả như self-test trước đó.

Nếu bits khớp 100% -> yên tâm rằng khác biệt FRR/FAR giữa exp001/exp002
CHỈ đến từ modulation (v0 vs v1), không bị lẫn ảnh hưởng của quantizer -
đúng nguyên tắc "chỉ đổi 1 biến số" đã thống nhất.

Nếu KHÔNG khớp -> quantizer cũng là 1 biến số đang bị đổi cùng lúc, cần
tách riêng thành 1 exp khác (v0_hard_bpsk + binarize_with_confidence) để
đo ảnh hưởng riêng của quantizer trước khi kết luận về modulation.

Cách chạy:
    python scripts/research/15_verify_quantizer_isolation.py
"""

import os
import sys
import csv
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib.utils import lssc_binary
from research.quantizer.v0_lssc_with_confidence import binarize_with_confidence

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


def collect_unique_embeddings_from_select_tier():
    names = set()
    for fname in ["select_genuine.csv", "select_impostor.csv"]:
        path = os.path.join(PAIRS_DIR, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.add((row["name_enroll"], row["imagenum_enroll"]))
                names.add((row["name_verify"], row["imagenum_verify"]))
    return list(names)


def main():
    handler = WiFaKeyHandler()
    unique_embs = collect_unique_embeddings_from_select_tier()
    print(f"Kiểm tra trên {len(unique_embs)} embedding duy nhất trong tập select...\n")

    n_match, n_mismatch = 0, 0
    max_diff_bits = 0

    for name, imagenum in unique_embs:
        emb = load_embedding(name, imagenum)
        projected = np.dot(emb, handler.M_matrix)

        bits_baseline = (
            lssc_binary(projected.reshape(1, -1), interval=handler.intervals)
            .flatten()
            .astype(np.uint8)
        )
        bits_new, _ = binarize_with_confidence(projected, handler.intervals)

        if np.array_equal(bits_baseline, bits_new):
            n_match += 1
        else:
            n_mismatch += 1
            diff = np.sum(bits_baseline != bits_new)
            max_diff_bits = max(max_diff_bits, diff)

    print(f"Khớp hoàn toàn: {n_match}/{len(unique_embs)}")
    print(f"KHÔNG khớp:     {n_mismatch}/{len(unique_embs)}")
    if n_mismatch > 0:
        print(f"Số bit lệch nhiều nhất trong 1 embedding: {max_diff_bits}")
        print("\n❌ Quantizer KHÔNG tương đương -> exp002 đang đổi CẢ quantizer lẫn")
        print("   modulation cùng lúc. Cần chạy thêm exp001b (v0_hard_bpsk +")
        print("   binarize_with_confidence) để tách ảnh hưởng của quantizer riêng.")
    else:
        print(
            "\n✅ Quantizer tương đương 100% trên toàn bộ dữ liệu thật của tập select."
        )
        print("   -> Yên tâm: khác biệt FRR/FAR giữa exp001/exp002 CHỈ đến từ")
        print("   modulation (v0 vs v1), không bị lẫn quantizer.")


if __name__ == "__main__":
    main()
