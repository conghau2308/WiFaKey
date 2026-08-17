"""
diagnose_mask_entropy_v3.py

Chẩn đoán tổng hợp entropy của reliability mask:
- Entropy mask pattern (liên kết danh tính)
- Entropy giá trị bit (lộ khóa)
- Tương quan giữa các vị trí mask
- Đề xuất mức độ an toàn
"""

import os
import sys
import csv
import numpy as np
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536
FEATURE_LENGTH = 832


def load_embedding(name, imagenum):
    path = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "embeddings_cache",
        f"{name}_{int(imagenum):04d}.npy",
    )
    return np.load(path)


def load_genuine_pairs(max_pairs=None):
    pairs_csv = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "pairs",
        "tune_genuine.csv",
    )
    pairs = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                (
                    row["name_enroll"],
                    int(row["imagenum_enroll"]),
                    row["name_verify"],
                    int(row["imagenum_verify"]),
                )
            )
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    pairs = load_genuine_pairs()
    print(f"Số cặp genuine: {len(pairs)}")

    # Thu thập mask và bit cho từng người dùng
    user_data = {}  # identity -> list of (mask, bits)
    for name_e, img_e, name_v, img_v in pairs:
        if name_e not in user_data:
            emb = load_embedding(name_e, img_e)
            b_full = handler._binarize_full(emb).astype(np.uint8)
            projected = np.dot(emb, handler.M_matrix)
            _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
            selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[
                :FEATURE_LENGTH
            ]
            selection_indices.sort()
            mask = np.zeros(FULL_BITS, dtype=np.uint8)
            mask[selection_indices] = 1
            user_data[name_e] = (mask, b_full)

    n_users = len(user_data)
    print(f"Số người dùng duy nhất: {n_users}")

    # 1. Phân tích mask pattern
    masks = np.array([data[0] for data in user_data.values()])
    p_selected = masks.mean(axis=0)
    p_selected = np.clip(p_selected, 1e-12, 1 - 1e-12)
    entropy_pattern = -(
        p_selected * np.log2(p_selected) + (1 - p_selected) * np.log2(1 - p_selected)
    )
    total_entropy_pattern = np.sum(entropy_pattern)
    print(
        f"\n1. Entropy mask pattern: {total_entropy_pattern:.2f} / {FULL_BITS:.2f} bit ({100*total_entropy_pattern/FULL_BITS:.1f}%)"
    )

    # 2. Phân tích giá trị bit tại vị trí được chọn
    bit_counts = np.zeros(FULL_BITS, dtype=int)
    selected_counts = np.zeros(FULL_BITS, dtype=int)
    for mask, bits in user_data.values():
        selected_positions = np.where(mask == 1)[0]
        selected_counts[selected_positions] += 1
        bit_counts[selected_positions] += bits[selected_positions]

    valid = selected_counts > 0
    p_bit1 = np.divide(
        bit_counts, selected_counts, out=np.full(FULL_BITS, 0.5), where=valid
    )
    p_bit1 = np.clip(p_bit1, 1e-12, 1 - 1e-12)
    entropy_bit = -(p_bit1 * np.log2(p_bit1) + (1 - p_bit1) * np.log2(1 - p_bit1))
    avg_entropy_bit = np.mean(entropy_bit[valid])
    print(
        f"2. Entropy trung bình mỗi bit (tại vị trí được chọn): {avg_entropy_bit:.4f} bit"
    )
    print(f"   Rò rỉ so với lý tưởng (1.0): {100*(1-avg_entropy_bit):.1f}%")

    # 3. Phân tích tương quan (đầy đủ hơn)
    print("\n3. Phân tích tương quan...")
    # Lấy mẫu ngẫu nhiên 200 vị trí để tính tương quan
    sample_positions = np.random.choice(
        FULL_BITS, size=min(200, FULL_BITS), replace=False
    )
    corr_matrix = np.corrcoef(masks[:, sample_positions].T)
    off_diag = corr_matrix[np.triu_indices(len(sample_positions), k=1)]
    mean_abs_corr = np.mean(np.abs(off_diag))
    max_abs_corr = np.max(np.abs(off_diag))
    print(f"   Tương quan trung bình (|r|): {mean_abs_corr:.6f}")
    print(f"   Tương quan lớn nhất (|r|): {max_abs_corr:.6f}")

    # 4. Entropy hiệu dụng (ước tính)
    # Nếu các vị trí độc lập hoàn toàn, entropy tổng = tổng entropy từng vị trí
    # Nếu có tương quan, entropy thực tế thấp hơn
    total_entropy_independent = np.sum(entropy_bit[valid])
    # Giảm nhẹ do tương quan (dùng công thức đơn giản)
    effective_entropy = total_entropy_independent * (1 - mean_abs_corr)
    print(f"\n4. Entropy hiệu dụng (ước tính): {effective_entropy:.2f} bit")

    # 5. Đánh giá an toàn
    print("\n5. Đánh giá an toàn:")
    if avg_entropy_bit > 0.95:
        print("   ✓ Entropy bit rất tốt (>0.95). Hệ thống an toàn.")
    elif avg_entropy_bit > 0.85:
        print("   ⚠ Entropy bit khá (0.85-0.95). Cần theo dõi thêm.")
    else:
        print("   ✗ Entropy bit thấp (<0.85). Cần biện pháp khắc phục!")

    if total_entropy_pattern > 0.95 * FULL_BITS:
        print("   ✓ Entropy mask pattern tốt. Chống liên kết danh tính hiệu quả.")
    else:
        print("   ⚠ Entropy mask pattern thấp. Có thể liên kết danh tính!")

    handler.sess.close()


if __name__ == "__main__":
    main()
