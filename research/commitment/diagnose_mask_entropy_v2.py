"""
diagnose_mask_entropy_v2.py

Phân tích entropy của reliability mask:
  - Đo entropy từng vị trí và entropy tổng của mask.
  - Kiểm tra tính độc lập giữa các vị trí (pairwise correlation).
  - So sánh với mask ngẫu nhiên.
"""

import os
import sys
import csv
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536
FEATURE_LENGTH = 832


# --- Loaders ---
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
    # Khởi tạo handler để dùng M_matrix và intervals
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    pairs = load_genuine_pairs()  # 881 cặp
    print(f"Số cặp genuine: {len(pairs)}")

    # Thu thập mask cho từng người dùng (dùng ảnh enroll)
    # Mỗi người có thể xuất hiện nhiều lần, ta chỉ lấy mask từ ảnh enroll đầu tiên gặp
    user_masks = {}  # identity -> mask (1536,)
    for name_e, img_e, name_v, img_v in pairs:
        if name_e not in user_masks:
            emb = load_embedding(name_e, img_e)
            b_full = handler._binarize_full(emb).astype(np.uint8)
            projected = np.dot(emb, handler.M_matrix)
            _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
            selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[
                :FEATURE_LENGTH
            ]
            mask = np.zeros(FULL_BITS, dtype=np.uint8)
            mask[selection_indices] = 1
            user_masks[name_e] = mask

    masks = np.array(list(user_masks.values()))  # (N_users, 1536)
    n_users = len(masks)
    print(f"Số người dùng duy nhất: {n_users}")

    # --- 1. Entropy từng vị trí và entropy tổng giả định độc lập ---
    p_one = masks.mean(axis=0)  # xác suất được chọn cho từng vị trí
    p_one = np.clip(p_one, 1e-12, 1 - 1e-12)  # tránh log(0)
    entropy_per_pos = -(p_one * np.log2(p_one) + (1 - p_one) * np.log2(1 - p_one))
    total_entropy_independent = np.sum(entropy_per_pos)

    # --- 2. Kiểm tra tính độc lập: tính tương quan pairwise ---
    # Để tránh tính ma trận 1536x1536 quá lớn, ta lấy mẫu ngẫu nhiên 100 vị trí
    sample_positions = np.random.choice(
        FULL_BITS, size=min(100, FULL_BITS), replace=False
    )
    corr_matrix = np.corrcoef(masks[:, sample_positions].T)
    # Lấy các giá trị tương quan ngoài đường chéo
    off_diag = corr_matrix[np.triu_indices(len(sample_positions), k=1)]
    mean_abs_corr = np.mean(np.abs(off_diag))
    max_abs_corr = np.max(np.abs(off_diag))

    # --- 3. So sánh với mask ngẫu nhiên ---
    # Mask ngẫu nhiên: mỗi vị trí có xác suất được chọn = 832/1536 ≈ 0.5417
    p_random = FEATURE_LENGTH / FULL_BITS
    entropy_per_pos_random = -(
        p_random * np.log2(p_random) + (1 - p_random) * np.log2(1 - p_random)
    )
    total_entropy_random = FULL_BITS * entropy_per_pos_random

    # --- 4. In kết quả ---
    print("\n=== PHÂN TÍCH ENTROPY RELIABILITY MASK ===")
    print(
        f"Entropy trung bình mỗi vị trí: {np.mean(entropy_per_pos):.4f} bit (max=1.0)"
    )
    print(f"Entropy tổng mask (giả định độc lập): {total_entropy_independent:.2f} bit")
    print(f"Entropy mask ngẫu nhiên (độc lập): {total_entropy_random:.2f} bit")
    print(
        f"Tỷ lệ entropy so với ngẫu nhiên: {total_entropy_independent/total_entropy_random*100:.1f}%"
    )

    print(f"\nTương quan pairwise trung bình (|r|): {mean_abs_corr:.6f}")
    print(f"Tương quan pairwise lớn nhất (|r|): {max_abs_corr:.6f}")

    # Nhận xét
    print("\n=== NHẬN XÉT ===")
    if mean_abs_corr < 0.01 and max_abs_corr < 0.05:
        print(
            "Các vị trí mask GẦN NHƯ ĐỘC LẬP. Entropy tổng ước tính từ giả định độc lập là CHÍNH XÁC."
        )
    else:
        print(
            "CẢNH BÁO: Có tương quan giữa các vị trí mask. Entropy thực tế THẤP HƠN ước tính độc lập."
        )
        print("         Cần phân tích thêm để định lượng chính xác rò rỉ.")

    loss_percent = 100 * (1 - total_entropy_independent / total_entropy_random)
    if loss_percent > 10:
        print(
            f"\n=> CẢNH BÁO: Mask làm giảm {loss_percent:.1f}% entropy so với random."
        )
        print("   Điều này có thể làm yếu bảo mật nếu không được xử lý.")
    else:
        print(
            f"\n=> Mask chỉ làm giảm {loss_percent:.1f}% entropy, vẫn trong ngưỡng an toàn."
        )

    handler.sess.close()


if __name__ == "__main__":
    main()
