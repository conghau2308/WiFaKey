"""
diagnose_mask_entropy.py

Đo phân phối bit tại các vị trí được chọn bởi margin selection,
ước tính entropy và so sánh với random selection.
"""

import os, sys, csv, numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536


class MarginSelectionHandler(SecureWiFaKeyHandler):
    def get_margin_mask(self, feature_vector_float):
        """Trả về mask 1536 bit (1 tại vị trí được chọn, 0 nếu không)."""
        projected = np.dot(feature_vector_float, self.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, self.intervals)
        indices = np.argpartition(-margin, self.feature_length)[: self.feature_length]
        mask = np.zeros(FULL_BITS, dtype=np.uint8)
        mask[indices] = 1
        return mask


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


def load_unique_identities():
    """Trả về list các (name, imagenum) duy nhất từ tune_genuine.csv (chỉ lấy 1 ảnh/người)."""
    pairs_csv = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "pairs",
        "tune_genuine.csv",
    )
    identities = {}
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name_enroll"]
            if name not in identities:
                identities[name] = (name, int(row["imagenum_enroll"]))
    return list(identities.values())


def main():
    # Khởi tạo handler
    data_dir = os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    handler = MarginSelectionHandler(data_path=data_dir)

    # Lấy danh sách các ảnh duy nhất
    identities = load_unique_identities()
    print(f"Số người dùng duy nhất: {len(identities)}")

    # Thống kê
    selection_counts = np.zeros(FULL_BITS, dtype=np.int32)  # số lần được chọn
    bit_sum = np.zeros(
        FULL_BITS, dtype=np.float64
    )  # tổng giá trị bit (để tính xác suất bit=1)

    for name, imagenum in identities:
        emb = load_embedding(name, imagenum)
        mask = handler.get_margin_mask(emb)
        selection_counts += mask

        # Lấy bit thực tế tại các vị trí được chọn
        b_full = handler._binarize_full(emb).astype(np.uint8)
        bit_sum += b_full * mask  # chỉ cộng bit ở vị trí mask=1

    # Xác suất được chọn
    prob_selected = selection_counts / len(identities)

    # Xác suất bit=1 tại mỗi vị trí (chỉ tính trên những lần được chọn)
    prob_1 = np.divide(
        bit_sum,
        selection_counts,
        out=np.full_like(bit_sum, 0.5),
        where=selection_counts > 0,
    )

    # Entropy mỗi bit (chỉ xét vị trí được chọn ít nhất 1 lần)
    valid = selection_counts > 0
    p = prob_1[valid]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    entropy_per_bit = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    avg_entropy = np.mean(entropy_per_bit)

    # Entropy của random selection (lý tưởng)
    random_entropy = 1.0  # mỗi bit có p=0.5 -> entropy=1

    print(f"\n=== KẾT QUẢ PHÂN TÍCH ENTROPY MASK ===")
    print(f"Số người dùng: {len(identities)}")
    print(f"Số vị trí được chọn ít nhất 1 lần: {np.sum(valid)}/{FULL_BITS}")
    print(f"Xác suất được chọn trung bình: {np.mean(prob_selected):.4f}")
    print(
        f"Xác suất bit=1 trung bình (tại vị trí được chọn): {np.mean(prob_1[valid]):.4f}"
    )
    print(f"Entropy trung bình mỗi bit (tại vị trí được chọn): {avg_entropy:.4f} bit")
    print(f"Entropy lý tưởng (random, p=0.5): {random_entropy:.4f} bit")

    # So sánh với random
    if avg_entropy > 0.99:
        print("\n=> Mask gần như không làm giảm entropy. An toàn.")
    elif avg_entropy > 0.95:
        print("\n=> Mask làm giảm nhẹ entropy (<5%). Vẫn chấp nhận được.")
    else:
        print(
            f"\n=> CẢNH BÁO: Mask làm giảm đáng kể entropy ({(1-avg_entropy)*100:.1f}%). Cần phân tích thêm!"
        )

    # Phân phối selection count
    print(f"\nPhân phối số lần được chọn:")
    print(f"  Min: {selection_counts.min()}, Max: {selection_counts.max()}")
    print(
        f"  Mean: {selection_counts.mean():.1f}, Median: {np.median(selection_counts):.1f}"
    )

    # Tỷ lệ vị trí "luôn được chọn" hoặc "không bao giờ được chọn"
    always = np.sum(selection_counts == len(identities))
    never = np.sum(selection_counts == 0)
    print(f"  Số vị trí LUÔN được chọn: {always}")
    print(f"  Số vị trí KHÔNG BAO GIỜ được chọn: {never}")

    handler.sess.close()


if __name__ == "__main__":
    main()
