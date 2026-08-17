"""
diagnose_ber_per_bit.py

Chẩn đoán bản đồ BER thô: đo tần suất lật bit cho từng vị trí trong 1536 bit
(phân rã theo dimension và threshold) trên tập genuine tune của LFW,
sử dụng SecureWiFaKeyHandler. Lưu kết quả ra file CSV để phân tích.
"""

import os
import sys
import csv
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler

# --- Cấu hình ---
DATASET = "labeled_faces_in_the_wild"
TIER = "tune"
CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET, "embeddings_cache"
)
PAIRS_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET, "pairs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "ber_per_bit_analysis.csv")


# --- Loaders ---
def load_embedding(name, imagenum):
    return np.load(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def load_genuine_pairs(max_pairs=None):
    path = os.path.join(PAIRS_DIR, f"{TIER}_genuine.csv")
    pairs = []
    with open(path, newline="") as f:
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
    print("Khởi tạo SecureWiFaKeyHandler...")
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    pairs = load_genuine_pairs()  # 881 cặp
    print(f"Phân tích {len(pairs)} cặp genuine...")

    # Mảng đếm lỗi: full 1536 bit
    error_counts = np.zeros(1536, dtype=np.int32)
    total_counts = np.zeros(1536, dtype=np.int32)

    # Phân rã theo dimension (512) và threshold (3)
    error_by_dim_thr = np.zeros((512, 3), dtype=np.int32)
    total_by_dim_thr = np.zeros((512, 3), dtype=np.int32)

    for idx, (name_e, img_e, name_v, img_v) in enumerate(pairs):
        emb_enroll = load_embedding(name_e, img_e)
        emb_verify = load_embedding(name_v, img_v)

        # Lấy b_full 1536 bit của cả enroll và verify
        b_enroll = handler._binarize_full(emb_enroll).astype(np.uint8)
        b_verify = handler._binarize_full(emb_verify).astype(np.uint8)

        errors = b_enroll != b_verify
        error_counts += errors
        total_counts += 1

        # Cập nhật theo dimension và threshold
        for dim in range(512):
            for thr in range(3):
                pos = dim * 3 + thr
                if errors[pos]:
                    error_by_dim_thr[dim, thr] += 1
                total_by_dim_thr[dim, thr] += 1

        if (idx + 1) % 200 == 0:
            print(f"  Đã xử lý {idx+1} cặp...")

    # Tính BER
    ber_full = error_counts / total_counts

    # Lưu CSV chi tiết
    print(f"Lưu kết quả vào {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["bit_position", "dimension", "threshold", "error_count", "total", "BER"]
        )
        for pos in range(1536):
            dim = pos // 3
            thr = pos % 3
            writer.writerow(
                [pos, dim, thr, error_counts[pos], total_counts[pos], ber_full[pos]]
            )

    # Phân tích tổng quan
    print("\n=== KẾT QUẢ PHÂN TÍCH ===")
    print(f"Tổng số bit: 1536, tổng số cặp: {len(pairs)}")
    print(f"BER trung bình toàn bộ 1536 bit: {np.mean(ber_full):.4f}")

    # Top 10% vị trí lỗi nhiều nhất
    top10pct = int(1536 * 0.1)
    top_indices = np.argsort(-error_counts)[:top10pct]
    total_errors = np.sum(error_counts)
    errors_in_top = np.sum(error_counts[top_indices])
    print(
        f"Top 10% vị trí ({top10pct} bit) có BER trung bình: {np.mean(ber_full[top_indices]):.4f}, "
        f"chiếm {errors_in_top/total_errors*100:.1f}% tổng lỗi"
    )

    # BER theo dimension (trung bình 3 bit)
    ber_by_dim = np.sum(error_by_dim_thr, axis=1) / np.sum(total_by_dim_thr, axis=1)
    top_dims = np.argsort(-ber_by_dim)[:20]
    print("\n20 dimension có BER cao nhất:")
    for d in top_dims:
        print(
            f"  Dim {d}: BER={ber_by_dim[d]:.4f} (lỗi: {np.sum(error_by_dim_thr[d])}/{np.sum(total_by_dim_thr[d])})"
        )

    # Phân bố theo ngưỡng
    print("\nPhân bố lỗi theo ngưỡng (0,1,2):")
    for thr in range(3):
        total_thr = np.sum(total_by_dim_thr[:, thr])
        err_thr = np.sum(error_by_dim_thr[:, thr])
        print(f"  Threshold {thr}: BER={err_thr/total_thr:.4f} ({err_thr}/{total_thr})")

    # Nhận xét định hướng
    print("\n--- NHẬN XÉT ---")
    if errors_in_top / total_errors > 0.3:
        print("Lỗi tập trung đáng kể (>30% lỗi nằm trong top 10% vị trí).")
        print("=> Hướng (1) lượng tử hóa thích nghi từng chiều rất hứa hẹn.")
    else:
        print("Lỗi phân bố khá đều.")
        print("=> Nên ưu tiên hướng (2) Test-Time Augmentation hoặc (3) Fusion.")

    handler.sess.close()


if __name__ == "__main__":
    main()
