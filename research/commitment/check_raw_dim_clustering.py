"""
check_raw_dim_clustering.py

Kiểm tra giả thuyết: pool chọn theo intra_flip per-bit có bị dồn cụm vào ít
raw dimension hơn hẳn so với chọn ngẫu nhiên đều, do mỗi raw dimension (sau
M_matrix, trước LSSC) sinh ra `block` bit LIÊN TIẾP và tương quan mạnh với
nhau (thermometer code) — nếu đúng, đây là nguyên nhân GMR của pool-selection
tệ hơn uniform (burst error thay vì lỗi rời rạc).

Cách chạy:
    python research/commitment/check_raw_dim_clustering.py \\
        --wifakey-data-dir wifakey_module/data \\
        --reliability-scores research/commitment/reliability_tune.npy \\
        --pool-size 900
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wifakey-data-dir", required=True)
    ap.add_argument("--reliability-scores", required=True)
    ap.add_argument("--pool-size", type=int, required=True)
    args = ap.parse_args()

    intervals = np.load(
        os.path.join(args.wifakey_data_dir, "binarization_intervals.npy")
    )
    block = int(np.asarray(intervals).size)
    F = np.load(args.reliability_scores)
    full_len = F.shape[0]
    n_raw_dims = full_len // block

    print(f"full_binary_length={full_len}, block={block}, n_raw_dims={n_raw_dims}")

    pool = np.argsort(-F)[: args.pool_size]
    raw_dims_in_pool = pool // block
    unique_raw_dims = np.unique(raw_dims_in_pool)

    print(f"\nPool theo reliability (M={args.pool_size}):")
    print(
        f"  Số raw dimension DUY NHẤT được đại diện: {len(unique_raw_dims)} / {n_raw_dims}"
    )
    print(
        f"  -> Trung bình {args.pool_size/len(unique_raw_dims):.2f} bit/raw-dimension trong pool"
    )

    # So sánh với chọn ngẫu nhiên đều (kỳ vọng lý thuyết)
    rng = np.random.default_rng(0)
    trials = 20
    unique_counts = []
    for _ in range(trials):
        random_pool = rng.choice(full_len, size=args.pool_size, replace=False)
        unique_counts.append(len(np.unique(random_pool // block)))
    print(f"\nChọn ngẫu nhiên đều (trung bình {trials} lần):")
    print(f"  Số raw dimension duy nhất: {np.mean(unique_counts):.1f} / {n_raw_dims}")
    print(
        f"  -> Trung bình {args.pool_size/np.mean(unique_counts):.2f} bit/raw-dimension"
    )

    ratio = len(unique_raw_dims) / np.mean(unique_counts)
    print(f"\n=> Tỷ lệ độ đa dạng raw-dimension (reliability / random): {ratio:.2f}")
    if ratio < 0.85:
        print(
            "   CẢNH BÁO: pool reliability dồn cụm rõ rệt vào ít raw dimension hơn hẳn"
        )
        print("   so với random — xác nhận giả thuyết burst-correlation.")
    else:
        print("   Không thấy dồn cụm rõ rệt — cần tìm nguyên nhân khác.")


if __name__ == "__main__":
    main()
