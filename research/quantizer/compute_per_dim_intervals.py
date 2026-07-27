"""
compute_per_dim_intervals.py

C.3 — Adaptive bit-allocation: tính ngưỡng lượng tử hoá (LSSC) RIÊNG cho
từng chiều trong 512 chiều "projected" (sau M_matrix), thay vì dùng 1 bộ
ngưỡng DUY NHẤT chung cho cả 512 chiều như hiện tại (binarization_intervals.npy
gốc là 1 mảng (n_thr,) áp dụng như nhau cho mọi dimension).

Ý tưởng: mỗi chiều có phân phối giá trị khác nhau (mean/scale khác nhau,
độ "gần biên" khác nhau) -- dùng chung 1 bộ ngưỡng cho mọi chiều là bỏ qua
sự khác biệt này. Tính ngưỡng equiprobable RIÊNG cho từng chiều (percentile
của chính phân phối chiều đó trên tập calibration) giúp mỗi chiều được chia
bin cân bằng theo đúng phân phối của nó, có thể giảm BER raw ngay từ bước
nhị phân hoá -- TRƯỚC CẢ khi tới bước selection/commitment.

CHỈ dùng dữ liệu population (embeddings từ tune_genuine.csv + tune_impostor.csv,
LẤY HỢP CÁC EMBEDDING DUY NHẤT xuất hiện trong 2 file này) -- không dùng
dữ liệu riêng của cá nhân nào tại thời điểm enroll thật, an toàn theo đúng
nguyên tắc population-level đã thống nhất từ trước.

Output: per_dim_intervals.npy, shape (512, n_thr) -- mỗi hàng là bộ ngưỡng
equiprobable riêng cho 1 dimension.

Cách chạy:
    python research/quantizer/compute_per_dim_intervals.py \\
        --wifakey-data-dir wifakey_module/data \\
        --pairs-dir datasets/processed/labeled_faces_in_the_wild/pairs \\
        --cache-dir datasets/processed/labeled_faces_in_the_wild/embeddings_cache \\
        --n-thr 3 \\
        --output research/quantizer/per_dim_intervals.npy
"""

import argparse
import csv
import os

import numpy as np


def load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def collect_unique_embeddings(pairs_csv: str, cache_dir: str, seen: dict):
    """Đọc 1 file pairs CSV (cột name_enroll/imagenum_enroll/name_verify/
    imagenum_verify), nạp thêm các embedding CHƯA THẤY vào dict `seen`
    (key = (name, imagenum), value = embedding array). Tránh nạp trùng
    cùng 1 ảnh nhiều lần."""
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for prefix in ("enroll", "verify"):
                key = (row[f"name_{prefix}"], row[f"imagenum_{prefix}"])
                if key not in seen:
                    try:
                        seen[key] = load_embedding(cache_dir, key[0], key[1])
                    except FileNotFoundError:
                        continue
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wifakey-data-dir", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument(
        "--n-thr",
        type=int,
        default=3,
        help="Số ngưỡng/chiều (khớp n_thr gốc, thường=3)",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    M_matrix = np.load(os.path.join(args.wifakey_data_dir, "M_matrix.npy"))
    n_dims = M_matrix.shape[0]  # thường = 512
    print(f"M_matrix shape: {M_matrix.shape} -> {n_dims} chiều projected")

    print(
        "Đang thu thập embedding duy nhất từ tune_genuine.csv + tune_impostor.csv ..."
    )
    seen = {}
    collect_unique_embeddings(
        os.path.join(args.pairs_dir, "tune_genuine.csv"), args.cache_dir, seen
    )
    collect_unique_embeddings(
        os.path.join(args.pairs_dir, "tune_impostor.csv"), args.cache_dir, seen
    )
    embeddings = np.stack(list(seen.values()), axis=0)
    print(f"Tổng {embeddings.shape[0]} embedding duy nhất.")

    projected = np.dot(embeddings, M_matrix)  # (n_samples, n_dims)
    print(f"Projected shape: {projected.shape}")

    n_thr = args.n_thr
    per_dim_intervals = np.zeros((n_dims, n_thr), dtype=np.float64)

    # Ngưỡng equiprobable: chia phân phối của MỖI chiều thành (n_thr+1) bin
    # có xác suất bằng nhau -- percentile tại 100*(k/(n_thr+1)) với k=1..n_thr.
    percentiles = [100 * k / (n_thr + 1) for k in range(1, n_thr + 1)]
    for j in range(n_dims):
        per_dim_intervals[j, :] = np.percentile(projected[:, j], percentiles)

    np.save(args.output, per_dim_intervals)
    print(
        f"\nĐã lưu per_dim_intervals shape {per_dim_intervals.shape} vào: {args.output}"
    )

    # So sánh nhanh với bộ ngưỡng chung cũ để thấy mức độ khác biệt giữa các chiều
    global_intervals_path = os.path.join(
        args.wifakey_data_dir, "binarization_intervals.npy"
    )
    if os.path.exists(global_intervals_path):
        global_intervals = np.load(global_intervals_path)
        print(
            f"\nBộ ngưỡng CHUNG cũ (toàn bộ 512 chiều dùng chung): {global_intervals}"
        )
        print(
            f"Ngưỡng RIÊNG theo chiều: min={per_dim_intervals.min():.4f}, "
            f"max={per_dim_intervals.max():.4f}, "
            f"độ lệch chuẩn giữa các chiều (trung bình theo cột): "
            f"{per_dim_intervals.std(axis=0)}"
        )
        print(
            "-> độ lệch chuẩn giữa các chiều CÀNG LỚN, ngưỡng riêng theo chiều "
            "càng có khả năng khác biệt có ý nghĩa so với dùng 1 bộ ngưỡng chung."
        )


if __name__ == "__main__":
    main()
