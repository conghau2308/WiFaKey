"""
sweep_pool_size.py

Sweep tham số pool_size (M) cho ReliabilitySelectionWiFaKeyHandler (v2), dùng
dữ liệu THẬT (embeddings_cache + tune_genuine.csv/tune_impostor.csv), thay vì
số liệu mô phỏng tổng hợp. Tái tạo đúng dạng bảng đã dùng để so sánh trước đó
(genuine P95, tỷ lệ pass, impostor margin) cho từng M candidate.

Cách tính: với mỗi cặp (enroll, verify), selection là 1 lần chọn ngẫu nhiên
feature_length vị trí trong pool M vị trí đáng tin nhất — đúng như thực tế
(mỗi lần enroll chỉ chọn 1 lần). BER = Hamming(b1[sel], b2[sel]) / feature_length.
Lặp lại toàn bộ pool trials với các seed khác nhau (--trials-per-pair) để làm
mượt phân phối P95 (mặc định 5, vì P95 ước lượng từ 1 mẫu/cặp có thể noisy).

Cách chạy:
    python research/commitment/sweep_pool_size.py \\
        --project-root . \\
        --reliability-scores research/commitment/reliability_tune.npy \\
        --feature-length 832 \\
        --pool-sizes 832,1000,1100,1200,1400,1600 \\
        --threshold 0.1762 \\
        --trials-per-pair 5
"""

import argparse
import csv
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_lib import utils as wfk_utils


def load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    path = os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy")
    return np.load(path)


def binarize(
    embedding: np.ndarray, M_matrix: np.ndarray, intervals: np.ndarray
) -> np.ndarray:
    projected = np.dot(embedding, M_matrix)
    projected_2d = np.expand_dims(projected, axis=0)
    binary = wfk_utils.lssc_binary(projected_2d, interval=intervals).flatten()
    return binary.astype(np.uint8)


def load_pair_flip_vectors(
    pairs_csv: str, cache_dir: str, M_matrix, intervals, full_len: int
):
    """Trả về ma trận (n_pairs, full_len) các vector XOR (flip indicator) cho mỗi cặp."""
    flips = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e1 = load_embedding(
                    cache_dir, row["name_enroll"], row["imagenum_enroll"]
                )
                e2 = load_embedding(
                    cache_dir, row["name_verify"], row["imagenum_verify"]
                )
            except FileNotFoundError:
                continue
            b1 = binarize(e1, M_matrix, intervals)
            b2 = binarize(e2, M_matrix, intervals)
            if b1.shape[0] != full_len:
                continue
            flips.append((b1 != b2).astype(np.uint8))
    return np.array(flips, dtype=np.uint8)  # (n_pairs, full_len)


def ber_distribution_for_pool(
    flip_matrix: np.ndarray,
    pool: np.ndarray,
    feature_length: int,
    trials_per_pair: int,
    rng,
):
    """
    Với mỗi cặp, lấy trials_per_pair lần chọn ngẫu nhiên feature_length vị trí
    trong pool -> tính BER = mean(flip) trên các vị trí đã chọn.
    Trả về mảng phẳng tất cả BER (n_pairs * trials_per_pair,).
    """
    n_pairs = flip_matrix.shape[0]
    bers = np.empty(n_pairs * trials_per_pair, dtype=np.float64)
    idx = 0
    for i in range(n_pairs):
        row = flip_matrix[i]
        for _ in range(trials_per_pair):
            sel = rng.choice(pool, size=feature_length, replace=False)
            bers[idx] = row[sel].mean()
            idx += 1
    return bers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--reliability-scores",
        required=True,
        help="File .npy từ compute_reliability_scores.py",
    )
    ap.add_argument("--feature-length", type=int, default=832)
    ap.add_argument(
        "--pool-sizes",
        type=str,
        required=True,
        help="Danh sách M candidate, phân cách dấu phẩy, vd: 832,1000,1200,1400",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.1762, help="Ngưỡng sửa lỗi LDPC (BER)"
    )
    ap.add_argument("--trials-per-pair", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )

    M_matrix = np.load(os.path.join(data_dir, "M_matrix.npy"))
    intervals = np.load(os.path.join(data_dir, "binarization_intervals.npy"))
    full_len = M_matrix.shape[0] * int(np.asarray(intervals).size)

    F = np.load(args.reliability_scores)
    assert (
        F.shape[0] == full_len
    ), f"reliability_scores length {F.shape[0]} != full_len {full_len}"

    print("Đang load flip vectors cho genuine/impostor pairs (tune set) ...")
    gen_flips = load_pair_flip_vectors(
        os.path.join(pairs_dir, "tune_genuine.csv"),
        cache_dir,
        M_matrix,
        intervals,
        full_len,
    )
    imp_flips = load_pair_flip_vectors(
        os.path.join(pairs_dir, "tune_impostor.csv"),
        cache_dir,
        M_matrix,
        intervals,
        full_len,
    )
    print(
        f"  genuine pairs: {gen_flips.shape[0]}, impostor pairs: {imp_flips.shape[0]}"
    )

    pool_sizes = [int(x) for x in args.pool_sizes.split(",")]
    rng = np.random.default_rng(args.seed)

    header = f"{'M':>8} | {'genuine P95':>12} | {'genuine pass%':>14} | {'impostor min%':>14} | {'impostor P5%':>13}"
    print("\n" + header)
    print("-" * len(header))

    for M in pool_sizes:
        if M < args.feature_length:
            print(f"{M:>8} | bỏ qua (M < feature_length={args.feature_length})")
            continue
        if M > full_len:
            print(f"{M:>8} | bỏ qua (M > full_binary_length={full_len})")
            continue

        pool = np.argsort(-F)[:M]

        gen_bers = ber_distribution_for_pool(
            gen_flips, pool, args.feature_length, args.trials_per_pair, rng
        )
        imp_bers = ber_distribution_for_pool(
            imp_flips, pool, args.feature_length, args.trials_per_pair, rng
        )

        gen_p95 = np.percentile(gen_bers, 95)
        gen_pass_rate = (gen_bers <= args.threshold).mean() * 100
        imp_min = imp_bers.min()
        imp_p5 = np.percentile(imp_bers, 5)

        print(
            f"{M:>8} | {gen_p95*100:>11.2f}% | {gen_pass_rate:>13.1f}% | "
            f"{imp_min*100:>13.2f}% | {imp_p5*100:>12.2f}%"
        )

    print(
        "\nChọn M nhỏ nhất đạt genuine pass% ~100 và impostor min%/P5% vẫn cách xa "
        f"ngưỡng {args.threshold*100:.2f}% — đó là điểm cân bằng usability/reroll tốt nhất "
        "để đưa vào ReliabilitySelectionWiFaKeyHandler trước khi chạy lại kiểm chứng trên select_*."
    )


if __name__ == "__main__":
    main()
