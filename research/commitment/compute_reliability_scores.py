"""
compute_reliability_scores.py

Tính điểm tin cậy Fisher F_i cho từng vị trí bit trong không gian nhị phân
đầy đủ (full_binary_length, sau M_matrix + LSSC), dùng đúng
tune_genuine.csv / tune_impostor.csv (Tầng 1) + embeddings_cache/ đã có sẵn
trong project — KHÔNG chạm vào select_*.csv hay pairs.csv (Tầng 3), giữ
đúng nguyên tắc tách calibration khỏi tập đánh giá cuối.

Công thức (per-bit, dựa trên tỉ lệ lật bit — bit là Bernoulli nên
var = p(1-p)):

    intra_flip_i = P(b_enroll_i != b_verify_i | cặp genuine)   (tính trên tune_genuine.csv)
    inter_flip_i = P(b_enroll_i != b_verify_i | cặp impostor)  (tính trên tune_impostor.csv)

    F_i = (inter_flip_i - intra_flip_i)^2 / (var(intra_flip_i) + var(inter_flip_i) + eps)

Bit "đáng tin" = intra_flip thấp (ổn định trong cùng người) VÀ inter_flip
cao, gần 0.5 (phân biệt tốt giữa người khác nhau) => F_i cao.

Đây là bước tính OFFLINE, 1 lần, population-level — không dùng dữ liệu
riêng của bất kỳ user cá nhân nào tại thời điểm enroll thật, nên an toàn để
dùng làm tham số công khai cho ReliabilitySelectionWiFaKeyHandler (v2).

Cách chạy (mặc định đường dẫn khớp cấu trúc project hiện tại, chạy từ
project_root):

    python research/commitment/compute_reliability_scores.py \\
        --project-root . \\
        --output research/commitment/reliability_tune.npy
"""

import argparse
import csv
import os

import numpy as np

from wifakey_module.wifakey_lib import utils as wfk_utils


def load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    path = os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy")
    return np.load(path)


def binarize(
    embedding: np.ndarray, M_matrix: np.ndarray, intervals: np.ndarray
) -> np.ndarray:
    """Tái tạo đúng WiFaKeyHandler._binarize_full, không cần khởi tạo TF session."""
    projected = np.dot(embedding, M_matrix)
    projected_2d = np.expand_dims(projected, axis=0)
    binary = wfk_utils.lssc_binary(projected_2d, interval=intervals).flatten()
    return binary.astype(np.uint8)


def compute_flip_rate(
    pairs_csv: str,
    cache_dir: str,
    M_matrix: np.ndarray,
    intervals: np.ndarray,
    full_len: int,
):
    flips_sum = np.zeros(full_len, dtype=np.float64)
    n_used = 0
    n_skipped = 0

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
                n_skipped += 1
                continue

            b1 = binarize(e1, M_matrix, intervals)
            b2 = binarize(e2, M_matrix, intervals)
            if b1.shape[0] != full_len:
                n_skipped += 1
                continue

            flips_sum += (b1 != b2).astype(np.float64)
            n_used += 1

    if n_used == 0:
        raise RuntimeError(f"Không có cặp hợp lệ nào trong {pairs_csv}")

    return flips_sum / n_used, n_used, n_skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".", help="Thư mục gốc project")
    ap.add_argument(
        "--wifakey-data-dir",
        default=None,
        help="Mặc định: <project-root>/wifakey_module/data",
    )
    ap.add_argument(
        "--pairs-dir",
        default=None,
        help="Mặc định: <project-root>/datasets/processed/labeled_faces_in_the_wild/pairs",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Mặc định: <project-root>/datasets/processed/labeled_faces_in_the_wild/embeddings_cache",
    )
    ap.add_argument(
        "--output", required=True, help="Đường dẫn lưu reliability_scores.npy"
    )
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )

    print(f"Load M_matrix / intervals từ {data_dir} ...")
    M_matrix = np.load(os.path.join(data_dir, "M_matrix.npy"))
    intervals = np.load(os.path.join(data_dir, "binarization_intervals.npy"))
    n_thr = int(np.asarray(intervals).size)
    full_len = M_matrix.shape[0] * n_thr
    print(
        f"full_binary_length = {full_len}  (M_matrix rows={M_matrix.shape[0]}, n_thr={n_thr})"
    )

    tune_genuine_csv = os.path.join(pairs_dir, "tune_genuine.csv")
    tune_impostor_csv = os.path.join(pairs_dir, "tune_impostor.csv")

    print(f"\nTính intra_flip_rate trên {tune_genuine_csv} ...")
    intra_flip, n_gen, skip_gen = compute_flip_rate(
        tune_genuine_csv, cache_dir, M_matrix, intervals, full_len
    )
    print(f"  Dùng {n_gen} cặp genuine (bỏ {skip_gen} cặp thiếu embedding).")
    print(f"  intra_flip trung bình toàn bộ dimension: {intra_flip.mean():.4f}")

    print(f"\nTính inter_flip_rate trên {tune_impostor_csv} ...")
    inter_flip, n_imp, skip_imp = compute_flip_rate(
        tune_impostor_csv, cache_dir, M_matrix, intervals, full_len
    )
    print(f"  Dùng {n_imp} cặp impostor (bỏ {skip_imp} cặp thiếu embedding).")
    print(f"  inter_flip trung bình toàn bộ dimension: {inter_flip.mean():.4f}")

    var_intra = intra_flip * (1 - intra_flip)
    var_inter = inter_flip * (1 - inter_flip)
    F = (inter_flip - intra_flip) ** 2 / (var_intra + var_inter + args.eps)

    print(f"\nF_i: min={F.min():.4f}, max={F.max():.4f}, mean={F.mean():.4f}")
    print(f"Top-10 dimension đáng tin nhất (index): {np.argsort(-F)[:10].tolist()}")

    np.save(args.output, F)
    print(f"\nĐã lưu reliability scores vào: {args.output}")
    print(
        "Dùng file này làm --reliability-scores-path khi khởi tạo "
        "ReliabilitySelectionWiFaKeyHandler (v2)."
    )


if __name__ == "__main__":
    main()
