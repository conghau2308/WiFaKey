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
        --output research/commitment/out_reliability_scores/reliability_tune.npy
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

    # LƯU Ý QUAN TRỌNG (rút ra từ benchmark thật trên tune set): với hệ thống
    # hiện tại, FAR đo được = 0.0000% ở MỌI mức M/kappa đã thử — nghĩa là
    # impostor separation đang dư thừa margin rất lớn, KHÔNG phải ràng buộc
    # đang bó buộc hệ thống. Công thức Fisher cân bằng cả intra VÀ inter sẽ
    # lãng phí "ngân sách chọn lọc" vào việc tối ưu inter_flip (không cần
    # thiết lúc này), đánh đổi với việc chọn đúng bit có intra_flip thấp
    # nhất — đây là nguyên nhân pool theo Fisher-score có thể cho GMR TỆ HƠN
    # cả chọn đều. Vì vậy dùng "F_i" chỉ dựa thuần trên intra_flip thấp,
    # bỏ hẳn ảnh hưởng của inter_flip khỏi tiêu chí xếp hạng.
    #
    # Nếu sau này FAR không còn dư thừa (ví dụ M lớn hơn khiến impostor bị
    # kéo gần ngưỡng), cần khôi phục lại thành phần inter_flip — nhưng phải
    # đo lại FAR thật trước khi quyết định, không giả định trước.
    eps = args.eps
    F = -intra_flip  # xếp hạng giảm dần theo -intra_flip == tăng dần theo intra_flip
    F_fisher_reference = (inter_flip - intra_flip) / np.sqrt(
        var_intra + var_inter + eps
    )

    n_inverted = int((inter_flip < intra_flip).sum())
    print(
        f"\nSố dimension có intra_flip > inter_flip (bit 'ngược hướng', PHẢI bị "
        f"loại khỏi pool): {n_inverted} / {full_len} "
        f"({100*n_inverted/full_len:.1f}%)"
    )

    print(
        f"\nF_i (intra-only, dùng để chọn pool): min={F.min():.4f}, max={F.max():.4f}, mean={F.mean():.4f}"
    )
    print(
        f"  (đối chiếu) Fisher-score cũ (intra+inter): min={F_fisher_reference.min():.4f}, "
        f"max={F_fisher_reference.max():.4f}, mean={F_fisher_reference.mean():.4f}"
    )
    top10_intra_only = np.argsort(-F)[:10]
    top10_fisher = np.argsort(-F_fisher_reference)[:10]
    overlap = len(set(top10_intra_only.tolist()) & set(top10_fisher.tolist()))
    print(f"  Số dimension trùng nhau trong top-10 giữa 2 tiêu chí: {overlap}/10")
    print(
        f"Top-10 dimension đáng tin nhất (intra_flip thấp nhất): {top10_intra_only.tolist()}"
    )

    np.save(args.output, F)
    print(f"\nĐã lưu reliability scores vào: {args.output}")
    print(
        "Dùng file này làm --reliability-scores-path khi khởi tạo "
        "ReliabilitySelectionWiFaKeyHandler (v2)."
    )


if __name__ == "__main__":
    main()
