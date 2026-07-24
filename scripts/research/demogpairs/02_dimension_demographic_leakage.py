"""
04_dimension_demographic_leakage.py

Đo mức độ mỗi chiều trong embedding 512-d (AdaFace) "rò rỉ" thông tin
nhân khẩu học (fold: Asian/Black/White x Females/Males), rồi đối chiếu
với ranking Fisher-ratio (identity-discriminability) đã chọn trước đó.

ĐƠN VỊ MẪU LÀ IDENTITY, KHÔNG PHẢI ẢNH — QUAN TRỌNG:
  Nhiều ảnh của cùng 1 identity không độc lập với nhau (cùng khuôn mặt,
  cùng fold). Nếu đưa thẳng từng ảnh vào ANOVA, các mẫu trùng lặp giả
  (pseudo-replication) sẽ làm F-statistic bị thổi phồng giả tạo — hệ
  quả là "tìm thấy" leakage ở mức không có thật. Vì vậy script này
  gộp embedding các ảnh cùng identity thành 1 vector trung bình
  (identity-level), rồi mới chạy ANOVA trên các vector đó theo fold.
  Đơn vị thống kê = identity, không phải ảnh.

PHƯƠNG PHÁP:
  Với mỗi chiều d trong 512 chiều: chạy 1-way ANOVA (scipy.f_oneway)
  giữa 6 nhóm fold, lấy F-statistic + p-value + eta-squared (effect
  size, để tránh chỉ nhìn p-value — với n_identity lớn, p-value có
  thể "significant" dù effect size rất nhỏ, không thực sự đáng lo).
  Sau đó hiệu chỉnh multiple-comparison (512 test cùng lúc) bằng
  Benjamini-Hochberg FDR — không dùng ngưỡng p<0.05 thô cho từng test.

CẦN BẠN CUNG CẤP: đường dẫn tới ranking Fisher-ratio identity đã tính
trước đó (FISHER_RATIO_SCORES_PATH bên dưới) để chạy được phần so sánh
overlap ở cuối. Nếu chưa có, script vẫn chạy được phần leakage score,
chỉ bỏ qua bước so sánh cuối (sẽ in cảnh báo rõ ràng, không âm thầm bỏ).

Cách chạy:
    python scripts/04_dimension_demographic_leakage.py
"""

import os
import sys
import csv
import numpy as np
from collections import defaultdict
from scipy import stats
from statsmodels.stats.multitest import multipletests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

DATASET_NAME = "demogpairs"

IMAGE_METADATA_CSV = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "image_metadata.csv"
)
EMBEDDINGS_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)
OUTPUT_CSV = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "dimension_leakage_scores.csv"
)

# TODO(bạn): trỏ tới file ranking Fisher-ratio identity đã tính trước đó.
# GIẢ ĐỊNH TẠM: file .npy shape (512,), giá trị càng cao = dimension càng
# discriminative cho identity. SỬA lại theo đúng format thật của bạn (có
# thể là .csv 2 cột [dim_index, fisher_score], hoặc list 277 index đã chọn
# sẵn — nếu vậy sửa hàm load_fisher_ranking() bên dưới).
FISHER_RATIO_SCORES_PATH = os.path.join(
    _PROJECT_ROOT, "research", "dimension_selection", "fisher_ratio_scores.npy"
)
N_SELECTED_DIMS = 277  # số chiều đã/định chọn theo Fisher-ratio, dùng cho top-K overlap

EMBEDDING_DIM = 512
FDR_ALPHA = 0.05


def load_identity_fold_map():
    """Đọc image_metadata.csv -> {identity: fold}. Không dùng ảnh lẻ ở đây,
    chỉ cần biết identity nào thuộc fold nào (đã đảm bảo 1-1 ở bước 03b)."""
    identity_to_fold = {}
    identity_to_cache_files = defaultdict(list)

    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["cache_filename"]:
                continue
            identity = row["identity"]
            identity_to_fold[identity] = row["fold"]
            identity_to_cache_files[identity].append(row["cache_filename"])

    return identity_to_fold, identity_to_cache_files


def compute_identity_mean_embeddings(identity_to_cache_files):
    """Với mỗi identity, load toàn bộ embedding ảnh của nó, lấy trung bình
    -> 1 vector 512-d đại diện cho identity đó (đơn vị thống kê đúng)."""
    identity_mean_emb = {}
    n_missing = 0

    for identity, cache_files in identity_to_cache_files.items():
        vecs = []
        for fname in cache_files:
            fpath = os.path.join(EMBEDDINGS_CACHE_DIR, fname)
            if not os.path.exists(fpath):
                n_missing += 1
                continue
            vecs.append(np.load(fpath))
        if not vecs:
            continue
        identity_mean_emb[identity] = np.mean(np.stack(vecs, axis=0), axis=0)

    if n_missing > 0:
        print(
            f"*** CẢNH BÁO: {n_missing} file embedding trong metadata "
            f"không tìm thấy trên đĩa (cache bị xóa/di chuyển?). ***"
        )

    return identity_mean_emb


def run_anova_per_dimension(identity_mean_emb, identity_to_fold):
    """Trả về list dict: [{dim, f_stat, p_value, eta_sq}, ...] độ dài 512."""
    fold_groups = defaultdict(list)  # fold -> list of 512-d vectors
    for identity, emb in identity_mean_emb.items():
        fold = identity_to_fold[identity]
        fold_groups[fold].append(emb)

    fold_names = sorted(fold_groups.keys())
    print(
        "Số identity dùng cho ANOVA theo fold:",
        {f: len(fold_groups[f]) for f in fold_names},
    )
    for f in fold_names:
        if len(fold_groups[f]) < 5:
            print(
                f"*** CẢNH BÁO: fold '{f}' chỉ có {len(fold_groups[f])} "
                f"identity — quá ít để ANOVA đáng tin cậy. ***"
            )

    fold_matrices = {f: np.stack(v, axis=0) for f, v in fold_groups.items()}
    n_total = sum(m.shape[0] for m in fold_matrices.values())

    results = []
    for d in range(EMBEDDING_DIM):
        groups_d = [fold_matrices[f][:, d] for f in fold_names]
        f_stat, p_value = stats.f_oneway(*groups_d)

        # eta-squared = SS_between / SS_total (effect size, độc lập cỡ mẫu)
        grand_mean = np.concatenate(groups_d).mean()
        ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups_d)
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_d)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        results.append(
            {
                "dim": d,
                "f_stat": f_stat if np.isfinite(f_stat) else 0.0,
                "p_value": p_value if np.isfinite(p_value) else 1.0,
                "eta_sq": eta_sq,
            }
        )

    # Hiệu chỉnh multiple comparison (512 test đồng thời) — Benjamini-Hochberg FDR
    p_values = [r["p_value"] for r in results]
    reject, p_adj, _, _ = multipletests(p_values, alpha=FDR_ALPHA, method="fdr_bh")
    for r, rej, padj in zip(results, reject, p_adj):
        r["p_adj_fdr"] = padj
        r["significant_after_fdr"] = bool(rej)

    return results, n_total


def load_fisher_ranking():
    """Trả về mảng (512,) điểm Fisher-ratio identity, hoặc None nếu chưa
    có file / format không khớp giả định .npy shape (512,)."""
    if not os.path.exists(FISHER_RATIO_SCORES_PATH):
        print(
            f"\n*** Không tìm thấy {FISHER_RATIO_SCORES_PATH} — bỏ qua "
            f"bước so sánh overlap với Fisher-ratio ranking. Cung cấp "
            f"đúng đường dẫn/format thật rồi chạy lại phần này. ***"
        )
        return None

    scores = np.load(FISHER_RATIO_SCORES_PATH)
    if scores.shape != (EMBEDDING_DIM,):
        print(
            f"*** CẢNH BÁO: {FISHER_RATIO_SCORES_PATH} có shape "
            f"{scores.shape}, không khớp giả định ({EMBEDDING_DIM},). "
            f"Sửa load_fisher_ranking() cho đúng format thật. Bỏ qua so sánh. ***"
        )
        return None
    return scores


def compare_rankings(leakage_results, fisher_scores):
    """So sánh overlap giữa top-K leakage-score cao nhất và top-K
    fisher-score cao nhất (K = N_SELECTED_DIMS)."""
    leakage_sorted = sorted(leakage_results, key=lambda r: r["f_stat"], reverse=True)
    leakage_top = set(r["dim"] for r in leakage_sorted[:N_SELECTED_DIMS])

    fisher_order = np.argsort(-fisher_scores)  # giảm dần
    fisher_top = set(fisher_order[:N_SELECTED_DIMS].tolist())

    overlap = leakage_top & fisher_top
    jaccard = len(overlap) / len(leakage_top | fisher_top)

    # Spearman rank correlation trên toàn bộ 512 chiều (không chỉ top-K)
    leakage_rank = np.array([r["f_stat"] for r in leakage_results])
    spearman_rho, spearman_p = stats.spearmanr(leakage_rank, fisher_scores)

    print(
        f"\n=== SO SÁNH LEAKAGE RANKING vs FISHER RANKING (top {N_SELECTED_DIMS}) ==="
    )
    print(
        f"Overlap: {len(overlap)}/{N_SELECTED_DIMS} chiều "
        f"({100 * len(overlap) / N_SELECTED_DIMS:.1f}%)"
    )
    print(f"Jaccard index (top-K): {jaccard:.4f}")
    print(
        f"Spearman rho (toàn bộ 512 chiều, leakage vs fisher): "
        f"{spearman_rho:.4f} (p={spearman_p:.2e})"
    )
    print(
        "\nDIỄN GIẢI:\n"
        "  - Overlap cao + Spearman rho dương mạnh -> các chiều Fisher chọn\n"
        "    cũng chính là chiều rò rỉ nhân khẩu học nhiều nhất -> CẦN sửa\n"
        "    cách chọn dimension (multi-objective, xem đề xuất trước đó).\n"
        "  - Overlap thấp / Spearman rho gần 0 hoặc âm -> Fisher-ratio hiện\n"
        "    tại tình cờ an toàn, không cần sửa cách chọn dimension, nhưng\n"
        "    vẫn nên báo cáo con số này như bằng chứng khoa học.\n"
    )
    return {
        "overlap_count": len(overlap),
        "overlap_dims": sorted(overlap),
        "jaccard": jaccard,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def main():
    identity_to_fold, identity_to_cache_files = load_identity_fold_map()
    print(f"Đã load {len(identity_to_fold)} identity từ image_metadata.csv.")

    identity_mean_emb = compute_identity_mean_embeddings(identity_to_cache_files)
    print(f"Tính được embedding trung bình cho {len(identity_mean_emb)} identity.")

    leakage_results, n_total = run_anova_per_dimension(
        identity_mean_emb, identity_to_fold
    )

    n_sig = sum(1 for r in leakage_results if r["significant_after_fdr"])
    print(
        f"\nSố chiều có leakage 'significant' sau hiệu chỉnh FDR (alpha={FDR_ALPHA}): "
        f"{n_sig}/{EMBEDDING_DIM}"
    )

    # Ghi kết quả đầy đủ ra CSV, sort theo F-statistic giảm dần (leakage cao nhất trước)
    leakage_sorted = sorted(leakage_results, key=lambda r: r["f_stat"], reverse=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dim", "f_stat", "p_value", "p_adj_fdr", "significant_after_fdr", "eta_sq"]
        )
        for r in leakage_sorted:
            writer.writerow(
                [
                    r["dim"],
                    r["f_stat"],
                    r["p_value"],
                    r["p_adj_fdr"],
                    r["significant_after_fdr"],
                    r["eta_sq"],
                ]
            )
    print(f"Đã ghi ranking đầy đủ (sort theo F-stat) vào: {OUTPUT_CSV}")

    print("\nTop 15 chiều leakage cao nhất (dim, F-stat, eta_sq, sig_after_fdr):")
    for r in leakage_sorted[:15]:
        print(
            f"  dim={r['dim']:>3}  F={r['f_stat']:>8.2f}  "
            f"eta_sq={r['eta_sq']:.4f}  sig={r['significant_after_fdr']}"
        )

    fisher_scores = load_fisher_ranking()
    if fisher_scores is not None:
        compare_rankings(leakage_results, fisher_scores)


if __name__ == "__main__":
    main()
