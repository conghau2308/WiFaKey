"""
05_dimension_separation_audit.py

Đo Genuine_BER_d / Impostor_BER_d THEO TỪNG DIMENSION (0..511, mỗi dim
ứng N_THR bit, dim-major — xác nhận qua self-check, không giả định),
TÁCH RIÊNG theo từng fold nhân khẩu học trong DemogPairs. Sau đó audit
chéo: thứ hạng "dimension nào tốt nhất" có ổn định qua các fold không.

Đây là bằng chứng go/no-go trực tiếp nhất cho việc chọn lại 277 dimension
(thay vì cắt theo vị trí) — đo THẲNG trên bit thật sau lượng tử hóa (dùng
đúng M_matrix + utils.lssc_binary gốc), không phải proxy trên embedding
liên tục như Fisher-ratio (04_fisher_and_leakage_by_dimension.py). Coi
04 là phụ (câu hỏi privacy/leakage riêng), 05 này là chính cho câu hỏi
"chọn dim theo separation có công bằng giữa các nhóm không".

DÙNG TRỰC TIẾP OUTPUT CÓ SẴN CỦA 03a/03b — không cần file trung gian mới:
  - datasets/processed/demogpairs/image_metadata.csv (từ 03a, không dùng
    trực tiếp ở đây nhưng là nguồn gốc của 2 file dưới)
  - datasets/processed/demogpairs/pairs/audit_genuine.csv (từ 03b)
  - datasets/processed/demogpairs/pairs/audit_impostor_samefold.csv (từ 03b)
  Cố tình KHÔNG dùng audit_impostor_crossfold.csv ở đây — separation_d
  phải đo impostor CÙNG FOLD với genuine để so sánh công bằng giữa các
  fold (khác fold sẽ trộn 2 nguồn biến thiên, không tách được).

LƯU Ý VỀ CỠ MẪU (từ thông tin bạn cung cấp): mỗi identity ban đầu có
12-13 ảnh, nhưng sau lọc FaceProcessor (low-confidence/no-face/spoof)
chỉ còn TRUNG BÌNH 3-4 ảnh/identity. Nghĩa là mỗi identity chỉ đóng góp
~3-6 cặp genuine (C(3,2)=3 đến C(4,2)=6) — standard error của
separation_d ở mức fold có thể đáng kể, KHÔNG bỏ qua cột SE trong kết
quả, không chỉ nhìn giá trị trung bình.

Cách chạy:
    python scripts/05_dimension_separation_audit.py
"""

import os
import sys
import csv
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_lib import utils

DATASET_NAME = "demogpairs"
DATA_DIR = os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
M_MATRIX_NPY = os.path.join(DATA_DIR, "M_matrix.npy")
INTERVALS_NPY = os.path.join(DATA_DIR, "binarization_intervals.npy")

PAIRS_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs")
GENUINE_CSV = os.path.join(PAIRS_DIR, "audit_genuine.csv")
IMPOSTOR_SAMEFOLD_CSV = os.path.join(PAIRS_DIR, "audit_impostor_samefold.csv")
EMBEDDINGS_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)

OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "out_dim_separation_audit")
os.makedirs(OUT_DIR, exist_ok=True)

N_DIMS_BASELINE_BUDGET = 277  # khớp 04_fisher_and_leakage_by_dimension.py
SE_WARNING_THRESHOLD = 0.03  # SE(separation) > 3% -> cảnh báo nghi ngờ nhiễu

FOLDS = [
    "Asian_Females",
    "Asian_Males",
    "Black_Females",
    "Black_Males",
    "White_Females",
    "White_Males",
]

# ---- Load ma trận lượng tử hóa gốc, suy ra N_THR/D_DIMS từ chính dữ liệu ----
M_matrix = np.load(M_MATRIX_NPY)
intervals = np.load(INTERVALS_NPY)
thr_sorted = np.sort(np.asarray(intervals, dtype=np.float64).reshape(-1))
rev_thr = thr_sorted[::-1]
N_THR = thr_sorted.size
D_DIMS = M_matrix.shape[1]
print(f"[load] M_matrix={M_matrix.shape}  N_THR={N_THR}  D_DIMS={D_DIMS}")


def real_binarize(emb: np.ndarray) -> np.ndarray:
    """Dùng đúng hàm gốc utils.lssc_binary — nguồn sự thật duy nhất."""
    projected = np.dot(np.asarray(emb, dtype=np.float64), M_matrix)
    return (
        utils.lssc_binary(projected[None, :], interval=intervals)
        .flatten()
        .astype(np.uint8)
    )


def _candidate_thermometer(v: np.ndarray) -> np.ndarray:
    """Công thức thermometer-code tự viết, CHỈ để đối chiếu self-check —
    không dùng để tính kết quả thật (real_binarize mới là nguồn thật)."""
    cmp = v[:, None] >= rev_thr[None, :]
    return cmp.astype(np.uint8).reshape(-1)


def self_check(sample_embs: list, n_probe: int = 8):
    for emb in sample_embs[:n_probe]:
        v = np.dot(np.asarray(emb, np.float64), M_matrix)
        cand = _candidate_thermometer(v)
        real = real_binarize(emb)
        if not np.array_equal(cand, real):
            raise SystemExit(
                "[self-check] FAIL — công thức đối chiếu không khớp "
                "utils.lssc_binary gốc. DỪNG, không tin bất kỳ số nào bên dưới "
                "cho tới khi tìm ra chỗ lệch."
            )
    print(
        f"[self-check] PASS trên {min(n_probe, len(sample_embs))} mẫu — "
        f"xác nhận dim-major layout và N_THR bit/dim đúng như giả định."
    )


_embedding_cache = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    if cache_filename not in _embedding_cache:
        path = os.path.join(EMBEDDINGS_CACHE_DIR, cache_filename)
        _embedding_cache[cache_filename] = np.load(path)
    return _embedding_cache[cache_filename]


def load_pairs_by_fold():
    """Đọc audit_genuine.csv + audit_impostor_samefold.csv (từ 03b),
    trả về {fold: {"genuine": [(cache1,cache2),...], "impostor": [...]}}."""
    pairs_by_fold = defaultdict(lambda: {"genuine": [], "impostor": []})

    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs_by_fold[row["fold"]]["genuine"].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )

    with open(IMPOSTOR_SAMEFOLD_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs_by_fold[row["fold"]]["impostor"].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )

    return pairs_by_fold


def per_dimension_ber(pairs: list) -> tuple:
    """Trả (ber_per_dim shape (D_DIMS,), n_pairs)."""
    if not pairs:
        return np.full(D_DIMS, np.nan), 0
    flips_sum = np.zeros(D_DIMS * N_THR, dtype=np.float64)
    for f1, f2 in pairs:
        b1 = real_binarize(load_embedding(f1))
        b2 = real_binarize(load_embedding(f2))
        flips_sum += (b1 != b2).astype(np.float64)
    flips_mean_per_bit = flips_sum / len(pairs)
    return flips_mean_per_bit.reshape(D_DIMS, N_THR).mean(axis=1), len(pairs)


def standard_error(p: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.full_like(p, np.nan)
    return np.sqrt(np.clip(p * (1 - p), 0, None) / n)


def analyse_fold(fold_name: str, pairs: dict) -> dict:
    genuine_pairs = pairs["genuine"]
    impostor_pairs = pairs["impostor"]

    gen_ber, n_gen = per_dimension_ber(genuine_pairs)
    imp_ber, n_imp = per_dimension_ber(impostor_pairs)
    separation = imp_ber - gen_ber

    se_gen = standard_error(gen_ber, n_gen)
    se_imp = standard_error(imp_ber, n_imp)
    se_sep = np.sqrt(se_gen**2 + se_imp**2)

    print(f"[{fold_name}] genuine_pairs={n_gen}  impostor_pairs={n_imp}")
    if n_gen == 0 or n_imp == 0:
        print(
            f"  *** CẢNH BÁO: fold '{fold_name}' thiếu genuine hoặc "
            f"impostor pairs — bỏ qua fold này trong audit chéo. ***"
        )

    n_high_se = int(np.nansum(se_sep > SE_WARNING_THRESHOLD))
    print(
        f"  -> {n_high_se}/{D_DIMS} dimension có SE(separation) > "
        f"{SE_WARNING_THRESHOLD:.0%} (nghi ngờ nhiễu thống kê nếu tỷ lệ cao, "
        f"đặc biệt với ~3-4 ảnh/identity như DemogPairs sau lọc)"
    )

    return {
        "fold": fold_name,
        "genuine_ber": gen_ber,
        "impostor_ber": imp_ber,
        "separation": separation,
        "se_separation": se_sep,
        "n_genuine": n_gen,
        "n_impostor": n_imp,
    }


def audit_cross_fold(results: list):
    valid = [r for r in results if r["n_genuine"] > 0 and r["n_impostor"] > 0]
    if len(valid) < 2:
        print("\n*** Không đủ fold hợp lệ (>=2) để audit chéo. Dừng. ***")
        return None, None

    names = [r["fold"] for r in valid]
    sep_matrix = np.stack([r["separation"] for r in valid])

    print("\n" + "=" * 70)
    print("AUDIT CHÉO FOLD — Spearman rank correlation của separation_d")
    print("=" * 70)
    n = len(names)
    corr_matrix = np.eye(n)
    rhos = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, p = spearmanr(sep_matrix[i], sep_matrix[j])
            corr_matrix[i, j] = corr_matrix[j, i] = rho
            rhos.append(rho)
            print(f"  {names[i]:16s} vs {names[j]:16s}: rho={rho:+.3f} (p={p:.1e})")

    print("\n" + "-" * 70)
    print(f"OVERLAP top-{N_DIMS_BASELINE_BUDGET} dimension giữa các fold (Jaccard)")
    print("-" * 70)
    top_sets = []
    for r in valid:
        order = np.argsort(-r["separation"])
        top_sets.append(set(order[:N_DIMS_BASELINE_BUDGET].tolist()))
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccard = inter / union if union else 0.0
            print(
                f"  {names[i]:16s} vs {names[j]:16s}: "
                f"overlap={inter}/{N_DIMS_BASELINE_BUDGET} "
                f"({inter/N_DIMS_BASELINE_BUDGET:.1%})  jaccard={jaccard:.3f}"
            )

    avg_rho = float(np.mean(rhos))
    print("\n" + "=" * 70)
    print(f"Trung bình Spearman rho giữa mọi cặp fold: {avg_rho:+.3f}")
    if avg_rho >= 0.7:
        verdict = (
            "GO — xếp hạng dimension ỔN ĐỊNH qua các fold nhân khẩu học. "
            "An toàn để gộp dữ liệu, chọn dim theo separation trên tập gộp."
        )
    elif avg_rho >= 0.4:
        verdict = (
            "BIÊN GIỚI — có tương quan nhưng không mạnh. Cân nhắc chọn dim "
            "theo tiêu chí WORST-CASE (min separation qua mọi fold) thay vì "
            "trung bình gộp, để không thiên vị fold chiếm ưu thế trong dữ liệu."
        )
    else:
        verdict = (
            "NO-GO — xếp hạng dimension KHÁC NHAU đáng kể giữa các fold. "
            "Bằng chứng thiên lệch thật. Cân nhắc dừng hướng chọn dim theo "
            "separation, hoặc báo cáo đây như phát hiện khoa học (âm tính) "
            "và giữ nguyên '832 bit đầu' như phương án trung lập."
        )
    print(verdict)
    print("=" * 70)

    return corr_matrix, top_sets


def main():
    pairs_by_fold = load_pairs_by_fold()
    missing_folds = [f for f in FOLDS if f not in pairs_by_fold]
    if missing_folds:
        print(
            f"*** CẢNH BÁO: các fold sau không có trong audit_genuine.csv/"
            f"audit_impostor_samefold.csv: {missing_folds} ***"
        )

    # self-check trên vài embedding thật đầu tiên gặp được
    probe_files = []
    for fold_data in pairs_by_fold.values():
        for f1, f2 in fold_data["genuine"][:4]:
            probe_files.extend([f1, f2])
        if len(probe_files) >= 8:
            break
    self_check([load_embedding(f) for f in probe_files[:8]])

    results = [
        analyse_fold(fold_name, pairs_by_fold[fold_name]) for fold_name in pairs_by_fold
    ]

    corr_matrix, top_sets = audit_cross_fold(results)

    out_path = os.path.join(OUT_DIR, "separation_by_fold.npz")
    np.savez(
        out_path,
        folds=[r["fold"] for r in results],
        separation=np.stack([r["separation"] for r in results]),
        genuine_ber=np.stack([r["genuine_ber"] for r in results]),
        impostor_ber=np.stack([r["impostor_ber"] for r in results]),
        se_separation=np.stack([r["se_separation"] for r in results]),
        corr_matrix=corr_matrix if corr_matrix is not None else np.array([]),
    )
    print(f"\n[out] Kết quả thô -> {out_path}")


if __name__ == "__main__":
    main()
