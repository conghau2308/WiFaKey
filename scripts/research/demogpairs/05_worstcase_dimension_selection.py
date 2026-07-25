"""
07_worstcase_dimension_selection.py

Thử tiêu chí chọn dimension THEO WORST-CASE thay vì trung bình gộp:
  worst_case_score[d] = min over folds of separation_d
rồi chọn 277 chiều có worst_case_score cao nhất — nghĩa là chiều đó phải
"đủ tốt" ở MỌI nhóm nhân khẩu học, không chỉ tốt trung bình hoặc tốt cho
nhóm chiếm ưu thế trong dữ liệu.

BẮT BUỘC: tách select/validate theo IDENTITY (không phải theo pair) để
tránh leakage — nửa "select" dùng để TÍNH worst_case_score và chọn top
dim, nửa "validate" (chưa từng dùng lúc chọn) dùng để ĐO separation thật
của tập dim đã chọn. Nếu không tách, kết quả sẽ lạc quan giả tạo do
overfitting vào chính dữ liệu dùng để chọn — cùng nguyên tắc tune/select
đã áp dụng cho LFW trước đó.

SO SÁNH 2 PHƯƠNG ÁN trên validate set (chưa từng thấy lúc chọn dim):
  - baseline: dim 0..276 (cắt theo vị trí, hệ thống gốc)
  - worst-case: 277 dim có worst_case_score cao nhất (tính từ select set)
Đo trên MỖI fold: genuine_ber, impostor_ber, separation trung bình qua
tập dim đã chọn — và đặc biệt: separation NHỎ NHẤT qua các fold (đúng
đại lượng ta đang cố tối ưu) có thật sự cải thiện so với baseline không.

Cách chạy:
    python scripts/07_worstcase_dimension_selection.py
"""

import os
import sys
import csv
import numpy as np
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_lib import utils

DATASET_NAME = "demogpairs"
DATA_DIR = os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
M_MATRIX_NPY = os.path.join(DATA_DIR, "M_matrix.npy")
INTERVALS_NPY = os.path.join(DATA_DIR, "binarization_intervals.npy")

IMAGE_METADATA_CSV = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "image_metadata.csv"
)
PAIRS_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs")
GENUINE_CSV = os.path.join(PAIRS_DIR, "audit_genuine.csv")
IMPOSTOR_SAMEFOLD_CSV = os.path.join(PAIRS_DIR, "audit_impostor_samefold.csv")
EMBEDDINGS_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)

OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "out_dim_separation_audit")
os.makedirs(OUT_DIR, exist_ok=True)

N_DIMS_BUDGET = 277
RNG_SEED = 42
SELECT_FRACTION = 0.5  # % identity dùng để chọn dim, còn lại để validate

M_matrix = np.load(M_MATRIX_NPY)
intervals = np.load(INTERVALS_NPY)
N_THR = np.asarray(intervals).reshape(-1).size
D_DIMS = M_matrix.shape[1]


def real_binarize(emb: np.ndarray) -> np.ndarray:
    projected = np.dot(np.asarray(emb, dtype=np.float64), M_matrix)
    return (
        utils.lssc_binary(projected[None, :], interval=intervals)
        .flatten()
        .astype(np.uint8)
    )


_embedding_cache = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    if cache_filename not in _embedding_cache:
        path = os.path.join(EMBEDDINGS_CACHE_DIR, cache_filename)
        _embedding_cache[cache_filename] = np.load(path)
    return _embedding_cache[cache_filename]


def load_identity_fold_map():
    identity_to_fold = {}
    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["cache_filename"]:
                identity_to_fold[row["identity"]] = row["fold"]
    return identity_to_fold


def split_identities_select_validate(identity_to_fold, rng):
    """Chia identity mỗi fold thành 2 tập rời nhau: select / validate."""
    by_fold = defaultdict(list)
    for identity, fold in identity_to_fold.items():
        by_fold[fold].append(identity)

    select_ids, validate_ids = set(), set()
    for fold, identities in by_fold.items():
        identities = sorted(identities)  # thứ tự cố định trước khi shuffle
        idx = rng.permutation(len(identities))
        n_select = int(len(identities) * SELECT_FRACTION)
        for k, i in enumerate(idx):
            if k < n_select:
                select_ids.add(identities[i])
            else:
                validate_ids.add(identities[i])
    return select_ids, validate_ids


def load_and_split_pairs(select_ids, validate_ids):
    """Trả về select_pairs[fold]={"genuine":[...], "impostor":[...]} và
    validate_pairs[fold]={...} tương tự — mỗi pair chỉ thuộc 1 tập, dựa
    trên identity (impostor pair bị BỎ nếu 2 identity nằm khác tập, để
    tránh leakage)."""
    select_pairs = defaultdict(lambda: {"genuine": [], "impostor": []})
    validate_pairs = defaultdict(lambda: {"genuine": [], "impostor": []})

    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            identity, fold = row["identity"], row["fold"]
            pair = (row["cache_filename_1"], row["cache_filename_2"])
            if identity in select_ids:
                select_pairs[fold]["genuine"].append(pair)
            elif identity in validate_ids:
                validate_pairs[fold]["genuine"].append(pair)

    n_dropped_impostor = 0
    with open(IMPOSTOR_SAMEFOLD_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fold = row["fold"]
            id1, id2 = row["identity_1"], row["identity_2"]
            pair = (row["cache_filename_1"], row["cache_filename_2"])
            if id1 in select_ids and id2 in select_ids:
                select_pairs[fold]["impostor"].append(pair)
            elif id1 in validate_ids and id2 in validate_ids:
                validate_pairs[fold]["impostor"].append(pair)
            else:
                n_dropped_impostor += 1  # 2 identity nằm khác tập -> bỏ, tránh leakage

    print(
        f"[split] Bỏ {n_dropped_impostor} impostor pairs do 2 identity nằm "
        f"khác tập select/validate (tránh leakage)."
    )
    return select_pairs, validate_pairs


def per_dimension_ber(pairs: list) -> np.ndarray:
    if not pairs:
        return np.full(D_DIMS, np.nan)
    flips_sum = np.zeros(D_DIMS * N_THR, dtype=np.float64)
    for f1, f2 in pairs:
        b1 = real_binarize(load_embedding(f1))
        b2 = real_binarize(load_embedding(f2))
        flips_sum += (b1 != b2).astype(np.float64)
    return (flips_sum / len(pairs)).reshape(D_DIMS, N_THR).mean(axis=1)


def compute_separation_all_folds(pairs_by_fold):
    """Trả về dict fold -> separation[512], chỉ cho fold có đủ dữ liệu."""
    sep_by_fold = {}
    for fold, pairs in pairs_by_fold.items():
        g, i = pairs["genuine"], pairs["impostor"]
        if len(g) < 10 or len(i) < 10:
            print(
                f"  *** {fold}: quá ít pairs ({len(g)} genuine / {len(i)} "
                f"impostor) trong tập này — bỏ qua fold. ***"
            )
            continue
        sep_by_fold[fold] = per_dimension_ber(i) - per_dimension_ber(g)
    return sep_by_fold


def evaluate_dimset(dim_set: set, pairs_by_fold: dict, label: str):
    """Đo genuine_ber/impostor_ber/separation trung bình TRÊN TẬP DIM ĐÃ
    CHỌN, cho mỗi fold, dùng pairs_by_fold (nên là validate set)."""
    dim_idx = sorted(dim_set)
    print(f"\n--- Đánh giá '{label}' trên validate set ---")
    fold_results = {}
    for fold, pairs in pairs_by_fold.items():
        g, i = pairs["genuine"], pairs["impostor"]
        if len(g) < 10 or len(i) < 10:
            continue
        gen_ber_full = per_dimension_ber(g)
        imp_ber_full = per_dimension_ber(i)
        gen_ber = np.nanmean(gen_ber_full[dim_idx])
        imp_ber = np.nanmean(imp_ber_full[dim_idx])
        sep = imp_ber - gen_ber
        fold_results[fold] = sep
        print(
            f"  {fold:16s}: genuine_ber={gen_ber:.4f}  impostor_ber={imp_ber:.4f}  "
            f"separation={sep:+.4f}"
        )
    if fold_results:
        worst = min(fold_results.values())
        avg = float(np.mean(list(fold_results.values())))
        print(
            f"  -> separation trung bình qua fold: {avg:+.4f}   "
            f"separation NHỎ NHẤT (worst-case thật, out-of-sample): {worst:+.4f}"
        )
    return fold_results


def main():
    rng = np.random.default_rng(RNG_SEED)

    identity_to_fold = load_identity_fold_map()
    select_ids, validate_ids = split_identities_select_validate(identity_to_fold, rng)
    print(f"Identity: select={len(select_ids)}  validate={len(validate_ids)}")

    select_pairs, validate_pairs = load_and_split_pairs(select_ids, validate_ids)

    print("\n=== Tính separation_d trên SELECT set (dùng để chọn dim) ===")
    sep_select = compute_separation_all_folds(select_pairs)
    if len(sep_select) < 2:
        print("*** Không đủ fold hợp lệ trong select set. Dừng. ***")
        return

    sep_matrix = np.stack(list(sep_select.values()))  # (n_folds, 512)
    worst_case_score = sep_matrix.min(axis=0)  # (512,) — điểm worst-case mỗi dim

    worstcase_dims = set(np.argsort(-worst_case_score)[:N_DIMS_BUDGET].tolist())
    baseline_dims = set(range(N_DIMS_BUDGET))
    overlap_wc_baseline = len(worstcase_dims & baseline_dims)
    print(
        f"\nOverlap worst-case-selected vs baseline (dim 0..{N_DIMS_BUDGET-1}): "
        f"{overlap_wc_baseline}/{N_DIMS_BUDGET} "
        f"({100*overlap_wc_baseline/N_DIMS_BUDGET:.1f}%)"
    )

    print("\n=== So sánh trên VALIDATE set (chưa từng dùng để chọn dim) ===")
    baseline_results = evaluate_dimset(
        baseline_dims, validate_pairs, "baseline (dim 0..276)"
    )
    worstcase_results = evaluate_dimset(
        worstcase_dims, validate_pairs, "worst-case selection"
    )

    common_folds = set(baseline_results) & set(worstcase_results)
    print("\n" + "=" * 70)
    print("KẾT LUẬN — cải thiện worst-case thật (out-of-sample) so với baseline")
    print("=" * 70)
    if common_folds:
        baseline_worst = min(baseline_results[f] for f in common_folds)
        worstcase_worst = min(worstcase_results[f] for f in common_folds)
        print(f"Baseline — separation nhỏ nhất qua các fold: {baseline_worst:+.4f}")
        print(
            f"Worst-case selection — separation nhỏ nhất qua các fold: {worstcase_worst:+.4f}"
        )
        if worstcase_worst > baseline_worst:
            print(
                "-> CẢI THIỆN thật: chọn dim theo worst-case giúp fold yếu nhất "
                "tốt hơn baseline, ngay cả khi đo trên dữ liệu chưa từng dùng để chọn."
            )
        else:
            print(
                "-> KHÔNG cải thiện (hoặc worse) trên validate set — tiêu chí "
                "worst-case bị overfit vào select set, hoặc hiệu ứng nhân khẩu học "
                "quá mạnh để 1 tập dim cố định giải quyết được."
            )

    np.savez(
        os.path.join(OUT_DIR, "worstcase_selection_result.npz"),
        worst_case_score=worst_case_score,
        worstcase_dims=np.array(sorted(worstcase_dims)),
        baseline_results=np.array(list(baseline_results.items()), dtype=object),
        worstcase_results=np.array(list(worstcase_results.items()), dtype=object),
    )
    print(f"\n[out] -> {os.path.join(OUT_DIR, 'worstcase_selection_result.npz')}")


if __name__ == "__main__":
    main()
