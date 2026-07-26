"""
09_decorrelation_impostor_margin.py

Thử ý tưởng: trừ mean quần thể + bỏ 1-2 thành phần PCA toàn cục đầu
("mean face" — mã hóa ánh sáng/pose/nhân khẩu, không phải danh tính)
TRƯỚC khi lượng tử hóa, xem có tăng Impostor_BER mà ít ảnh hưởng
Genuine_BER hay không (mục tiêu: tăng margin, không phải giảm FAR trung
bình — FAR đã xác nhận =0/10500 ở bước trước).

BA BIẾN THỂ, để cô lập đúng hiệu ứng của decorrelation (không lẫn với
hiệu ứng "chỉ cần refit ngưỡng trên DemogPairs" — 1 yếu tố gây nhiễu dễ
bị bỏ sót):
  0. production   — M_matrix + binarization_intervals.npy GỐC, y hệt 05
                     (self-check đối chiếu utils.lssc_binary thật).
  1. refit_only   — CÙNG M_matrix, nhưng ngưỡng tính lại (quantile) trên
                     select set của DemogPairs, KHÔNG decorrelation.
                     Đây là nhóm đối chứng: nếu riêng việc refit ngưỡng
                     trên DemogPairs (khác population gốc dùng để tính
                     intervals.npy) đã thay đổi kết quả, phải tách được
                     phần đó ra khỏi hiệu ứng thật của decorrelation.
  2. decorrelated — trừ mean (select set) + bỏ N_DROP_PC thành phần PCA
                     đầu (fit trên select set) + threshold refit trên
                     KHÔNG GIAN ĐÃ BIẾN ĐỔI (select set).
Whitening (chuẩn hóa phương sai từng hướng) mặc định TẮT — bật qua
ENABLE_WHITENING nếu variant 2 cho tín hiệu tốt, để test riêng hiệu ứng
whitening cộng thêm (advisor's simulation không thấy lợi ích whitening
với dim độc lập, nhưng tự nhận AdaFace thật có tương quan — cần test
riêng, không gộp chung 1 lần với PC-drop để khỏi nhầm hiệu ứng nào gây
ra thay đổi).

BẮT BUỘC: mọi threshold refit đều fit trên SELECT set, đo trên VALIDATE
set (identity-disjoint, như 07) — tránh overfitting.

Cách chạy:
    python scripts/09_decorrelation_impostor_margin.py
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

RNG_SEED = 42
SELECT_FRACTION = 0.5
N_DROP_PC = 2  # số thành phần PCA toàn cục bị bỏ (advisor đề xuất 1-2)
ENABLE_WHITENING = False  # bật riêng ở lần chạy sau nếu variant 2 có tín hiệu

M_matrix = np.load(M_MATRIX_NPY)
production_intervals = np.load(INTERVALS_NPY)
N_THR = np.asarray(production_intervals).reshape(-1).size
D_DIMS = M_matrix.shape[1]
print(
    f"[load] M_matrix={M_matrix.shape}  N_THR={N_THR}  D_DIMS={D_DIMS}  "
    f"N_DROP_PC={N_DROP_PC}  ENABLE_WHITENING={ENABLE_WHITENING}"
)


def project(embedding: np.ndarray) -> np.ndarray:
    return np.dot(np.asarray(embedding, dtype=np.float64), M_matrix)


def thermometer_bits(v: np.ndarray, thr_sorted: np.ndarray) -> np.ndarray:
    """Tái hiện đúng logic thermometer-code gốc: 1 bộ ngưỡng DÙNG CHUNG
    cho mọi dimension (khớp cách intervals.npy gốc là mảng 1 chiều)."""
    rev_thr = thr_sorted[::-1]
    cmp = v[:, None] >= rev_thr[None, :]
    return cmp.astype(np.uint8).reshape(-1)


def real_binarize_production(embedding: np.ndarray) -> np.ndarray:
    projected = project(embedding)[None, :]
    return (
        utils.lssc_binary(projected, interval=production_intervals)
        .flatten()
        .astype(np.uint8)
    )


def self_check_reimplementation(sample_embs: list):
    """Xác nhận thermometer_bits() tự viết cho kết quả GIỐNG HỆT
    utils.lssc_binary thật khi dùng đúng M_matrix + intervals gốc —
    trước khi tin bất kỳ variant refit/decorrelated nào dùng chung hàm
    thermometer_bits() này."""
    thr_sorted = np.sort(np.asarray(production_intervals).reshape(-1))
    for emb in sample_embs[:8]:
        v = project(emb)
        cand = thermometer_bits(v, thr_sorted)
        real = real_binarize_production(emb)
        if not np.array_equal(cand, real):
            raise SystemExit(
                "[self-check] FAIL — thermometer_bits() tự viết không khớp "
                "utils.lssc_binary thật. DỪNG, không tin variant refit_only/"
                "decorrelated cho tới khi sửa xong."
            )
    print(
        f"[self-check] PASS — thermometer_bits() khớp production trên "
        f"{min(8, len(sample_embs))} mẫu."
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


def split_identities(identity_to_fold, rng):
    by_fold = defaultdict(list)
    for identity, fold in identity_to_fold.items():
        by_fold[fold].append(identity)
    select_ids, validate_ids = set(), set()
    for fold, identities in by_fold.items():
        identities = sorted(identities)
        idx = rng.permutation(len(identities))
        n_select = int(len(identities) * SELECT_FRACTION)
        for k, i in enumerate(idx):
            (select_ids if k < n_select else validate_ids).add(identities[i])
    return select_ids, validate_ids


def load_and_split_pairs(select_ids, validate_ids):
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

    with open(IMPOSTOR_SAMEFOLD_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fold = row["fold"]
            id1, id2 = row["identity_1"], row["identity_2"]
            pair = (row["cache_filename_1"], row["cache_filename_2"])
            if id1 in select_ids and id2 in select_ids:
                select_pairs[fold]["impostor"].append(pair)
            elif id1 in validate_ids and id2 in validate_ids:
                validate_pairs[fold]["impostor"].append(pair)

    return select_pairs, validate_pairs


def all_select_projected(select_pairs):
    """Gộp toàn bộ ảnh xuất hiện trong select_pairs (mọi fold) thành 1
    ma trận (N, D_DIMS) đã project qua M_matrix — dùng để fit mean/PCA/
    ngưỡng. Dedup theo cache_filename để không đếm lặp ảnh xuất hiện ở
    nhiều pair."""
    seen = set()
    vecs = []
    for fold_data in select_pairs.values():
        for pairs in (fold_data["genuine"], fold_data["impostor"]):
            for f1, f2 in pairs:
                for f in (f1, f2):
                    if f not in seen:
                        seen.add(f)
                        vecs.append(project(load_embedding(f)))
    return np.stack(vecs, axis=0)


def fit_pca_drop(projected_select: np.ndarray, n_drop: int):
    mean = projected_select.mean(axis=0)
    centered = projected_select - mean
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    drop_components = Vt[:n_drop]  # (n_drop, D) — hàng trực chuẩn
    explained_ratio = (S[:n_drop] ** 2).sum() / (S**2).sum()
    print(
        f"  [PCA] {n_drop} PC đầu giải thích {explained_ratio:.1%} tổng phương sai "
        f"(tính trên select set, {projected_select.shape[0]} ảnh)."
    )
    remaining_std = None
    if ENABLE_WHITENING:
        transformed = centered - (centered @ drop_components.T) @ drop_components
        remaining_std = transformed.std(axis=0)
        remaining_std = np.where(remaining_std > 1e-8, remaining_std, 1.0)
    return mean, drop_components, remaining_std


def apply_pca_drop(
    projected: np.ndarray, mean, drop_components, remaining_std
) -> np.ndarray:
    centered = projected - mean
    removed = (centered @ drop_components.T) @ drop_components
    transformed = centered - removed
    if remaining_std is not None:
        transformed = transformed / remaining_std
    return transformed


def fit_quantile_thresholds(values_2d: np.ndarray, n_thr: int) -> np.ndarray:
    """values_2d: (N, D) — pool toàn bộ N*D giá trị lại rồi lấy quantile,
    khớp cách intervals.npy gốc là 1 bộ ngưỡng dùng chung mọi dimension."""
    pooled = values_2d.reshape(-1)
    qs = np.linspace(0, 1, n_thr + 2)[1:-1]
    return np.sort(np.quantile(pooled, qs))


def per_dimension_ber(pairs: list, bits_fn) -> np.ndarray:
    if not pairs:
        return np.full(D_DIMS, np.nan)
    flips_sum = np.zeros(D_DIMS * N_THR, dtype=np.float64)
    for f1, f2 in pairs:
        b1 = bits_fn(load_embedding(f1))
        b2 = bits_fn(load_embedding(f2))
        flips_sum += (b1 != b2).astype(np.float64)
    return (flips_sum / len(pairs)).reshape(D_DIMS, N_THR).mean(axis=1)


def evaluate_variant(label: str, bits_fn, validate_pairs: dict) -> dict:
    print(f"\n--- Variant '{label}' — đo trên validate set ---")
    fold_results = {}
    for fold, pairs in validate_pairs.items():
        g, i = pairs["genuine"], pairs["impostor"]
        if len(g) < 10 or len(i) < 10:
            continue
        gen_ber = float(np.nanmean(per_dimension_ber(g, bits_fn)))
        imp_ber = float(np.nanmean(per_dimension_ber(i, bits_fn)))
        sep = imp_ber - gen_ber
        fold_results[fold] = {
            "genuine_ber": gen_ber,
            "impostor_ber": imp_ber,
            "separation": sep,
        }
        print(
            f"  {fold:16s}: genuine_ber={gen_ber:.4f}  impostor_ber={imp_ber:.4f}  "
            f"separation={sep:+.4f}"
        )
    if fold_results:
        avg_sep = float(np.mean([r["separation"] for r in fold_results.values()]))
        worst_sep = min(r["separation"] for r in fold_results.values())
        avg_gen = float(np.mean([r["genuine_ber"] for r in fold_results.values()]))
        print(
            f"  -> trung bình separation={avg_sep:+.4f}  worst-case={worst_sep:+.4f}  "
            f"trung bình genuine_ber={avg_gen:.4f}"
        )
    return fold_results


def main():
    rng = np.random.default_rng(RNG_SEED)

    identity_to_fold = load_identity_fold_map()
    select_ids, validate_ids = split_identities(identity_to_fold, rng)
    print(f"Identity: select={len(select_ids)}  validate={len(validate_ids)}")

    select_pairs, validate_pairs = load_and_split_pairs(select_ids, validate_ids)

    # self-check bắt buộc trước khi tin bất kỳ variant nào
    probe_files = []
    for fold_data in select_pairs.values():
        for f1, f2 in fold_data["genuine"][:4]:
            probe_files.extend([f1, f2])
        if len(probe_files) >= 8:
            break
    self_check_reimplementation([load_embedding(f) for f in probe_files[:8]])

    projected_select = all_select_projected(select_pairs)
    print(f"\nSelect set: {projected_select.shape[0]} ảnh duy nhất (đã dedup).")

    # ---- Variant 0: production ----
    bits_production = real_binarize_production
    results_production = evaluate_variant(
        "0_production", bits_production, validate_pairs
    )

    # ---- Variant 1: refit_only (M_matrix gốc, ngưỡng refit trên select set) ----
    refit_thr = fit_quantile_thresholds(projected_select, N_THR)

    def bits_refit_only(embedding):
        return thermometer_bits(project(embedding), refit_thr)

    results_refit = evaluate_variant("1_refit_only", bits_refit_only, validate_pairs)

    # ---- Variant 2: decorrelated (mean-sub + drop top PC [+ whitening]) ----
    mean, drop_components, remaining_std = fit_pca_drop(projected_select, N_DROP_PC)
    projected_select_decorr = apply_pca_drop(
        projected_select, mean, drop_components, remaining_std
    )
    decorr_thr = fit_quantile_thresholds(projected_select_decorr, N_THR)

    def bits_decorrelated(embedding):
        v = apply_pca_drop(
            project(embedding)[None, :], mean, drop_components, remaining_std
        )[0]
        return thermometer_bits(v, decorr_thr)

    results_decorr = evaluate_variant(
        "2_decorrelated", bits_decorrelated, validate_pairs
    )

    # ---- So sánh tổng hợp ----
    print("\n" + "=" * 70)
    print("SO SÁNH 3 VARIANT (đo trên validate set, chưa từng dùng để fit)")
    print("=" * 70)
    common_folds = set(results_production) & set(results_refit) & set(results_decorr)
    header = f"{'fold':16s}  {'prod_sep':>10s}  {'refit_sep':>10s}  {'decorr_sep':>10s}  {'decorr_gen_ber':>15s}"
    print(header)
    for fold in sorted(common_folds):
        p = results_production[fold]["separation"]
        r = results_refit[fold]["separation"]
        d = results_decorr[fold]["separation"]
        dg = results_decorr[fold]["genuine_ber"]
        print(f"{fold:16s}  {p:+10.4f}  {r:+10.4f}  {d:+10.4f}  {dg:15.4f}")

    if common_folds:
        avg_p = np.mean([results_production[f]["separation"] for f in common_folds])
        avg_r = np.mean([results_refit[f]["separation"] for f in common_folds])
        avg_d = np.mean([results_decorr[f]["separation"] for f in common_folds])
        worst_p = min(results_production[f]["separation"] for f in common_folds)
        worst_r = min(results_refit[f]["separation"] for f in common_folds)
        worst_d = min(results_decorr[f]["separation"] for f in common_folds)
        print(
            f"\nTrung bình separation:  production={avg_p:+.4f}  refit_only={avg_r:+.4f}  "
            f"decorrelated={avg_d:+.4f}"
        )
        print(
            f"Worst-case separation:  production={worst_p:+.4f}  refit_only={worst_r:+.4f}  "
            f"decorrelated={worst_d:+.4f}"
        )
        print(
            "\nDIỄN GIẢI:\n"
            "  - So decorrelated vs refit_only (KHÔNG so trực tiếp với production)\n"
            "    mới cô lập đúng hiệu ứng của decorrelation — vì refit_only đã\n"
            "    gánh hết phần thay đổi do 'ngưỡng tính lại trên DemogPairs'.\n"
            "  - decorrelated > refit_only rõ rệt (cả avg và worst-case) -> bỏ\n"
            "    top-PC thật sự giúp tăng margin, đáng thử whitening/rotation tiếp.\n"
            "  - decorrelated ~ refit_only -> PCA-drop không giúp gì thêm ngoài\n"
            "    việc refit ngưỡng, cân nhắc dừng hướng này.\n"
            "  - decorrelated < refit_only -> bỏ PC đang phá hỏng tín hiệu danh\n"
            "    tính (PC đó có thể mang thông tin identity, không chỉ 'mean face')."
        )

    np.savez(
        os.path.join(OUT_DIR, "decorrelation_result.npz"),
        results_production=np.array(list(results_production.items()), dtype=object),
        results_refit=np.array(list(results_refit.items()), dtype=object),
        results_decorr=np.array(list(results_decorr.items()), dtype=object),
        n_drop_pc=N_DROP_PC,
        enable_whitening=ENABLE_WHITENING,
    )
    print(f"\n[out] -> {os.path.join(OUT_DIR, 'decorrelation_result.npz')}")


if __name__ == "__main__":
    main()
