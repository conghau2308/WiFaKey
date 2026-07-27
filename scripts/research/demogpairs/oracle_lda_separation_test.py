"""
oracle_lda_separation_test.py

MỤC ĐÍCH
--------
Trả lời 1 câu hỏi duy nhất: "Nếu dùng một phép chiếu tuyến tính CÓ GIÁM SÁT
(Fisher-LDA-style, tối ưu trực tiếp tỷ lệ between/within-identity scatter)
thay vì chọn subset 277/512 chiều theo vị trí (baseline) hay theo worst-case
separation (07_worstcase_dimension_selection.py), thì impostor separation có
cải thiện đáng kể không?"

Đây là bài test ORACLE: đo trên EMBEDDING LIÊN TỤC (trước binarize/mask/LDPC),
không tốn công enroll lại hệ thống thật hay retrain decoder. Nếu oracle này
cũng chỉ nhỉnh hơn baseline vài % (giống mức +1.3%/~0% đã thấy ở nhánh
dimension-selection/decorrelation), thì gần như chắc chắn học M_matrix thật
bằng LDA/metric-learning cũng không đáng đầu tư.

DỮ LIỆU ĐẦU VÀO (khớp đúng output của 03a/03b_*_demogpairs.py bạn đã gửi)
--------------------------------------------------------------------------
  datasets/processed/demogpairs/
    embeddings_cache/<cache_filename>.npy      (mỗi ảnh 1 vector 512-d)
    image_metadata.csv                          (identity, fold, cache_filename, status, ...)
    pairs/audit_genuine.csv                     (identity, cache_filename_1, cache_filename_2, fold)
    pairs/audit_impostor_samefold.csv           (fold, identity_1, cache_filename_1, identity_2, cache_filename_2)
    pairs/audit_impostor_crossfold.csv          (fold_1, fold_2, identity_1, cache_filename_1, identity_2, cache_filename_2)

SELECT/VALIDATE SPLIT (bạn xác nhận chưa có sẵn — script này tự tạo)
----------------------------------------------------------------------
Vì 03b không sinh split select/validate, script này tự chia identity thành
2 tập theo tỷ lệ SELECT_RATIO, STRATIFY THEO FOLD (để mỗi fold đều có mặt cân
đối ở cả 2 tập, không lệch demographic), seed cố định để tái lập được.
  - select_ids   : dùng để fit phép chiếu Fisher-LDA (Sb, Sw)
  - validate_ids : dùng để đo separation/AUC — CHỈ dùng các cặp mà CẢ HAI
                   identity đều thuộc validate_ids (tránh rò rỉ thông tin
                   từ select sang validate qua 1 identity xuất hiện ở cả 2 phía)

CÁCH BÁO CÁO SEPARATION (bạn để tôi tự quyết — lý do chọn cách này)
----------------------------------------------------------------------
Báo cáo CẢ 3 mức, không chỉ 1 con số:
  - same_fold   : genuine vs impostor cùng fold nhân khẩu học
  - cross_fold  : genuine vs impostor khác fold
  - combined    : genuine vs (same_fold + cross_fold gộp) — dùng làm quyết
                  định chính "có nên đầu tư LDA/metric-learning thật không"
Lý do giữ cả 3: nếu chỉ nhìn combined, có thể che mất trường hợp oracle giúp
same-fold nhưng lại làm cross-fold tệ hơn (hoặc ngược lại) — điều này quan
trọng vì bạn đang đồng thời quan tâm cả FAR tổng lẫn công bằng nhân khẩu học.

Chạy: python oracle_lda_separation_test.py
"""

from __future__ import annotations
import os
import csv
import random
import numpy as np
import sys
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, PROJECT_ROOT)

DATASET_NAME = "demogpairs"
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed", DATASET_NAME)
CACHE_DIR = os.path.join(DATA_DIR, "embeddings_cache")
IMAGE_METADATA_CSV = os.path.join(DATA_DIR, "image_metadata.csv")
GENUINE_CSV = os.path.join(DATA_DIR, "pairs", "audit_genuine.csv")
IMPOSTOR_SAME_CSV = os.path.join(DATA_DIR, "pairs", "audit_impostor_samefold.csv")
IMPOSTOR_CROSS_CSV = os.path.join(DATA_DIR, "pairs", "audit_impostor_crossfold.csv")

EMBEDDING_DIM = 512
N_KEEP = 277  # số chiều giữ lại, khớp baseline hệ thống
# Quét nhiều mức regularization Sw thay vì 1 giá trị cố định — để phân biệt
# 2 giả thuyết: (a) oracle tệ hơn baseline vì bản thân embedding space không
# có transform tuyến tính nào tốt hơn (kết luận không đổi dù regularize thế
# nào), hay (b) chỉ vì Fisher-LDA đang overfit do cỡ mẫu nhỏ (~3-4 ảnh/identity
# so với D=512) và sẽ cải thiện dần khi regularize mạnh hơn.
REG_EPS_SWEEP = [1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
SELECT_RATIO = 0.7  # tỷ lệ identity đưa vào select (còn lại vào validate)
SPLIT_SEED = 42  # khớp RANDOM_SEED trong 03b để nhất quán quy ước của bạn
WORST_CASE_DIMS: Optional[np.ndarray] = (
    None  # điền index array từ 07 nếu muốn so sánh thêm
)


# =====================================================================
# LOAD DỮ LIỆU THẬT
# =====================================================================
def load_identity_fold_map() -> dict[str, str]:
    """Đọc image_metadata.csv -> {identity: fold}. Cảnh báo nếu 1 identity
    có >1 fold khác nhau (không nên xảy ra, nhưng kiểm tra cho chắc)."""
    mapping: dict[str, str] = {}
    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["cache_filename"]:
                continue
            ident, fold = row["identity"], row["fold"]
            if ident in mapping and mapping[ident] != fold:
                print(
                    f"*** CẢNH BÁO: identity '{ident}' có >1 fold ({mapping[ident]} vs {fold}) ***"
                )
            mapping[ident] = fold
    return mapping


def build_select_validate_split(
    identity_to_fold: dict[str, str], select_ratio: float, seed: int
):
    """Stratify theo fold: mỗi fold tách riêng theo tỷ lệ, rồi gộp lại."""
    rng = random.Random(seed)
    ids_by_fold: dict[str, list[str]] = defaultdict(list)
    for ident, fold in identity_to_fold.items():
        ids_by_fold[fold].append(ident)

    select_ids, validate_ids = set(), set()
    for fold, ids in sorted(ids_by_fold.items()):
        ids_sorted = sorted(ids)  # thứ tự xác định trước khi shuffle -> tái lập được
        rng.shuffle(ids_sorted)
        n_select = int(round(len(ids_sorted) * select_ratio))
        select_ids.update(ids_sorted[:n_select])
        validate_ids.update(ids_sorted[n_select:])
        print(
            f"  fold={fold:15s} n_identity={len(ids_sorted):4d} "
            f"-> select={n_select:4d} validate={len(ids_sorted) - n_select:4d}"
        )

    return select_ids, validate_ids


def load_identity_to_cachefiles() -> dict[str, list[str]]:
    """identity -> list các cache_filename (ảnh) thuộc identity đó, chỉ tính
    ảnh trích xuất thành công."""
    out: dict[str, list[str]] = defaultdict(list)
    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["cache_filename"]:
                out[row["identity"]].append(row["cache_filename"])
    return out


class EmbeddingLoader:
    """Load .npy theo cache_filename, có cache trong RAM để không đọc đĩa
    lặp lại (1 ảnh có thể xuất hiện trong nhiều pairs)."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._mem: dict[str, np.ndarray] = {}

    def get(self, cache_filename: str) -> np.ndarray:
        if cache_filename not in self._mem:
            path = os.path.join(self.cache_dir, cache_filename)
            vec = np.load(path)
            assert (
                vec.shape[-1] == EMBEDDING_DIM
            ), f"Embedding {cache_filename} có shape {vec.shape}, kỳ vọng chiều cuối = {EMBEDDING_DIM}"
            self._mem[cache_filename] = vec.astype(np.float64).reshape(-1)
        return self._mem[cache_filename]


def load_embeddings_by_identity(
    identities: set[str],
    identity_to_cachefiles: dict[str, list[str]],
    loader: EmbeddingLoader,
) -> dict[str, np.ndarray]:
    """Dùng để FIT (Sw/Sb) — chỉ cần identity thuộc select_ids, load hết ảnh của
    identity đó thành 1 array (n_samples, 512)."""
    out = {}
    for ident in identities:
        files = identity_to_cachefiles.get(ident, [])
        if len(files) < 1:
            continue
        vecs = np.stack([loader.get(f) for f in files], axis=0)
        out[ident] = vecs
    return out


@dataclass
class Pair:
    cache_a: str
    cache_b: str


def load_genuine_pairs(validate_ids: set[str]) -> list[Pair]:
    pairs = []
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["identity"] in validate_ids:
                pairs.append(Pair(row["cache_filename_1"], row["cache_filename_2"]))
    return pairs


def load_impostor_pairs_same_fold(validate_ids: set[str]) -> list[Pair]:
    pairs = []
    with open(IMPOSTOR_SAME_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["identity_1"] in validate_ids and row["identity_2"] in validate_ids:
                pairs.append(Pair(row["cache_filename_1"], row["cache_filename_2"]))
    return pairs


def load_impostor_pairs_cross_fold(validate_ids: set[str]) -> list[Pair]:
    pairs = []
    with open(IMPOSTOR_CROSS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["identity_1"] in validate_ids and row["identity_2"] in validate_ids:
                pairs.append(Pair(row["cache_filename_1"], row["cache_filename_2"]))
    return pairs


# =====================================================================
# CORE — FIT + PROJECT + ĐO
# =====================================================================
def fit_fisher_projection(
    embeddings_by_id: dict[str, np.ndarray], reg_eps: float
) -> np.ndarray:
    """Giải generalized eigenvalue Sb v = lambda Sw v (Fisher-LDA-style),
    trả về ma trận chiếu (512, 512) cột sắp theo eigenvalue giảm dần."""
    all_ids = list(embeddings_by_id.keys())
    all_data = np.concatenate([embeddings_by_id[i] for i in all_ids], axis=0)
    global_mean = all_data.mean(axis=0)

    D = all_data.shape[1]
    Sw = np.zeros((D, D))
    Sb = np.zeros((D, D))

    for ident in all_ids:
        X = embeddings_by_id[ident]
        class_mean = X.mean(axis=0)
        Xc = X - class_mean
        Sw += Xc.T @ Xc
        n = X.shape[0]
        diff = (class_mean - global_mean).reshape(-1, 1)
        Sb += n * (diff @ diff.T)

    Sw_reg = Sw + reg_eps * np.trace(Sw) / D * np.eye(D)

    eigvals, eigvecs = np.linalg.eig(np.linalg.solve(Sw_reg, Sb))
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    order = np.argsort(eigvals)[::-1]
    print(f"  top-5 eigenvalues (Fisher ratio): {eigvals[order][:5]}")
    return eigvecs[:, order]


def project_vec(v: np.ndarray, dims_or_matrix) -> np.ndarray:
    if dims_or_matrix.ndim == 1:
        return v[dims_or_matrix]
    return dims_or_matrix.T @ v


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return 1.0 - float(an @ bn)


def pair_distances(
    pairs: list[Pair], dims_or_matrix, loader: EmbeddingLoader
) -> np.ndarray:
    dists = []
    for p in pairs:
        va = project_vec(loader.get(p.cache_a), dims_or_matrix)
        vb = project_vec(loader.get(p.cache_b), dims_or_matrix)
        dists.append(cosine_dist(va, vb))
    return np.array(dists)


def compute_stats(genuine_dists: np.ndarray, impostor_dists: np.ndarray) -> dict:
    pooled_std = np.sqrt((genuine_dists.var() + impostor_dists.var()) / 2) + 1e-12
    separation = (impostor_dists.mean() - genuine_dists.mean()) / pooled_std

    labels = np.concatenate(
        [np.zeros_like(genuine_dists), np.ones_like(impostor_dists)]
    )
    scores = np.concatenate([genuine_dists, impostor_dists])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return dict(
        mean_genuine=genuine_dists.mean(),
        mean_impostor=impostor_dists.mean(),
        separation=separation,
        auc=auc,
        n_genuine=len(genuine_dists),
        n_impostor=len(impostor_dists),
    )


def evaluate_variant(
    name: str,
    dims_or_matrix,
    loader: EmbeddingLoader,
    genuine_pairs,
    impostor_same_pairs,
    impostor_cross_pairs,
) -> list[dict]:
    n_dims = (
        dims_or_matrix.shape[1] if dims_or_matrix.ndim == 2 else len(dims_or_matrix)
    )
    gen_d = pair_distances(genuine_pairs, dims_or_matrix, loader)
    same_d = pair_distances(impostor_same_pairs, dims_or_matrix, loader)
    cross_d = pair_distances(impostor_cross_pairs, dims_or_matrix, loader)
    combined_d = np.concatenate([same_d, cross_d])

    rows = []
    for category, imp_d in [
        ("same_fold", same_d),
        ("cross_fold", cross_d),
        ("combined", combined_d),
    ]:
        stats = compute_stats(gen_d, imp_d)
        rows.append(dict(variant=name, n_dims=n_dims, category=category, **stats))
    return rows


def main():
    print("[1/6] Đọc identity -> fold từ image_metadata.csv...")
    identity_to_fold = load_identity_fold_map()
    print(f"  Tổng số identity: {len(identity_to_fold)}")

    print("\n[2/6] Chia select/validate theo identity, stratify theo fold...")
    select_ids, validate_ids = build_select_validate_split(
        identity_to_fold, SELECT_RATIO, SPLIT_SEED
    )
    print(
        f"  Tổng: select={len(select_ids)} identity | validate={len(validate_ids)} identity"
    )

    print("\n[3/6] Load embeddings tập select để fit Fisher-LDA...")
    identity_to_cachefiles = load_identity_to_cachefiles()
    loader = EmbeddingLoader(CACHE_DIR)
    select_embeddings = load_embeddings_by_identity(
        select_ids, identity_to_cachefiles, loader
    )
    print(f"  Đã load {len(select_embeddings)} identity cho tập select.")

    baseline_dims = np.arange(N_KEEP)

    print("\n[4/6] Load pairs tập validate và đo separation/AUC...")
    genuine_pairs = load_genuine_pairs(validate_ids)
    impostor_same_pairs = load_impostor_pairs_same_fold(validate_ids)
    impostor_cross_pairs = load_impostor_pairs_cross_fold(validate_ids)
    print(
        f"  genuine={len(genuine_pairs)} | impostor_same_fold={len(impostor_same_pairs)} "
        f"| impostor_cross_fold={len(impostor_cross_pairs)} (đã lọc theo validate_ids)"
    )

    all_rows = []
    all_rows += evaluate_variant(
        "baseline (positional 0..276)",
        baseline_dims,
        loader,
        genuine_pairs,
        impostor_same_pairs,
        impostor_cross_pairs,
    )
    if WORST_CASE_DIMS is not None:
        all_rows += evaluate_variant(
            "worst_case (từ 07)",
            WORST_CASE_DIMS,
            loader,
            genuine_pairs,
            impostor_same_pairs,
            impostor_cross_pairs,
        )

    print(
        "\n[5/6] Quét REG_EPS: fit lại Fisher-LDA cho từng mức regularization "
        "(chỉ dùng select, validate KHÔNG được đụng ở bước fit)..."
    )
    for reg_eps in REG_EPS_SWEEP:
        print(f"  --- REG_EPS={reg_eps} ---")
        proj_matrix_full = fit_fisher_projection(select_embeddings, reg_eps)
        oracle_lda_matrix = proj_matrix_full[:, :N_KEEP]
        variant_name = f"oracle_lda (reg_eps={reg_eps})"
        all_rows += evaluate_variant(
            variant_name,
            oracle_lda_matrix,
            loader,
            genuine_pairs,
            impostor_same_pairs,
            impostor_cross_pairs,
        )

    print(
        f"\n{'Variant':32s} {'category':11s} {'n_dims':>7s} {'n_gen':>7s} {'n_imp':>7s} "
        f"{'gen_dist':>9s} {'imp_dist':>9s} {'separation':>11s} {'AUC':>7s}"
    )
    for r in all_rows:
        print(
            f"{r['variant']:32s} {r['category']:11s} {r['n_dims']:7d} {r['n_genuine']:7d} "
            f"{r['n_impostor']:7d} {r['mean_genuine']:9.4f} {r['mean_impostor']:9.4f} "
            f"{r['separation']:11.4f} {r['auc']:7.4f}"
        )

    print("\nCÁCH ĐỌC KẾT QUẢ:")
    print("- Nhìn dòng 'combined' của từng oracle_lda(reg_eps=...) so với baseline.")
    print("- Nếu separation TĂNG DẦN theo reg_eps rồi VƯỢT hẳn baseline ở 1 mức nào")
    print("  đó -> oracle trước đây tệ hơn baseline chỉ vì regularize chưa đủ mạnh")
    print("  (Fisher-LDA overfit do cỡ mẫu nhỏ) -> vẫn còn dư địa thật, đáng cân")
    print("  nhắc đầu tư LDA/metric-learning cho M_matrix.")
    print("- Nếu separation tăng nhẹ theo reg_eps nhưng KHÔNG BAO GIỜ vượt baseline")
    print("  (dù reg_eps đã rất lớn, gần như ép Sw về identity matrix) -> củng cố")
    print("  kết luận: trần margin nằm ở chính embedding space, KHÔNG đáng đầu tư")
    print("  học M_matrix thật -> đóng hẳn nhánh LDA/metric-learning.")
    print("- Nếu same_fold và cross_fold cải thiện KHÁC NHAU nhiều ở mức reg_eps")
    print("  tốt nhất -> đáng ghi chú riêng, vì ảnh hưởng cả FAR tổng lẫn công")
    print("  bằng nhân khẩu học.")


if __name__ == "__main__":
    main()
