"""
build_oracle_lda_matrix.py

MỤC ĐÍCH
--------
Từ kết quả sweep reg_eps đã có (plateau ổn định ~+2.5-2.8% separation quanh
reg_eps=1.0-2.0), chọn 1 mức reg_eps cụ thể, fit lại Fisher-LDA CHỈ trên tập
select, rồi dựng ra 1 M_matrix MỚI (512x512, TRỰC GIAO — giống cấu trúc
M_matrix gốc bạn đang dùng) để test qua pipeline thật (enroll/verify) mà
KHÔNG cần tính lại binarization_intervals.npy.

TẠI SAO PHẢI TRỰC GIAO HÓA (không dùng thẳng eigenvector Fisher-LDA)
----------------------------------------------------------------------
M_matrix gốc của bạn là ma trận TRỰC GIAO HOÀN HẢO (đã kiểm tra: M@M.T = I,
mọi hàng/cột norm=1). Đây là lý do 1 bộ ngưỡng nhị phân hóa DÙNG CHUNG cho
cả 512 chiều vẫn cho ra bit ~equal-probable ở mọi chiều: phép quay trực giao
giữ nguyên phân phối biên tương tự nhau giữa các chiều.

Eigenvector của bài toán generalized eigenvalue (Fisher-LDA) thì KHÔNG trực
giao theo nghĩa Euclid (chỉ trực giao theo Sw) và có scale/variance rất khác
nhau giữa các chiều (chiều ranking cao có eigenvalue/variance lớn hơn hẳn
chiều ranking thấp). Nếu dùng thẳng, bộ ngưỡng chung hiện tại sẽ không còn
equal-probable cho các chiều mới -> kết quả FRR/FAR thật đo được sẽ bị nhiễu
bởi lệch ngưỡng, không phản ánh đúng "dimension mới có tốt hơn không".

Cách xử lý: chỉ giữ LẠI HƯỚNG (subspace) của 277 eigenvector top, trực giao
hóa chúng bằng QR để mỗi cột unit-norm + trực giao với nhau, rồi mở rộng
thêm 235 cột trực giao bù (span phần không gian còn lại) để đủ 512x512.
235 cột bù này KHÔNG ảnh hưởng gì tới kết quả, vì hệ thống cắt bit ngay tại
b_selected = b_masked[:832] -- chỉ 277 chiều đầu (831-vài bit đầu) được dùng.

Output: 1 file .npy MỚI, KHÔNG ghi đè M_matrix.npy gốc.

Chạy: python build_oracle_lda_matrix.py
"""

from __future__ import annotations
import os
import csv
import random
import numpy as np
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DATASET_NAME = "demogpairs"
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed", DATASET_NAME)
CACHE_DIR = os.path.join(DATA_DIR, "embeddings_cache")
IMAGE_METADATA_CSV = os.path.join(DATA_DIR, "image_metadata.csv")

EMBEDDING_DIM = 512
N_KEEP = 277
SELECT_RATIO = 0.7
SPLIT_SEED = (
    42  # PHẢI khớp với oracle_lda_separation_test.py để cùng 1 tập select/validate
)
REG_EPS_CHOSEN = (
    2.0  # điểm gần đỉnh plateau trong sweep đã chạy (dải ổn định 0.5-100 đều >baseline)
)

# Nơi lưu M_matrix mới — KHÔNG phải wifakey_module/data/M_matrix.npy gốc
OUTPUT_MATRIX_PATH = os.path.join(
    PROJECT_ROOT,
    "research",
    "modulation",
    "dimension_selection",
    f"M_matrix_oracle_lda_regeps{REG_EPS_CHOSEN}.npy",
)


# =====================================================================
# LOAD DATA (giống hệt oracle_lda_separation_test.py để dùng đúng 1 tập select)
# =====================================================================
def load_identity_fold_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["cache_filename"]:
                mapping[row["identity"]] = row["fold"]
    return mapping


def build_select_validate_split(
    identity_to_fold: dict[str, str], select_ratio: float, seed: int
):
    rng = random.Random(seed)
    ids_by_fold: dict[str, list[str]] = defaultdict(list)
    for ident, fold in identity_to_fold.items():
        ids_by_fold[fold].append(ident)

    select_ids = set()
    for fold, ids in sorted(ids_by_fold.items()):
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        n_select = int(round(len(ids_sorted) * select_ratio))
        select_ids.update(ids_sorted[:n_select])
    return select_ids


def load_identity_to_cachefiles() -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["cache_filename"]:
                out[row["identity"]].append(row["cache_filename"])
    return out


def load_embeddings_by_identity(
    identities: set[str], identity_to_cachefiles, cache_dir: str
) -> dict[str, np.ndarray]:
    out = {}
    for ident in identities:
        files = identity_to_cachefiles.get(ident, [])
        if not files:
            continue
        vecs = [
            np.load(os.path.join(cache_dir, f)).astype(np.float64).reshape(-1)
            for f in files
        ]
        out[ident] = np.stack(vecs, axis=0)
    return out


def fit_fisher_projection(
    embeddings_by_id: dict[str, np.ndarray], reg_eps: float
) -> np.ndarray:
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
    eigvals, eigvecs = eigvals.real, eigvecs.real
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def orthonormalize_and_extend(top_k_vectors: np.ndarray, D: int) -> np.ndarray:
    """
    top_k_vectors: (D, K) — K hướng discriminant đã chọn (chưa trực giao chuẩn).
    Trả về (D, D) TRỰC GIAO HOÀN CHỈNH: K cột đầu span đúng subspace của
    top_k_vectors (đã trực chuẩn hoá), D-K cột sau trực giao bù.
    """
    K = top_k_vectors.shape[1]

    # Bước 1: trực chuẩn hoá K cột đầu bằng QR (giữ đúng subspace ban đầu)
    Q1, _ = np.linalg.qr(top_k_vectors)  # (D, K), cột trực chuẩn

    # Bước 2: dựng phần bù trực giao — chiếu 1 ma trận ngẫu nhiên ra khỏi
    # span(Q1), rồi QR phần dư để lấy cơ sở trực chuẩn cho phần bù.
    rng = np.random.default_rng(SPLIT_SEED)  # seed cố định -> tái lập được
    R = rng.standard_normal((D, D - K))
    R_orth = R - Q1 @ (Q1.T @ R)
    Q2, _ = np.linalg.qr(R_orth)  # (D, D-K), trực chuẩn + trực giao với Q1

    M_new = np.concatenate([Q1, Q2], axis=1)  # (D, D)

    # Self-check bắt buộc trước khi tin dùng
    ortho_err = np.max(np.abs(M_new @ M_new.T - np.eye(D)))
    subspace_err = np.max(np.abs(Q1 @ Q1.T @ top_k_vectors - top_k_vectors))
    print(f"  self-check: max|M@M.T - I| = {ortho_err:.2e} (kỳ vọng ~1e-10 trở xuống)")
    print(
        f"  self-check: sai số subspace top-{K} sau QR = {subspace_err:.2e} (kỳ vọng ~0, QR không đổi subspace)"
    )
    assert (
        ortho_err < 1e-8
    ), "M_new KHÔNG trực giao đủ chính xác — không nên dùng, kiểm tra lại."

    return M_new


def main():
    print(
        "[1/4] Đọc identity -> fold, dựng lại ĐÚNG tập select đã dùng ở oracle_lda_separation_test.py..."
    )
    identity_to_fold = load_identity_fold_map()
    select_ids = build_select_validate_split(identity_to_fold, SELECT_RATIO, SPLIT_SEED)
    print(f"  select: {len(select_ids)} identity")

    print("\n[2/4] Load embeddings tập select...")
    identity_to_cachefiles = load_identity_to_cachefiles()
    select_embeddings = load_embeddings_by_identity(
        select_ids, identity_to_cachefiles, CACHE_DIR
    )
    print(f"  Đã load {len(select_embeddings)} identity.")

    print(f"\n[3/4] Fit Fisher-LDA với REG_EPS_CHOSEN={REG_EPS_CHOSEN}...")
    proj_full = fit_fisher_projection(select_embeddings, REG_EPS_CHOSEN)
    top_k = proj_full[:, :N_KEEP]

    print(
        "\n[4/4] Trực giao hoá + mở rộng thành M_matrix 512x512 trực giao hoàn chỉnh..."
    )
    M_new = orthonormalize_and_extend(top_k, EMBEDDING_DIM)

    os.makedirs(os.path.dirname(OUTPUT_MATRIX_PATH), exist_ok=True)
    np.save(OUTPUT_MATRIX_PATH, M_new)
    print(f"\n=== ĐÃ LƯU M_matrix MỚI (KHÔNG ghi đè bản gốc): {OUTPUT_MATRIX_PATH} ===")
    print("Bước tiếp theo: dùng wifakey_handler_lda_variant.py để load M_matrix này")
    print("qua wrapper (KHÔNG sửa wifakey_handler.py gốc), rồi chạy")
    print("run_oracle_lda_real_pipeline_test.py để đo FRR/FAR thật.")


if __name__ == "__main__":
    main()
