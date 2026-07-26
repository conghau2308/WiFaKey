"""
08_full_pipeline_tar_far_by_fold.py

Đo TAR/FAR THẬT qua toàn bộ pipeline production (mask ngẫu nhiên κ +
LDPC encode + Neural-MS decode), tách theo 6 fold nhân khẩu học VÀ tách
same-fold-impostor / cross-fold-impostor — không suy diễn qua công thức
lý thuyết hay mô phỏng.

MỤC ĐÍCH: kiểm chứng độc lập tuyên bố "impostor_min=22.95% > bán kính
sửa lỗi 17.62% -> FAR=0 đã đạt sẵn" từ tài liệu tham khảo — tuyên bố đó
dựa trên mô phỏng, không phải đo trên dữ liệu/pipeline thật, và không
tách theo nhóm nhân khẩu học. Script này đo trực tiếp: chạy đúng hàm
enroll()/verify() production, đếm tỷ lệ thành công thật.

API ĐÃ XÁC NHẬN từ mã nguồn gốc wifakey_handler.py (WiFaKeyHandler):
  enroll(feature_vector_float) -> (helper_data, mask_r, key_hash)
  verify(feature_vector_float, helper_data, mask_r, stored_key_hash) -> bool
mask_r sinh NGẪU NHIÊN mỗi lần gọi enroll() (không cố định theo identity)
— đúng thiết kế gốc, khớp cách script này gọi enroll() 1 lần/pair (không
tái sử dụng helper_data giữa các pair). Vẫn giữ self-check (self-match:
enroll rồi verify với ĐÚNG ảnh đó phải luôn thành công) chạy TRƯỚC vòng
lặp chính — nếu self-check fail, có nghĩa còn lệch API/import path, sửa
trước khi tin số liệu bên dưới.

LƯU Ý: verify() gốc tự in "[WiFaKey] Verify SUCCESS/FAILED" mỗi lần gọi
— với hàng nghìn pairs, console sẽ rất dài. Không ảnh hưởng kết quả,
chỉ gây khó đọc log; có thể bỏ qua hoặc tự thêm chặn stdout tạm thời
nếu muốn log gọn hơn.

CẢNH BÁO HIỆU NĂNG: verify() chạy qua Neural-MS (25 tầng) — có thể chậm
với hàng chục nghìn cặp (đặc biệt cross-fold: 15 cặp-fold x 3000 pairs
= 45000 lần decode). Có MAX_PAIRS_PER_CATEGORY để giới hạn khi chạy thử
lần đầu, tăng dần sau khi xác nhận self-check + tốc độ ổn.

Cách chạy:
    python scripts/08_full_pipeline_tar_far_by_fold.py
"""

import os
import sys
import csv
import time
import numpy as np
from collections import defaultdict
from itertools import combinations

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler

DATASET_NAME = "demogpairs"
PAIRS_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs")
GENUINE_CSV = os.path.join(PAIRS_DIR, "audit_genuine.csv")
IMPOSTOR_SAMEFOLD_CSV = os.path.join(PAIRS_DIR, "audit_impostor_samefold.csv")
IMPOSTOR_CROSSFOLD_CSV = os.path.join(PAIRS_DIR, "audit_impostor_crossfold.csv")
EMBEDDINGS_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)

OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "out_dim_separation_audit")
os.makedirs(OUT_DIR, exist_ok=True)

FOLDS = [
    "Asian_Females",
    "Asian_Males",
    "Black_Females",
    "Black_Males",
    "White_Females",
    "White_Males",
]

# Giới hạn số cặp/danh mục cho lần chạy đầu — TĂNG DẦN sau khi xác nhận
# self-check pass + biết tốc độ decode thật trên máy bạn.
MAX_PAIRS_PER_CATEGORY = 500
RNG_SEED = 42

_embedding_cache = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    if cache_filename not in _embedding_cache:
        path = os.path.join(EMBEDDINGS_CACHE_DIR, cache_filename)
        _embedding_cache[cache_filename] = np.load(path)
    return _embedding_cache[cache_filename]


# ==== Khớp API thật của WiFaKeyHandler (đã xác nhận từ mã nguồn gốc) ====
# enroll() trả (helper_data, mask_r, key_hash) — cả 3 đều cần thiết cho
# verify() sau này. "helper_data" trong toàn bộ phần còn lại của script
# này thực chất là bundle 3 phần tử đó — chỉ đóng gói/mở gói ở đúng 2
# hàm dưới đây, không cần sửa chỗ khác (measure_success_rate, self_check
# đều chỉ coi nó là 1 object cầm tay giữa enroll() và verify()).
def enroll(embedding: np.ndarray):
    helper_data, mask_r, key_hash = handler.enroll(embedding)
    return (helper_data, mask_r, key_hash)


def verify(embedding: np.ndarray, enroll_result) -> bool:
    helper_data, mask_r, key_hash = enroll_result
    return handler.verify(embedding, helper_data, mask_r, key_hash)


def self_check():
    """Enroll rồi verify với ĐÚNG ảnh đó — phải luôn thành công (self-
    match). Nếu fail, dừng ngay, đừng tin số liệu bên dưới."""
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f))
    emb = load_embedding(first_row["cache_filename_1"])

    n_trials = 5
    n_success = 0
    for _ in range(n_trials):
        helper_data = enroll(emb)
        ok = verify(emb, helper_data)
        n_success += int(bool(ok))

    print(f"[self-check] self-match: {n_success}/{n_trials} thành công.")
    if n_success != n_trials:
        raise SystemExit(
            "[self-check] FAIL — enroll+verify cùng 1 ảnh phải LUÔN thành "
            "công (self-match). Sửa phần ADAPT (enroll()/verify()) trước "
            "khi tin bất kỳ số liệu TAR/FAR nào bên dưới. DỪNG."
        )


def load_pairs():
    genuine = defaultdict(list)  # fold -> [(f1,f2)]
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            genuine[row["fold"]].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )

    impostor_same = defaultdict(list)  # fold -> [(f1,f2)]
    with open(IMPOSTOR_SAMEFOLD_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            impostor_same[row["fold"]].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )

    impostor_cross = defaultdict(list)  # (fold1,fold2) -> [(f1,f2)]
    with open(IMPOSTOR_CROSSFOLD_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["fold_1"], row["fold_2"])
            impostor_cross[key].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )

    return genuine, impostor_same, impostor_cross


def subsample(pairs: list, rng: np.random.Generator, max_n: int) -> list:
    if len(pairs) <= max_n:
        return pairs
    idx = rng.choice(len(pairs), size=max_n, replace=False)
    return [pairs[i] for i in idx]


def measure_success_rate(pairs: list, is_genuine: bool, label: str) -> dict:
    """Với genuine: enroll ảnh 1, verify ảnh 2 -> mong muốn True (TAR).
    Với impostor: enroll ảnh của identity 1, verify ảnh identity 2 ->
    mong muốn False (thành công = FAR, tức lỗ hổng)."""
    if not pairs:
        return {"n": 0, "success_rate": np.nan}

    n_success = 0
    t0 = time.time()
    for k, (f1, f2) in enumerate(pairs, start=1):
        helper_data = enroll(load_embedding(f1))
        ok = verify(load_embedding(f2), helper_data)
        n_success += int(bool(ok))
        if k % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"    [{label}] {k}/{len(pairs)}  ({elapsed:.1f}s, "
                f"~{elapsed/k*1000:.0f}ms/pair)"
            )

    rate = n_success / len(pairs)
    metric_name = "TAR" if is_genuine else "FAR"
    print(
        f"  [{label}] n={len(pairs)}  {metric_name}={rate:.4f} "
        f"({n_success}/{len(pairs)} verify thành công)"
    )
    return {"n": len(pairs), "success_rate": rate}


def main():
    self_check()

    genuine, impostor_same, impostor_cross = load_pairs()
    rng = np.random.default_rng(RNG_SEED)

    print("\n" + "=" * 70)
    print("TAR theo fold (genuine pairs)")
    print("=" * 70)
    tar_results = {}
    for fold in FOLDS:
        pairs = subsample(genuine.get(fold, []), rng, MAX_PAIRS_PER_CATEGORY)
        print(f"\n[{fold}] genuine:")
        tar_results[fold] = measure_success_rate(pairs, is_genuine=True, label=fold)

    print("\n" + "=" * 70)
    print("FAR same-fold theo fold (impostor cùng nhóm)")
    print("=" * 70)
    far_same_results = {}
    for fold in FOLDS:
        pairs = subsample(impostor_same.get(fold, []), rng, MAX_PAIRS_PER_CATEGORY)
        print(f"\n[{fold}] impostor same-fold:")
        far_same_results[fold] = measure_success_rate(
            pairs, is_genuine=False, label=fold
        )

    print("\n" + "=" * 70)
    print("FAR cross-fold theo từng cặp fold (impostor khác nhóm)")
    print("=" * 70)
    far_cross_results = {}
    for fold_a, fold_b in combinations(FOLDS, 2):
        key = (fold_a, fold_b)
        pairs = subsample(impostor_cross.get(key, []), rng, MAX_PAIRS_PER_CATEGORY)
        label = f"{fold_a} x {fold_b}"
        print(f"\n[{label}] impostor cross-fold:")
        far_cross_results[key] = measure_success_rate(
            pairs, is_genuine=False, label=label
        )

    print("\n" + "=" * 70)
    print("TỔNG KẾT — CẢNH BÁO nếu bất kỳ FAR > 0")
    print("=" * 70)
    any_far_positive = False
    for fold, r in far_same_results.items():
        if r["n"] > 0 and r["success_rate"] > 0:
            any_far_positive = True
            print(
                f"  *** FAR({fold}, same-fold) = {r['success_rate']:.4f} > 0 "
                f"— LỖ HỔNG THẬT, cần điều tra ngay. ***"
            )
    for (fa, fb), r in far_cross_results.items():
        if r["n"] > 0 and r["success_rate"] > 0:
            any_far_positive = True
            print(
                f"  *** FAR({fa} x {fb}, cross-fold) = {r['success_rate']:.4f} > 0 "
                f"— LỖ HỔNG THẬT, cần điều tra ngay. ***"
            )

    if not any_far_positive:
        print(
            "  Không phát hiện FAR>0 trong mẫu đã đo (mọi fold, mọi cặp-fold). "
            "Đây là bằng chứng THẬT (không suy diễn) cho tuyên bố FAR=0 — "
            "nhưng lưu ý: FAR=0 trên MẪU không có nghĩa FAR=0 tuyệt đối; "
            "tăng MAX_PAIRS_PER_CATEGORY nếu muốn khoảng tin cậy chặt hơn."
        )

    np.savez(
        os.path.join(OUT_DIR, "full_pipeline_tar_far.npz"),
        tar=np.array(list(tar_results.items()), dtype=object),
        far_samefold=np.array(list(far_same_results.items()), dtype=object),
        far_crossfold=np.array(list(far_cross_results.items()), dtype=object),
    )
    print(f"\n[out] -> {os.path.join(OUT_DIR, 'full_pipeline_tar_far.npz')}")


if __name__ == "__main__":
    handler = (
        WiFaKeyHandler()
    )  # dùng tham số mặc định (data_path/weights_path/biases_path) đã đúng theo mã gốc
    main()
