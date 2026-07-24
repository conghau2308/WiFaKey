"""
03b_build_pairs_demogpairs.py

Dựng genuine / impostor pairs cho DemogPairs từ image_metadata.csv
(sinh ra bởi 03a_extract_embeddings_demogpairs.py) — vì dataset này
KHÔNG có pairs list sẵn như LFW/CPLFW.

MỤC ĐÍCH KHÁC VỚI tune/select/final CỦA LFW:
  Bộ pairs này không dùng để hiệu chỉnh tham số (κ, LLR, decoder) — nó
  chỉ dùng để ĐO, không dùng để TUNE. Vì vậy không cần tách 3 tầng
  tune/select/final như LFW; rủi ro ở đây không phải overfitting tham
  số, mà là lấy mẫu thiên lệch (fold này nhiều mẫu hơn fold kia) làm
  méo kết luận về bias. Nguyên tắc thay thế: STRATIFY CHẶT theo fold,
  số lượng pairs mỗi fold/mỗi cặp-fold bằng nhau.

HAI LOẠI IMPOSTOR PAIRS được tách riêng (đây là điểm khác với LFW):
  - same-fold impostor: 2 identity khác nhau, CÙNG 1 fold nhân khẩu học
    (vd 2 người Asian_Females khác nhau). Dùng làm baseline FAR "bình
    thường" của từng nhóm.
  - cross-fold impostor: 2 identity khác nhau, KHÁC fold (vd 1 người
    Asian_Females với 1 người Black_Males). Đây là phép đo trực tiếp
    cho câu hỏi "hệ thống có dễ nhầm/khó nhầm bất thường giữa các nhóm
    nhân khẩu học hay không" — vector audit chính, không có trong pipeline
    gốc.

Cách chạy:
    python scripts/03b_build_pairs_demogpairs.py
"""

import os
import sys
import csv
import random
from collections import defaultdict
from itertools import combinations

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

DATASET_NAME = "demogpairs"

IMAGE_METADATA_CSV = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "image_metadata.csv"
)
PAIRS_OUT_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs"
)

RANDOM_SEED = 42

# Trần số genuine pair lấy từ 1 identity — tránh identity có nhiều ảnh
# áp đảo thống kê (vd 1 người có 40 ảnh sẽ sinh C(40,2)=780 cặp nếu
# không giới hạn).
MAX_GENUINE_PER_IDENTITY = 10

# Số impostor pairs lấy cho MỖI danh mục (mỗi fold riêng cho same-fold,
# mỗi cặp-fold riêng cho cross-fold) — cố định bằng nhau giữa các danh
# mục để so sánh FAR giữa nhóm không bị lệch do cỡ mẫu khác nhau.
TARGET_IMPOSTOR_PER_CATEGORY = 3000

FOLDS = [
    "Asian_Females",
    "Asian_Males",
    "Black_Females",
    "Black_Males",
    "White_Females",
    "White_Males",
]


def load_metadata():
    """Đọc image_metadata.csv, chỉ giữ ảnh trích xuất thành công
    (status 'ok' hoặc 'cached', tức có cache_filename)."""
    if not os.path.exists(IMAGE_METADATA_CSV):
        raise FileNotFoundError(f"Không tìm thấy {IMAGE_METADATA_CSV}. Chạy 03a trước.")

    by_identity = defaultdict(list)  # identity -> list[(cache_filename, fold)]
    fold_of_identity = {}

    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["cache_filename"]:
                continue  # trích xuất thất bại, bỏ qua
            identity = row["identity"]
            fold = row["fold"]
            by_identity[identity].append((row["cache_filename"], fold))

            # 1 identity phải luôn thuộc đúng 1 fold — cảnh báo nếu mâu thuẫn
            if identity in fold_of_identity and fold_of_identity[identity] != fold:
                print(
                    f"*** CẢNH BÁO: identity '{identity}' xuất hiện ở nhiều "
                    f"fold khác nhau ({fold_of_identity[identity]} vs {fold}). "
                    f"Kiểm tra lại 03a / metadata gốc. ***"
                )
            fold_of_identity[identity] = fold

    return by_identity, fold_of_identity


def build_genuine_pairs(by_identity, rng):
    """Với mỗi identity có >=2 ảnh, sample tối đa MAX_GENUINE_PER_IDENTITY
    cặp (identity, img1, img2, fold)."""
    rows = []
    for identity, images in by_identity.items():
        if len(images) < 2:
            continue
        fold = images[0][1]
        all_pairs = list(combinations(images, 2))
        rng.shuffle(all_pairs)
        chosen = all_pairs[:MAX_GENUINE_PER_IDENTITY]
        for (img1, _), (img2, _) in chosen:
            rows.append([identity, img1, img2, fold])
    return rows


def build_impostor_pairs_same_fold(by_identity, fold_of_identity, rng):
    """2 identity khác nhau, cùng fold."""
    identities_by_fold = defaultdict(list)
    for identity in by_identity:
        identities_by_fold[fold_of_identity[identity]].append(identity)

    rows = []
    for fold, identities in identities_by_fold.items():
        if len(identities) < 2:
            print(
                f"*** CẢNH BÁO: fold '{fold}' chỉ có {len(identities)} "
                f"identity, không đủ để sample impostor same-fold. ***"
            )
            continue

        n_sampled = 0
        attempts = 0
        max_attempts = TARGET_IMPOSTOR_PER_CATEGORY * 20
        while n_sampled < TARGET_IMPOSTOR_PER_CATEGORY and attempts < max_attempts:
            attempts += 1
            id1, id2 = rng.sample(identities, 2)
            img1 = rng.choice(by_identity[id1])[0]
            img2 = rng.choice(by_identity[id2])[0]
            rows.append([fold, id1, img1, id2, img2])
            n_sampled += 1

        if n_sampled < TARGET_IMPOSTOR_PER_CATEGORY:
            print(
                f"*** CẢNH BÁO: fold '{fold}' chỉ sample được {n_sampled}"
                f"/{TARGET_IMPOSTOR_PER_CATEGORY} impostor same-fold pairs "
                f"(hết attempts). ***"
            )
    return rows


def build_impostor_pairs_cross_fold(by_identity, fold_of_identity, rng):
    """2 identity khác nhau, khác fold — mọi cặp fold trong C(6,2)=15 tổ hợp."""
    identities_by_fold = defaultdict(list)
    for identity in by_identity:
        identities_by_fold[fold_of_identity[identity]].append(identity)

    rows = []
    for fold_a, fold_b in combinations(FOLDS, 2):
        ids_a = identities_by_fold.get(fold_a, [])
        ids_b = identities_by_fold.get(fold_b, [])
        if not ids_a or not ids_b:
            print(
                f"*** CẢNH BÁO: thiếu identity cho cặp fold "
                f"({fold_a}, {fold_b}) — bỏ qua. ***"
            )
            continue

        n_sampled = 0
        attempts = 0
        max_attempts = TARGET_IMPOSTOR_PER_CATEGORY * 20
        while n_sampled < TARGET_IMPOSTOR_PER_CATEGORY and attempts < max_attempts:
            attempts += 1
            id_a = rng.choice(ids_a)
            id_b = rng.choice(ids_b)
            img_a = rng.choice(by_identity[id_a])[0]
            img_b = rng.choice(by_identity[id_b])[0]
            rows.append([fold_a, fold_b, id_a, img_a, id_b, img_b])
            n_sampled += 1

        if n_sampled < TARGET_IMPOSTOR_PER_CATEGORY:
            print(
                f"*** CẢNH BÁO: cặp fold ({fold_a}, {fold_b}) chỉ sample "
                f"được {n_sampled}/{TARGET_IMPOSTOR_PER_CATEGORY} pairs. ***"
            )
    return rows


def main():
    os.makedirs(PAIRS_OUT_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    by_identity, fold_of_identity = load_metadata()
    n_identities = len(by_identity)
    n_images = sum(len(v) for v in by_identity.values())
    print(
        f"Đã load {n_identities} identity, {n_images} ảnh (thành công) "
        f"từ image_metadata.csv."
    )

    id_counts_per_fold = defaultdict(int)
    for identity, fold in fold_of_identity.items():
        id_counts_per_fold[fold] += 1
    print("Số identity theo fold:", dict(id_counts_per_fold))

    genuine_rows = build_genuine_pairs(by_identity, rng)
    print(
        f"\nGenuine pairs: {len(genuine_rows)} "
        f"(trần {MAX_GENUINE_PER_IDENTITY} cặp/identity)"
    )

    impostor_same_rows = build_impostor_pairs_same_fold(
        by_identity, fold_of_identity, rng
    )
    print(
        f"Impostor same-fold pairs: {len(impostor_same_rows)} "
        f"(mục tiêu {TARGET_IMPOSTOR_PER_CATEGORY}/fold x {len(FOLDS)} fold)"
    )

    impostor_cross_rows = build_impostor_pairs_cross_fold(
        by_identity, fold_of_identity, rng
    )
    n_fold_pairs = len(list(combinations(FOLDS, 2)))
    print(
        f"Impostor cross-fold pairs: {len(impostor_cross_rows)} "
        f"(mục tiêu {TARGET_IMPOSTOR_PER_CATEGORY}/cặp-fold x {n_fold_pairs} cặp-fold)"
    )

    genuine_path = os.path.join(PAIRS_OUT_DIR, "audit_genuine.csv")
    with open(genuine_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["identity", "cache_filename_1", "cache_filename_2", "fold"])
        writer.writerows(genuine_rows)

    same_path = os.path.join(PAIRS_OUT_DIR, "audit_impostor_samefold.csv")
    with open(same_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["fold", "identity_1", "cache_filename_1", "identity_2", "cache_filename_2"]
        )
        writer.writerows(impostor_same_rows)

    cross_path = os.path.join(PAIRS_OUT_DIR, "audit_impostor_crossfold.csv")
    with open(cross_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fold_1",
                "fold_2",
                "identity_1",
                "cache_filename_1",
                "identity_2",
                "cache_filename_2",
            ]
        )
        writer.writerows(impostor_cross_rows)

    print(
        f"\n=== HOÀN TẤT ===\nĐã ghi:\n  {genuine_path}\n  {same_path}\n  {cross_path}"
    )
    print(
        "\nTiếp theo có 2 hướng độc lập, dùng chung bộ pairs này:\n"
        "  (A) 04_dimension_demographic_leakage.py — dùng audit_genuine.csv,\n"
        "      tính F-statistic/mutual-info mỗi chiều embedding theo nhãn fold,\n"
        "      đối chiếu với ranking Fisher-ratio (identity) đã có.\n"
        "  (B) 05_verify_benchmark_by_fold.py — chạy enroll+verify() thật qua\n"
        "      audit_genuine.csv (đo FRR/BER theo fold) và cả 2 file impostor\n"
        "      (đo FAR same-fold vs cross-fold riêng biệt)."
    )


if __name__ == "__main__":
    main()
