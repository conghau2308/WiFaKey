"""
build_multisample_pairs_demogpairs.py

Xây pairs cho thử nghiệm majority-vote enrollment (C.2) trên DemogPairs,
dùng LEAVE-ONE-OUT để tận dụng tối đa dữ liệu (mỗi identity trung bình chỉ
vài ảnh):

  Với 1 identity có N ảnh (N >= K+1):
    với mỗi ảnh i trong N ảnh:
      - verify_image = ảnh i (giữ lại, KHÔNG dùng để vote)
      - enroll_images = chọn ngẫu nhiên K ảnh trong (N-1) ảnh còn lại
      -> N lượt thử genuine cho identity đó (thay vì chỉ 1 nếu chia cố định)

Sinh cả genuine (cùng identity, đúng leave-one-out ở trên) và impostor
(verify_image của identity A, đem so với enroll_images vote từ 1 identity B
khác ngẫu nhiên) để đo được cả GMR lẫn FAR qua majority-vote, so sánh trực
tiếp với K=1 (baseline không vote, đã có ở benchmark_real_decode.py).

Cách chạy:
    python scripts/build_multisample_pairs_demogpairs.py --k 3
    python scripts/build_multisample_pairs_demogpairs.py --k 5
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

IMAGE_METADATA_CSV = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", "demogpairs", "image_metadata.csv"
)
PAIRS_OUT_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", "demogpairs", "pairs"
)

RANDOM_SEED = 42

# Số impostor pair sinh ra CHO MỖI genuine trial (giữ tỉ lệ genuine:impostor
# cân bằng để benchmark GMR/FAR trên cùng 1 quy mô, giống tune_genuine/
# tune_impostor của LFW).
IMPOSTOR_PER_GENUINE = 1


def load_metadata():
    by_identity = defaultdict(list)  # identity -> [cache_filename, ...]
    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["cache_filename"]:
                continue
            by_identity[row["identity"]].append(row["cache_filename"])
    return by_identity


def build_genuine_leave_one_out(by_identity, K, rng):
    """Với mỗi identity có N >= K+1 ảnh, sinh N trial leave-one-out."""
    rows = []
    for identity, images in by_identity.items():
        n_img = len(images)
        if n_img < K + 1:
            continue
        for i in range(n_img):
            verify_img = images[i]
            pool = images[:i] + images[i + 1 :]  # N-1 ảnh còn lại
            enroll_imgs = rng.sample(pool, K) if len(pool) > K else pool
            rows.append(
                {
                    "identity": identity,
                    "verify_cache_filename": verify_img,
                    "enroll_cache_filenames": ";".join(enroll_imgs),
                    "k_actual": len(enroll_imgs),
                }
            )
    return rows


def build_impostor_from_genuine(by_identity, genuine_rows, rng):
    """Với mỗi genuine trial, ghép verify_image đó với enroll_images VOTE
    TỪ 1 IDENTITY KHÁC ngẫu nhiên (không phải identity của chính verify_image)
    -- đo FAR của majority-vote template so với người khác."""
    identities = list(by_identity.keys())
    rows = []
    for g in genuine_rows:
        for _ in range(IMPOSTOR_PER_GENUINE):
            other_identity = g["identity"]
            attempts = 0
            while other_identity == g["identity"] and attempts < 50:
                other_identity = rng.choice(identities)
                attempts += 1
            if other_identity == g["identity"]:
                continue  # không tìm được identity khác (dataset quá nhỏ), bỏ qua

            other_images = by_identity[other_identity]
            k_needed = g["k_actual"]
            enroll_imgs = (
                rng.sample(other_images, k_needed)
                if len(other_images) > k_needed
                else other_images
            )
            rows.append(
                {
                    "verify_identity": g["identity"],
                    "verify_cache_filename": g["verify_cache_filename"],
                    "enroll_identity": other_identity,
                    "enroll_cache_filenames": ";".join(enroll_imgs),
                    "k_actual": len(enroll_imgs),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--k", type=int, required=True, help="Số ảnh dùng để majority-vote enroll"
    )
    args = ap.parse_args()

    os.makedirs(PAIRS_OUT_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    by_identity = load_metadata()
    print(f"Đã load {len(by_identity)} identity từ {IMAGE_METADATA_CSV}")

    genuine_rows = build_genuine_leave_one_out(by_identity, args.k, rng)
    n_identities_used = len(set(r["identity"] for r in genuine_rows))
    print(
        f"\nK={args.k}: {len(genuine_rows)} genuine trial (leave-one-out) "
        f"từ {n_identities_used} identity đủ điều kiện (>= {args.k + 1} ảnh)"
    )

    impostor_rows = build_impostor_from_genuine(by_identity, genuine_rows, rng)
    print(
        f"Impostor trial (ghép chéo, {IMPOSTOR_PER_GENUINE}x/genuine): {len(impostor_rows)}"
    )

    genuine_path = os.path.join(PAIRS_OUT_DIR, f"multisample_K{args.k}_genuine.csv")
    with open(genuine_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "identity",
                "verify_cache_filename",
                "enroll_cache_filenames",
                "k_actual",
            ],
        )
        writer.writeheader()
        writer.writerows(genuine_rows)

    impostor_path = os.path.join(PAIRS_OUT_DIR, f"multisample_K{args.k}_impostor.csv")
    with open(impostor_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "verify_identity",
                "verify_cache_filename",
                "enroll_identity",
                "enroll_cache_filenames",
                "k_actual",
            ],
        )
        writer.writeheader()
        writer.writerows(impostor_rows)

    print(f"\n=== HOÀN TẤT ===\nĐã ghi:\n  {genuine_path}\n  {impostor_path}")
    print(
        "\nTiếp theo: viết handler majority-vote (đọc enroll_cache_filenames, "
        "binarize từng ảnh, vote per-bit -> b_full đồng thuận) + benchmark "
        "thật qua run_single_config.py-style, so K=1 (đã có, 42.45% GMR) "
        "với K hiện tại."
    )


if __name__ == "__main__":
    main()
