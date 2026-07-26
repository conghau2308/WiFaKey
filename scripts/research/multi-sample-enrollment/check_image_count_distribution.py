"""
check_image_count_distribution.py

Đếm số ảnh mỗi identity trong image_metadata.csv (DemogPairs, sinh bởi
03a_extract_embeddings_demogpairs.py) — trả lời câu hỏi "K=3 majority-vote
+ 1 ảnh giữ lại verify có đủ dữ liệu không" TRƯỚC khi build pairs cho C.2.

Cách chạy:
    python scripts/check_image_count_distribution.py
"""

import csv
import os
import sys
from collections import Counter, defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

IMAGE_METADATA_CSV = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", "demogpairs", "image_metadata.csv"
)


def main():
    by_identity = defaultdict(int)
    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["cache_filename"]:
                continue
            by_identity[row["identity"]] += 1

    counts = list(by_identity.values())
    n_total_identities = len(counts)
    dist = Counter(counts)

    print(f"Tổng số identity (có embedding hợp lệ): {n_total_identities}")
    print("\nPhân bố số ảnh/identity:")
    for n_img in sorted(dist.keys()):
        print(f"  {n_img} ảnh: {dist[n_img]} identity")

    print(
        "\n--- Khả năng dùng cho majority-vote enrollment (K enroll + 1 verify, không trùng ảnh) ---"
    )
    for K in [2, 3, 4, 5, 6]:
        need = K + 1
        eligible = sum(c for n_img, c in dist.items() if n_img >= need)
        # Với leave-one-out: mỗi identity có N ảnh (N>=need) cho được
        # N lần thử (mỗi lần giữ 1 ảnh khác nhau làm verify, vote K ảnh
        # còn lại -- nếu N > K thì chọn K trong N-1 ảnh còn lại mỗi lần).
        total_loo_trials = sum(
            n_img for n_img, c in dist.items() if n_img >= need for _ in range(c)
        )
        print(
            f"K={K} (cần >= {need} ảnh/identity): "
            f"{eligible} identity đủ điều kiện, "
            f"tối đa {total_loo_trials} lượt thử nếu dùng leave-one-out"
        )


if __name__ == "__main__":
    main()
