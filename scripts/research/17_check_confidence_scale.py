"""
14_check_confidence_scale.py

Nghi vấn: tham số scale=1.0 mặc định của SoftDistanceLLR không khớp với
thang giá trị thật của 'distance' (khoảng cách tới ngưỡng binarization)
tính từ M_matrix/binarization_intervals.npy THẬT của dự án.

Script này đo trực tiếp phân phối 'distance' trên vài embedding LFW thật,
in ra min/max/mean/percentile - để tìm giá trị scale phù hợp sao cho
magnitude LLR trung bình rơi vào khoảng ~1.0 (khớp thang mà decoder đã
được huấn luyện, tương đương hard BPSK).

Cách chạy:
    python scripts/research/14_check_confidence_scale.py
"""

import os
import sys
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from research.quantizer.v0_lssc_with_confidence import binarize_with_confidence

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)

PAIR_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "pairs",
)


def collect_unique_embeddings_from_tune():
    csv_files = [
        os.path.join(PAIR_DIR, "tune_genuine.csv"),
        os.path.join(PAIR_DIR, "tune_impostor.csv"),
    ]

    # dùng set các cặp (name, imagenum) thay vì chỉ name,
    # vì file embedding được đặt tên theo cả hai: {name}_{imagenum:04d}.npy
    keys = set()

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            keys.add((row["name_enroll"], int(row["imagenum_enroll"])))
            keys.add((row["name_verify"], int(row["imagenum_verify"])))

    embeddings = []

    for name, imagenum in sorted(keys):
        filename = f"{name}_{imagenum:04d}.npy"
        path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy embedding: {path}")
        embeddings.append(np.load(path))

    return embeddings


def main():
    handler = WiFaKeyHandler()

    print(f"handler.intervals = {handler.intervals}")
    print(
        f"handler.intervals dtype={handler.intervals.dtype}, "
        f"khoảng cách giữa các ngưỡng liên tiếp: "
        f"{np.diff(np.sort(handler.intervals))}\n"
    )

    embeddings = collect_unique_embeddings_from_tune()
    all_distances = []

    for emb in embeddings:
        projected = np.dot(emb, handler.M_matrix)
        _, confidence = binarize_with_confidence(projected, handler.intervals)
        all_distances.append(confidence)

    all_distances = np.concatenate(all_distances)

    print(f"=== Phân phối 'distance' trên {len(embeddings)} embedding thật ===")
    print(f"min={all_distances.min():.6f}")
    print(f"max={all_distances.max():.6f}")
    print(f"mean={all_distances.mean():.6f}")
    print(f"median={np.median(all_distances):.6f}")
    print(f"p10={np.percentile(all_distances, 10):.6f}")
    print(f"p90={np.percentile(all_distances, 90):.6f}")

    suggested_scale = 1.0 / max(all_distances.mean(), 1e-8)
    print(
        f"\n💡 Gợi ý scale để magnitude trung bình ≈ 1.0: scale ≈ {suggested_scale:.4f}"
    )
    print(
        f"   (magnitude hiện tại với scale=1.0: mean={all_distances.mean():.6f} "
        f"-> gần như LUÔN bị clip xuống min_mag=0.1 nếu mean < 0.1)"
    )

    if all_distances.mean() < 0.1:
        print(
            "\n❌ XÁC NHẬN: distance trung bình < min_mag=0.1 hiện tại - với scale=1.0,"
        )
        print("   hầu hết các vị trí bị clip về đúng 0.1, làm mất hết thông tin soft")
        print(
            "   VÀ giảm biên độ LLR xuống 10 lần so với hard BPSK (biên độ luôn 1.0)."
        )
        print(f"   -> Thử lại với scale={suggested_scale:.4f} thay vì 1.0.")


if __name__ == "__main__":
    main()
