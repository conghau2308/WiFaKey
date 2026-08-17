"""
build_hotspot_map.py

Xây dựng bản đồ điểm nóng (hotspot map) cho 1536 bit, dựa trên tần suất lỗi
quan sát được trên tập train của LFW (SecureWiFaKeyHandler).

Cách dùng:
    python build_hotspot_map.py --max-pairs 400 --output hotspot_map.npy
"""

import argparse
import os
import sys
import numpy as np
import csv

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)


def load_embedding(cache_dir, name, imagenum):
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def genuine_pairs_iter(pairs_csv, cache_dir, max_pairs=None):
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if max_pairs and count >= max_pairs:
                break
            try:
                e1 = load_embedding(
                    cache_dir, row["name_enroll"], row["imagenum_enroll"]
                )
                e2 = load_embedding(
                    cache_dir, row["name_verify"], row["imagenum_verify"]
                )
                yield e1, e2
                count += 1
            except Exception as e:
                print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--max-pairs", type=int, default=400)
    ap.add_argument("--output", default="hotspot_map.npy")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    data_dir = os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    lfw_pairs_dir = os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    lfw_cache_dir = os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )
    genuine_csv = os.path.join(lfw_pairs_dir, "tune_genuine.csv")

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    error_counts = np.zeros(1536, dtype=np.float32)
    total_counts = np.zeros(1536, dtype=np.float32)

    print(f"Quét {args.max_pairs} cặp để xây dựng hotspot map...")
    for feat_enroll, feat_verify in genuine_pairs_iter(
        genuine_csv, lfw_cache_dir, args.max_pairs
    ):
        # Enrollment
        b_full_enroll = handler._binarize_full(feat_enroll).astype(np.uint8)
        rng = np.random.default_rng()
        selection_indices = rng.choice(1536, size=832, replace=False)
        selection_mask = np.zeros(1536, dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected_enroll = b_full_enroll[selection_indices]

        # Verify
        b_full_verify = handler._binarize_full(feat_verify).astype(np.uint8)
        b_selected_verify = b_full_verify[selection_indices]

        # Lỗi bit tại các vị trí được chọn
        errors = (b_selected_enroll != b_selected_verify).astype(np.float32)

        # Cập nhật vào mảng toàn cục
        for idx, err in zip(selection_indices, errors):
            error_counts[idx] += err
            total_counts[idx] += 1.0

    # Tính xác suất lỗi cho từng vị trí, tránh chia cho 0
    prob_error = np.divide(
        error_counts,
        total_counts,
        out=np.zeros_like(error_counts),
        where=total_counts > 0,
    )

    # Làm mịn nhẹ (tránh trọng số 0 tuyệt đối)
    prob_error = np.clip(prob_error, 0.001, 0.999)

    # Chuẩn hóa thành trọng số (có thể tuyến tính hoặc log)
    # Ở đây dùng trực tiếp xác suất lỗi làm trọng số (sẽ nhân với alpha khi dùng)
    np.save(args.output, prob_error)
    print(f"Đã lưu hotspot map vào {args.output}")
    print(f"Min prob: {prob_error.min():.4f}, Max prob: {prob_error.max():.4f}")

    handler.sess.close()


if __name__ == "__main__":
    main()
