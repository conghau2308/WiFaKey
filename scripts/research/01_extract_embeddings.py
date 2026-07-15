"""
01_extract_embeddings.py

Duyệt qua TOÀN BỘ ảnh được tham chiếu trong các file match/mismatch CSV gốc
của LFW, chạy qua đúng pipeline production (FaceProcessor -> AdaFaceExtractor),
và cache embedding ra .npy để mọi experiment sau dùng lại — đảm bảo:
  1. Không phải chạy lại GPU inference mỗi lần thử nghiệm (tốn thời gian).
  2. Mọi version (v0, v1, v2...) dùng CHÍNH XÁC cùng 1 bộ embedding input,
     chỉ khác nhau ở phần crypto (modulation/decoder/quantizer) đang so sánh.

Ảnh bị FaceProcessor từ chối (no_face / low_confidence / spoof_detected /
no_landmarks) sẽ được ghi vào skipped_log.csv thay vì âm thầm bỏ qua —
để bạn biết chính xác bao nhiêu % dữ liệu bị loại và vì lý do gì.

Cách chạy:
    python scripts/01_extract_embeddings.py
"""

import os
import sys
import csv
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) # Đi lên 2 cấp
sys.path.insert(0, _PROJECT_ROOT)

from vision_module.face_processor import FaceProcessor
from feature_extractor.adaface_handler import AdaFaceExtractor

RAW_IMG_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "labeled_faces_in_the_wild", "lfw-deepfunneled"
)
RAW_CSV_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "labeled_faces_in_the_wild", "csv"
)
CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)
SKIPPED_LOG = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "skipped_log.csv",
)

PAIR_FILES = [
    "matchpairsDevTrain.csv",
    "matchpairsDevTest.csv",
    "mismatchpairsDevTrain.csv",
    "mismatchpairsDevTest.csv",
    # "pairs.csv" cố ý KHÔNG đưa vào đây trong giai đoạn tuning/selection.
    # Chỉ thêm vào khi chạy final report (tầng 3), chạy riêng, không lẫn với embedding cache dùng để tune.
]


def image_path(name: str, imagenum) -> str:
    return os.path.join(RAW_IMG_DIR, name, f"{name}_{int(imagenum):04d}.jpg")


def cache_path(name: str, imagenum) -> str:
    return os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy")


def collect_required_images() -> set:
    """Đọc mọi CSV, thu thập tập hợp (name, imagenum) duy nhất cần trích xuất embedding."""
    required = set()
    for fname in PAIR_FILES:
        fpath = os.path.join(RAW_CSV_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Không tìm thấy {fpath}, bỏ qua.")
            continue
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "mismatch" in fname:
                    # cột: name1, imagenum1, name2, imagenum2
                    required.add((row["name1"], row["imagenum1"]))
                    required.add((row["name2"], row["imagenum2"]))
                else:
                    # cột: name, imagenum1, imagenum2
                    required.add((row["name"], row["imagenum1"]))
                    required.add((row["name"], row["imagenum2"]))
    return required


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Đang đọc danh sách ảnh cần trích xuất từ CSV...")
    required = collect_required_images()
    print(f"→ Tổng {len(required)} ảnh duy nhất cần xử lý.")

    print(
        "Khởi tạo FaceProcessor + AdaFaceExtractor (dùng đúng pipeline production)..."
    )
    face_processor = FaceProcessor(
        det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7
    )
    adaface = AdaFaceExtractor(device="cuda")

    n_ok, n_skip, n_cached = 0, 0, 0
    skip_rows = []

    for name, imagenum in sorted(required):
        out_path = cache_path(name, imagenum)
        if os.path.exists(out_path):
            n_cached += 1
            continue

        img_path = image_path(name, imagenum)
        if not os.path.exists(img_path):
            skip_rows.append([name, imagenum, "file_not_found"])
            n_skip += 1
            continue

        raw_image = cv2.imread(img_path)
        aligned_rgb, status = face_processor.process(raw_image)
        if aligned_rgb is None:
            skip_rows.append([name, imagenum, status])
            n_skip += 1
            continue

        embedding = adaface.get_feature_vector(aligned_rgb)
        np.save(out_path, embedding)
        n_ok += 1

        if (n_ok + n_skip) % 200 == 0:
            print(
                f"  ... {n_ok+n_skip+n_cached}/{len(required)} "
                f"(ok={n_ok}, skip={n_skip}, cached={n_cached})"
            )

    with open(SKIPPED_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "imagenum", "reason"])
        writer.writerows(skip_rows)

    print("\n=== HOÀN TẤT ===")
    print(f"Thành công: {n_ok}")
    print(f"Đã cache từ trước: {n_cached}")
    print(f"Bị loại: {n_skip} (chi tiết: {SKIPPED_LOG})")
    if n_skip > 0:
        from collections import Counter

        reasons = Counter(r[2] for r in skip_rows)
        print("Lý do bị loại:", dict(reasons))


if __name__ == "__main__":
    main()
