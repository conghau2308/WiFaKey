"""
02b_extract_embeddings_face_detection.py

Trích xuất embedding cho dataset face-detection-and-re-identification.
Dataset này CHỈ có 1 person/thư mục với 2 ảnh (0.jpg, 1.jpg) -> chỉ tạo
được cặp GENUINE (0.jpg vs 1.jpg), KHÔNG có impostor.

Output khớp quy ước với run_ab_paired.py:
  - embeddings_cache/{name}_{imagenum:04d}.npy
  - pairs/select_genuine.csv  (cột: name_enroll, imagenum_enroll,
                                name_verify, imagenum_verify)
"""

import os
import sys
import csv
import cv2
import numpy as np
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

from vision_module.face_processor import FaceProcessor
from feature_extractor.adaface_handler import AdaFaceExtractor

DATASET_NAME = "face-detection-and-re-identification"

RAW_IMG_DIR = os.path.join(_PROJECT_ROOT, "datasets", "raw", DATASET_NAME, "train")
RAW_CSV_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", DATASET_NAME, "re-identification.csv"
)
CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)
SKIPPED_LOG = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "skipped_log.csv"
)
GENUINE_CSV = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs", "select_genuine.csv"
)

IMG_NAMES = {0: "0.jpg", 1: "1.jpg"}  # imagenum -> filename trong thư mục person


def image_path(name: str, imagenum: int) -> str:
    return os.path.join(RAW_IMG_DIR, name, IMG_NAMES[imagenum])


def cache_path(name: str, imagenum: int) -> str:
    # Cùng quy ước với _load_embedding() trong run_ab_paired.py
    return os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy")


def collect_required_persons() -> set:
    required = set()
    if not os.path.exists(RAW_CSV_DIR):
        raise FileNotFoundError(f"Không tìm thấy {RAW_CSV_DIR}.")
    with open(RAW_CSV_DIR, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            required.add(row["person"])
    return required


def extract_one(face_processor, adaface, name: str, imagenum: int):
    """Trả (thành_công: bool, status: str). Dùng cache nếu đã có sẵn."""
    out_path = cache_path(name, imagenum)
    if os.path.exists(out_path):
        return True, "cached"

    img_path = image_path(name, imagenum)
    if not os.path.exists(img_path):
        return False, "missing_image"

    raw_image = cv2.imread(img_path)
    if raw_image is None:
        return False, "unreadable_image"

    aligned_rgb, status = face_processor.process(raw_image)
    if aligned_rgb is None:
        return False, status

    embedding = adaface.get_feature_vector(aligned_rgb)
    np.save(out_path, embedding)
    return True, "ok"


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(GENUINE_CSV), exist_ok=True)

    required_persons = collect_required_persons()
    print(f"Đã thu thập {len(required_persons)} person cần trích xuất embedding.")

    print(
        "Khởi tạo FaceProcessor + AdaFaceExtractor (dùng đúng pipeline production)..."
    )
    face_processor = FaceProcessor(
        det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7
    )
    adaface = AdaFaceExtractor(device="cuda")

    n_ok, n_skip = 0, 0
    skip_rows = []
    genuine_rows = []

    for i, name in enumerate(sorted(required_persons), start=1):
        # Cả 2 ảnh (0.jpg=enroll, 1.jpg=verify) PHẢI thành công thì cặp
        # genuine mới hợp lệ - nếu 1 trong 2 fail thì bỏ luôn cả person.
        ok0, status0 = extract_one(face_processor, adaface, name, 0)
        if not ok0:
            skip_rows.append([name, "0.jpg", status0])
            n_skip += 1
            continue

        ok1, status1 = extract_one(face_processor, adaface, name, 1)
        if not ok1:
            skip_rows.append([name, "1.jpg", status1])
            n_skip += 1
            continue

        genuine_rows.append(
            {
                "name_enroll": name,
                "imagenum_enroll": 0,
                "name_verify": name,
                "imagenum_verify": 1,
            }
        )
        n_ok += 1

        if i % 200 == 0:
            print(f"  ... {i}/{len(required_persons)} (ok={n_ok}, skip={n_skip})")

    with open(SKIPPED_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person", "image", "reason"])
        writer.writerows(skip_rows)

    with open(GENUINE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name_enroll",
                "imagenum_enroll",
                "name_verify",
                "imagenum_verify",
            ],
        )
        writer.writeheader()
        writer.writerows(genuine_rows)

    print("\n=== HOÀN TẤT ===")
    print(f"Thành công (cả 2 ảnh): {n_ok}")
    print(f"Bị loại: {n_skip} (chi tiết: {SKIPPED_LOG})")
    if n_skip > 0:
        reasons = Counter(r[2] for r in skip_rows)
        print("Lý do bị loại:", dict(reasons))
    print(f"\nGhi genuine pairs vào: {GENUINE_CSV}")
    print(
        "Lưu ý: dataset này KHÔNG có impostor.csv -> chạy run_ab_paired.py sẽ "
        "tự động chỉ đo FRR, bỏ qua FAR."
    )


if __name__ == "__main__":
    main()
