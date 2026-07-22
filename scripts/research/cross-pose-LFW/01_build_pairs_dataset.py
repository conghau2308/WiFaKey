"""
03b_build_pairs_cplfw.py

Build select_genuine.csv / select_impostor.csv cho dataset CPLFW từ file
pairs_CPLFW.txt gốc, ĐÚNG FORMAT mà run_ab_paired.py đang đọc (cột:
name_enroll, imagenum_enroll, name_verify, imagenum_verify).

MẸO TƯƠNG THÍCH (giống 03a_extract_embeddings_cplfw.py): vì CPLFW không tổ
chức theo person/imagenum, ta dùng chính tên file ảnh (bỏ đuôi) làm
"name", và luôn gán imagenum=0 -> khớp với cache "{name}_0000.npy" đã tạo
ở bước 03a.

CHỈ CHẠY SAU KHI đã chạy xong 03a_extract_embeddings_cplfw.py, vì script
này sẽ chỉ giữ lại các cặp mà CẢ HAI ảnh đều đã có embedding trong cache
(bỏ qua cặp có ảnh bị FaceProcessor loại - vd không detect được mặt).

Cách chạy:
    python scripts/03b_build_pairs_cplfw.py
"""

import os
import sys
import csv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

DATASET_NAME = "cplfw"

CPLFW_PAIRS_TXT = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "cplfw", "pairs_CPLFW.txt"
)
CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)
PAIRS_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs")
GENUINE_CSV = os.path.join(PAIRS_DIR, "select_genuine.csv")
IMPOSTOR_CSV = os.path.join(PAIRS_DIR, "select_impostor.csv")


def image_id_from_filename(filename: str) -> str:
    return os.path.splitext(filename)[0]


def cache_path(image_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{image_id}_0000.npy")


def parse_pairs_line(line: str):
    """PHẢI GIỐNG HỆT parse_pairs_line() trong 03a_extract_embeddings_cplfw.py
    - nếu sửa 1 file thì sửa cả 2 để tránh lệch logic."""
    line = line.strip()
    if not line:
        return None
    tokens = line.split()
    if len(tokens) != 3:
        return None
    img1, img2, label_str = tokens
    if label_str not in ("0", "1"):
        return None
    return img1, img2, label_str == "1"


def main():
    if not os.path.exists(CPLFW_PAIRS_TXT):
        raise FileNotFoundError(f"Không tìm thấy {CPLFW_PAIRS_TXT}.")
    if not os.path.isdir(CACHE_DIR) or not os.listdir(CACHE_DIR):
        raise FileNotFoundError(
            f"Cache embedding trống ở {CACHE_DIR}. Chạy "
            f"scripts/03a_extract_embeddings_cplfw.py trước."
        )

    os.makedirs(PAIRS_DIR, exist_ok=True)

    with open(CPLFW_PAIRS_TXT, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    genuine_rows = []
    impostor_rows = []
    n_bad_format = 0
    n_missing_embedding = 0

    for line in raw_lines:
        parsed = parse_pairs_line(line)
        if parsed is None:
            if line.strip():
                n_bad_format += 1
            continue

        img1, img2, is_genuine = parsed
        id1 = image_id_from_filename(img1)
        id2 = image_id_from_filename(img2)

        # Bỏ qua cặp nếu 1 trong 2 ảnh không có embedding (bị FaceProcessor
        # loại ở bước 03a, vd không detect được mặt).
        if not os.path.exists(cache_path(id1)) or not os.path.exists(cache_path(id2)):
            n_missing_embedding += 1
            continue

        row = {
            "name_enroll": id1,
            "imagenum_enroll": 0,
            "name_verify": id2,
            "imagenum_verify": 0,
        }
        if is_genuine:
            genuine_rows.append(row)
        else:
            impostor_rows.append(row)

    fieldnames = ["name_enroll", "imagenum_enroll", "name_verify", "imagenum_verify"]

    with open(GENUINE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(genuine_rows)

    with open(IMPOSTOR_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(impostor_rows)

    print("=== HOÀN TẤT BUILD PAIRS CPLFW ===")
    print(f"Genuine pairs : {len(genuine_rows)}  -> {GENUINE_CSV}")
    print(f"Impostor pairs: {len(impostor_rows)}  -> {IMPOSTOR_CSV}")
    if n_bad_format > 0:
        print(
            f"CẢNH BÁO: {n_bad_format} dòng trong pairs_CPLFW.txt không đúng "
            f"format giả định - xem lại parse_pairs_line()."
        )
    if n_missing_embedding > 0:
        print(
            f"Đã bỏ qua {n_missing_embedding} cặp vì thiếu embedding (ảnh bị "
            f"loại ở bước trích xuất - xem datasets/processed/{DATASET_NAME}/"
            f"skipped_log.csv)."
        )
    print(
        "\nTiếp theo: chạy "
        "'python experiments/run_ab_paired.py --dataset cplfw' "
        "(nhớ thêm 'cplfw' vào KNOWN_DATASETS trong run_ab_paired.py trước)."
    )


if __name__ == "__main__":
    main()
