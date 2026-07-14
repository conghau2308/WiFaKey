"""
02_build_pairs_dataset.py

Ghép các file CSV gốc của LFW thành đúng 3 tầng dữ liệu đã thống nhất:

    tune_*     <- matchpairsDevTrain.csv / mismatchpairsDevTrain.csv   (Tầng 1: hiệu chỉnh tham số)
    select_*   <- matchpairsDevTest.csv  / mismatchpairsDevTest.csv    (Tầng 2: chọn version tốt nhất)
    final_*    <- pairs.csv (10-fold chính thức)                       (Tầng 3: CHỈ chạy 1 lần cuối)

Mỗi cặp trong output trỏ tới đường dẫn embedding cache (.npy) đã tạo ở
bước 01, và tự động LOẠI BỎ cặp nào có ảnh bị FaceProcessor từ chối
(tra theo skipped_log.csv) — kèm thống kê số lượng bị loại để bạn biết
tỷ lệ dữ liệu thực dùng được là bao nhiêu.

Cách chạy:
    python scripts/02_build_pairs_dataset.py
"""

import os
import sys
import csv

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

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
OUT_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
)
SKIPPED_LOG = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "skipped_log.csv",
)


def load_skipped() -> set:
    skipped = set()
    if os.path.exists(SKIPPED_LOG):
        with open(SKIPPED_LOG, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                skipped.add((row["name"], row["imagenum"]))
    return skipped


def cache_exists(name: str, imagenum) -> bool:
    return os.path.exists(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def build_genuine(match_csv: str, out_csv: str, skipped: set):
    kept, dropped = 0, 0
    rows_out = []
    with open(os.path.join(RAW_CSV_DIR, match_csv), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name, n1, n2 = row["name"], row["imagenum1"], row["imagenum2"]
            if (name, n1) in skipped or (name, n2) in skipped:
                dropped += 1
                continue
            if not (cache_exists(name, n1) and cache_exists(name, n2)):
                dropped += 1
                continue
            rows_out.append([name, n1, name, n2, 1])  # is_genuine=1
            kept += 1

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name_enroll",
                "imagenum_enroll",
                "name_verify",
                "imagenum_verify",
                "is_genuine",
            ]
        )
        writer.writerows(rows_out)

    print(f"  {match_csv}: giữ {kept} / loại {dropped}")


def build_impostor(mismatch_csv: str, out_csv: str, skipped: set):
    kept, dropped = 0, 0
    rows_out = []
    with open(os.path.join(RAW_CSV_DIR, mismatch_csv), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n1_name, n1_num = row["name1"], row["imagenum1"]
            n2_name, n2_num = row["name2"], row["imagenum2"]
            if (n1_name, n1_num) in skipped or (n2_name, n2_num) in skipped:
                dropped += 1
                continue
            if not (cache_exists(n1_name, n1_num) and cache_exists(n2_name, n2_num)):
                dropped += 1
                continue
            rows_out.append([n1_name, n1_num, n2_name, n2_num, 0])  # is_genuine=0
            kept += 1

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name_enroll",
                "imagenum_enroll",
                "name_verify",
                "imagenum_verify",
                "is_genuine",
            ]
        )
        writer.writerows(rows_out)

    print(f"  {mismatch_csv}: giữ {kept} / loại {dropped}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    skipped = load_skipped()
    print(f"Số ảnh bị loại ở bước trích xuất embedding: {len(skipped)}\n")

    print("=== Tầng 1: TUNE (hiệu chỉnh tham số — κ, scale LLR, fine-tune decoder) ===")
    build_genuine(
        "matchpairsDevTrain.csv", os.path.join(OUT_DIR, "tune_genuine.csv"), skipped
    )
    build_impostor(
        "mismatchpairsDevTrain.csv", os.path.join(OUT_DIR, "tune_impostor.csv"), skipped
    )

    print(
        "\n=== Tầng 2: SELECT (so sánh & chọn version tốt nhất — exp001 vs exp002 vs ...) ==="
    )
    build_genuine(
        "matchpairsDevTest.csv", os.path.join(OUT_DIR, "select_genuine.csv"), skipped
    )
    build_impostor(
        "mismatchpairsDevTest.csv",
        os.path.join(OUT_DIR, "select_impostor.csv"),
        skipped,
    )

    print(
        "\n=== Tầng 3 (pairs.csv - benchmark chính thức) KHÔNG được build tự động ở đây ==="
    )
    print(
        "   Chỉ build và chạy nó khi đã CHỐT xong version cuối cùng, chạy đúng 1 lần,"
    )
    print("   để giữ tính khách quan của con số báo cáo cuối.")


if __name__ == "__main__":
    main()
