"""
03c_convert_demogpairs_to_ab_format.py

Chuyển audit_genuine.csv / audit_impostor_samefold.csv / audit_impostor_crossfold.csv
(sinh bởi 03b) sang định dạng mà run_ab_paired.py hiểu được:
  pairs/{tier}_genuine.csv   cột: name_enroll,imagenum_enroll,name_verify,imagenum_verify
  pairs/{tier}_impostor.csv  cùng cột

Sinh ra 2 tier độc lập (genuine giống nhau, impostor khác nhau):
  - demog_samefold  -> đo FAR trong-nhóm
  - demog_crossfold -> đo FAR liên-nhóm (audit bias chính)

Lý do KHÔNG gộp 2 loại impostor làm 1 file: run_ab_paired.py chỉ tính 1 FAR/1
McNemar tổng cho mỗi tier, gộp lại sẽ che mất chênh lệch same vs cross-fold
mà 03b cố tình tách ra để audit.

Cách chạy:
    python scripts/03c_convert_demogpairs_to_ab_format.py
"""

import os
import re
import sys
import csv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

DATASET_NAME = "demogpairs"
PAIRS_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs")

# khớp cache_path() trong 03a: "{identity}_{imagenum:04d}.npy"
# identity có thể chứa "_", nên bắt 4 chữ số cuối trước ".npy" làm imagenum
CACHE_FILENAME_RE = re.compile(r"^(?P<name>.+)_(?P<imagenum>\d{4})\.npy$")


def parse_cache_filename(cache_filename: str):
    m = CACHE_FILENAME_RE.match(cache_filename)
    if m is None:
        raise ValueError(
            f"cache_filename không khớp mẫu '{{identity}}_{{imagenum:04d}}.npy': "
            f"{cache_filename}"
        )
    return m.group("name"), int(m.group("imagenum"))


def convert_genuine():
    src = os.path.join(PAIRS_DIR, "audit_genuine.csv")
    rows_out = []
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name1, num1 = parse_cache_filename(row["cache_filename_1"])
            name2, num2 = parse_cache_filename(row["cache_filename_2"])
            rows_out.append([name1, num1, name2, num2])
    return rows_out


def convert_impostor(
    src_filename: str, id_col_1: str, id_col_2: str, file_col_1: str, file_col_2: str
):
    src = os.path.join(PAIRS_DIR, src_filename)
    rows_out = []
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name1, num1 = parse_cache_filename(row[file_col_1])
            name2, num2 = parse_cache_filename(row[file_col_2])
            rows_out.append([name1, num1, name2, num2])
    return rows_out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["name_enroll", "imagenum_enroll", "name_verify", "imagenum_verify"]
        )
        writer.writerows(rows)
    print(f"  -> {path} ({len(rows)} cặp)")


def main():
    genuine_rows = convert_genuine()

    same_rows = convert_impostor(
        "audit_impostor_samefold.csv",
        "identity_1",
        "identity_2",
        "cache_filename_1",
        "cache_filename_2",
    )
    cross_rows = convert_impostor(
        "audit_impostor_crossfold.csv",
        "identity_1",
        "identity_2",
        "cache_filename_1",
        "cache_filename_2",
    )

    print("Ghi tier 'demog_samefold' (genuine dùng chung, impostor = same-fold):")
    write_csv(os.path.join(PAIRS_DIR, "demog_samefold_genuine.csv"), genuine_rows)
    write_csv(os.path.join(PAIRS_DIR, "demog_samefold_impostor.csv"), same_rows)

    print("Ghi tier 'demog_crossfold' (genuine dùng chung, impostor = cross-fold):")
    write_csv(os.path.join(PAIRS_DIR, "demog_crossfold_genuine.csv"), genuine_rows)
    write_csv(os.path.join(PAIRS_DIR, "demog_crossfold_impostor.csv"), cross_rows)


if __name__ == "__main__":
    main()
