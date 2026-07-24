"""
03a_extract_embeddings_demogpairs.py

Trích xuất embedding cho dataset DemogPairs.

CẤU TRÚC DATASET DEMOGPAIRS (khác LFW và khác CPLFW):
  root/
    MetaData/
      Asian_Females.txt
      Asian_Males.txt
      Black_Females.txt
      Black_Males.txt
      White_Females.txt
      White_Males.txt
    DemogPairs/
      <identity_folder>/
        002.jpg
        004.jpg
        007.jpg
        ...

  Mỗi file MetaData/<Fold>.txt có 1 dòng HEADER "db_code image_path" rồi
  tới các dòng dữ liệu dạng:
      <db_code> <image_path>
  ví dụ:
      CWF able_wanamakok/002.jpg

  KHÔNG CÓ pairs list (khác CPLFW/LFW) — DemogPairs chỉ cho biết ảnh nào
  thuộc fold nhân khẩu học nào và thuộc identity nào (tên thư mục con).
  Vì vậy script này, ngoài việc trích xuất embedding, còn ghi ra 1 file
  metadata CSV (identity, imagenum, fold, db_code, cache_filename) để
  script xây pairs (03b) dùng — không có pairs.txt để đối chiếu như CPLFW.

QUY ƯỚC IMAGENUM:
  Không dùng index tuần tự như CPLFW (vì CPLFW không có "person/số ảnh"),
  DemogPairs đã có sẵn tên file dạng số ("002.jpg") bên trong thư mục
  identity, giống LFW gốc — nên lấy thẳng số đó làm imagenum. Điều này
  nhất quán, không phụ thuộc thứ tự duyệt thư mục, và khớp quy ước
  "{name}_{imagenum:04d}.npy" mà run_ab_paired.py đang dùng.

QUAN TRỌNG - HÃY KIỂM TRA TRƯỚC KHI CHẠY:
  Mình không tự tải được DemogPairs để xác nhận 100% định dạng dòng
  MetaData, nên script in ra 5 dòng đầu của MỖI file fold trước khi
  parse, và CẢNH BÁO (không âm thầm bỏ qua) nếu có dòng không đúng
  2 token hoặc image_path không khớp mẫu "<identity>/<số>.jpg". Nếu có
  cảnh báo, mở file .txt bằng tay rồi sửa hàm parse_metadata_line().

Cách chạy:
    python scripts/03a_extract_embeddings_demogpairs.py
"""

import os
import re
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

DATASET_NAME = "demogpairs"

# CHỈNH LẠI 2 ĐƯỜNG DẪN NÀY cho khớp với nơi bạn giải nén DemogPairs
DEMOGPAIRS_IMG_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "demogpairs", "DemogPairs"
)
DEMOGPAIRS_METADATA_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "demogpairs", "MetaData"
)

CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)
SKIPPED_LOG = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "skipped_log.csv"
)
# Metadata output — thay thế vai trò của pairs.txt (DemogPairs không có
# sẵn). 03b_build_pairs_demogpairs.py sẽ đọc file này để dựng genuine
# (cùng identity) / impostor (khác identity, same-fold & cross-fold).
IMAGE_METADATA_CSV = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "image_metadata.csv"
)

FOLD_FILES = [
    "Asian_Females.txt",
    "Asian_Males.txt",
    "Black_Females.txt",
    "Black_Males.txt",
    "White_Females.txt",
    "White_Males.txt",
]

# "<identity>/<số ảnh>.jpg" — số ảnh có thể có hoặc không có số 0 đệm đầu
IMAGE_PATH_PATTERN = re.compile(
    r"^(?P<identity>[^/\\]+)[/\\](?P<imagenum>\d+)\.jpe?g$", re.IGNORECASE
)


def fold_name_from_filename(filename: str) -> str:
    """'Asian_Females.txt' -> 'Asian_Females'."""
    return os.path.splitext(filename)[0]


def parse_metadata_line(line: str):
    """Parse 1 dòng dữ liệu của MetaData/<Fold>.txt.

    GIẢ ĐỊNH format: "<db_code> <image_path>" cách nhau bởi whitespace,
    image_path dạng "<identity>/<số>.jpg".

    Trả về dict {db_code, identity, imagenum, image_path} hoặc None nếu
    dòng không hợp lệ (dòng trống, header, sai số token, image_path
    không khớp mẫu <identity>/<số>.jpg).

    NẾU FORMAT THỰC TẾ KHÁC, CHỈ CẦN SỬA HÀM NÀY - phần còn lại của
    script không cần đổi.
    """
    line = line.strip()
    if not line:
        return None
    tokens = line.split()
    if len(tokens) != 2:
        return None
    db_code, image_path = tokens

    # Dòng header "db_code image_path" sẽ rơi vào đây vì "image_path"
    # (token thứ 2) không khớp mẫu <identity>/<số>.jpg -> tự động bị
    # loại, không cần check riêng chuỗi "db_code".
    image_path_norm = image_path.replace("\\", "/")
    m = IMAGE_PATH_PATTERN.match(image_path_norm)
    if m is None:
        return None

    return {
        "db_code": db_code,
        "identity": m.group("identity"),
        "imagenum": int(m.group("imagenum")),
        "image_path": image_path_norm,
    }


def collect_required_images() -> list:
    """Đọc cả 6 file MetaData/<Fold>.txt, gộp thành 1 danh sách record.

    Trả về list các dict: {db_code, identity, imagenum, image_path, fold}.
    Nếu cùng 1 image_path xuất hiện ở >1 fold (không nên xảy ra), giữ
    lần xuất hiện đầu tiên và CẢNH BÁO — dữ liệu nhân khẩu học mâu thuẫn
    nhau cần được người dùng kiểm tra tay, không tự ý chọn.
    """
    all_records = []
    seen_paths = {}  # image_path -> fold (để phát hiện trùng)
    total_bad = 0

    for fname in FOLD_FILES:
        fpath = os.path.join(DEMOGPAIRS_METADATA_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Không tìm thấy {fpath}. Kiểm tra lại DEMOGPAIRS_METADATA_DIR."
            )
        fold = fold_name_from_filename(fname)

        with open(fpath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        print(f"\n5 dòng đầu tiên của {fname} (kiểm tra format bằng mắt):")
        for line in raw_lines[:5]:
            print(f"    {line.rstrip()}")

        n_bad = 0
        n_added = 0
        for line in raw_lines:
            parsed = parse_metadata_line(line)
            if parsed is None:
                if line.strip():
                    n_bad += 1
                continue

            image_path = parsed["image_path"]
            if image_path in seen_paths:
                print(
                    f"    *** CẢNH BÁO trùng ảnh giữa 2 fold: {image_path} "
                    f"đã ở fold '{seen_paths[image_path]}', giờ lại xuất hiện "
                    f"ở fold '{fold}' — bỏ qua lần sau, giữ lần đầu. ***"
                )
                continue

            record = dict(parsed)
            record["fold"] = fold
            all_records.append(record)
            seen_paths[image_path] = fold
            n_added += 1

        print(f"  -> {fname}: {n_added} ảnh hợp lệ, {n_bad} dòng không parse được.")
        total_bad += n_bad

    if total_bad > 0:
        print(
            f"\n*** TỔNG CẢNH BÁO: {total_bad} dòng trên toàn bộ 6 file không "
            f"khớp format giả định 'db_code <identity>/<số>.jpg'. Hãy mở file "
            f"tương ứng kiểm tra và sửa parse_metadata_line() nếu cần. ***\n"
        )

    return all_records


def image_abspath(image_path: str) -> str:
    return os.path.join(DEMOGPAIRS_IMG_DIR, image_path)


def cache_path(identity: str, imagenum: int) -> str:
    # khớp quy ước "{name}_{imagenum:04d}.npy" trong run_ab_paired.py
    return os.path.join(CACHE_DIR, f"{identity}_{imagenum:04d}.npy")


def extract_one(face_processor, adaface, record: dict):
    """Trả (thành_công: bool, status: str). Dùng cache nếu đã có sẵn."""
    out_path = cache_path(record["identity"], record["imagenum"])
    if os.path.exists(out_path):
        return True, "cached"

    img_path = image_abspath(record["image_path"])
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
    os.makedirs(os.path.dirname(IMAGE_METADATA_CSV), exist_ok=True)

    records = collect_required_images()
    print(
        f"\nĐã thu thập {len(records)} ảnh duy nhất cần trích xuất embedding "
        f"trên {len(FOLD_FILES)} fold nhân khẩu học."
    )

    fold_counts = Counter(r["fold"] for r in records)
    print("Phân bố theo fold:", dict(fold_counts))

    print(
        "\nKhởi tạo FaceProcessor + AdaFaceExtractor (dùng đúng pipeline production)..."
    )
    face_processor = FaceProcessor(
        det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7
    )
    adaface = AdaFaceExtractor(device="cuda")

    n_ok, n_skip = 0, 0
    skip_rows = []
    metadata_rows = []

    for i, record in enumerate(records, start=1):
        ok, status = extract_one(face_processor, adaface, record)
        cache_filename = os.path.basename(
            cache_path(record["identity"], record["imagenum"])
        )

        metadata_rows.append(
            [
                record["identity"],
                record["imagenum"],
                record["fold"],
                record["db_code"],
                record["image_path"],
                cache_filename if ok else "",
                status,
            ]
        )

        if ok:
            n_ok += 1
        else:
            skip_rows.append([record["image_path"], status])
            n_skip += 1

        if i % 500 == 0:
            print(f"  ... {i}/{len(records)} (ok={n_ok}, skip={n_skip})")

    with open(SKIPPED_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "reason"])
        writer.writerows(skip_rows)

    with open(IMAGE_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "identity",
                "imagenum",
                "fold",
                "db_code",
                "image_path",
                "cache_filename",
                "status",
            ]
        )
        writer.writerows(metadata_rows)

    print("\n=== HOÀN TẤT TRÍCH XUẤT EMBEDDING DEMOGPAIRS ===")
    print(f"Thành công: {n_ok}")
    print(f"Bị loại: {n_skip} (chi tiết: {SKIPPED_LOG})")
    if n_skip > 0:
        reasons = Counter(r[1] for r in skip_rows)
        print("Lý do bị loại:", dict(reasons))
    print(
        f"Metadata đầy đủ (identity/fold/cache_filename) đã ghi vào:\n  {IMAGE_METADATA_CSV}"
    )
    print(
        "\nTiếp theo: viết scripts/03b_build_pairs_demogpairs.py, đọc "
        "image_metadata.csv (không có pairs.txt sẵn như CPLFW) để tự dựng:\n"
        "  - genuine pairs: cùng 'identity', ghép các ảnh với nhau\n"
        "  - impostor pairs: khác 'identity', nên tách rõ same-fold "
        "(vd Asian_Females vs Asian_Females) và cross-fold (vd Asian_Females "
        "vs Black_Males) để phục vụ audit tương quan nhân khẩu học/FAR."
    )


if __name__ == "__main__":
    main()
