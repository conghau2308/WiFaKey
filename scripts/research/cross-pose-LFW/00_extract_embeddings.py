"""
03a_extract_embeddings_cplfw.py

Trích xuất embedding cho dataset CPLFW (Cross-Pose LFW).

CẤU TRÚC DATASET CPLFW (khác LFW gốc và khác face-detection-and-re-id):
  - Không tổ chức theo person/imagenum.jpg, mà là 1 thư mục ẢNH PHẲNG
    (thường tên "images/" trong gói tải về từ whdeng.cn/CPLFW) + 1 file
    "pairs_CPLFW.txt" liệt kê từng cặp ảnh cần verify.
  - Mỗi dòng pairs_CPLFW.txt (GIẢ ĐỊNH, xem ghi chú ở dưới):
        <ten_anh_1> <ten_anh_2> <nhan 0/1>
    nhan=1 -> genuine (cùng người), nhan=0 -> impostor (khác người).

QUAN TRỌNG - HÃY KIỂM TRA TRƯỚC KHI CHẠY:
  Mình không thể tự tải file pairs_CPLFW.txt để xác nhận 100% format dòng,
  nên script này sẽ IN RA 5 dòng đầu tiên của file trước khi parse, và nếu
  có dòng nào không đúng 3 token (tên1, tên2, nhãn) sẽ CẢNH BÁO thay vì
  âm thầm bỏ qua. Nếu số dòng lỗi > 0, hãy mở pairs_CPLFW.txt bằng tay,
  xem đúng format rồi sửa lại hàm parse_pairs_line() bên dưới.

MẸO TƯƠNG THÍCH với run_ab_paired.py (không cần sửa file đó):
  run_ab_paired.py load embedding qua quy ước "{name}_{imagenum:04d}.npy".
  CPLFW không có khái niệm "imagenum" theo person, nên ta dùng chính TÊN
  FILE ẢNH (bỏ đuôi .jpg) làm "name", và luôn gán imagenum=0 cố định.
  -> cache lưu thành "{ten_anh_khong_duoi}_0000.npy".

Yêu cầu chuẩn bị trước khi chạy:
  1. Tải CPLFW từ http://www.whdeng.cn/CPLFW/index.html (link Baidu hoặc
     Google Drive trên trang), giải nén.
  2. Đặt thư mục ảnh gốc (chưa align, để FaceProcessor tự detect+align
     giống pipeline production) vào CPLFW_RAW_IMG_DIR bên dưới, và file
     pairs_CPLFW.txt vào CPLFW_PAIRS_TXT.

Cách chạy:
    python scripts/03a_extract_embeddings_cplfw.py
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

DATASET_NAME = "cplfw"

# CHỈNH LẠI 2 ĐƯỜNG DẪN NÀY cho khớp với nơi bạn giải nén CPLFW
CPLFW_RAW_IMG_DIR = os.path.join(_PROJECT_ROOT, "datasets", "raw", "cplfw", "images")
CPLFW_PAIRS_TXT = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "cplfw", "pairs_CPLFW.txt"
)

CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)
SKIPPED_LOG = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "skipped_log.csv"
)


def image_id_from_filename(filename: str) -> str:
    """Bỏ đuôi mở rộng, dùng làm 'name' duy nhất cho 1 ảnh."""
    return os.path.splitext(filename)[0]


def image_path(filename: str) -> str:
    return os.path.join(CPLFW_RAW_IMG_DIR, filename)


def cache_path(image_id: str) -> str:
    # imagenum luôn = 0 -> khớp quy ước _load_embedding() trong run_ab_paired.py
    return os.path.join(CACHE_DIR, f"{image_id}_0000.npy")


def parse_pairs_line(line: str):
    """Parse 1 dòng của pairs_CPLFW.txt.

    GIẢ ĐỊNH format: "<img1> <img2> <label 0/1>" cách nhau bởi whitespace.
    Trả về (img1, img2, is_genuine) hoặc None nếu dòng không hợp lệ
    (dòng trống, sai số token, label không phải 0/1).

    NẾU FORMAT THỰC TẾ KHÁC (vd 4 cột, hoặc phân tách bằng dấu phẩy),
    CHỈ CẦN SỬA HÀM NÀY - phần còn lại của 2 script không cần đổi.
    """
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


def collect_required_images() -> set:
    """Đọc pairs_CPLFW.txt, thu thập tập hợp filename ảnh cần trích xuất
    embedding (dedup vì 1 ảnh có thể xuất hiện trong nhiều cặp)."""
    if not os.path.exists(CPLFW_PAIRS_TXT):
        raise FileNotFoundError(
            f"Không tìm thấy {CPLFW_PAIRS_TXT}. Tải CPLFW và đặt đúng đường dẫn."
        )

    with open(CPLFW_PAIRS_TXT, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    print("5 dòng đầu tiên của pairs_CPLFW.txt (kiểm tra format bằng mắt):")
    for line in raw_lines[:5]:
        print(f"    {line.rstrip()}")

    required = set()
    n_bad = 0
    for line in raw_lines:
        parsed = parse_pairs_line(line)
        if parsed is None:
            if line.strip():  # bỏ qua dòng trống, chỉ đếm dòng lỗi thật sự
                n_bad += 1
            continue
        img1, img2, _ = parsed
        required.add(img1)
        required.add(img2)

    if n_bad > 0:
        print(
            f"\n*** CẢNH BÁO: {n_bad} dòng không parse được theo format giả định "
            f"'<img1> <img2> <0/1>'. Hãy mở pairs_CPLFW.txt kiểm tra và sửa "
            f"hàm parse_pairs_line() nếu cần. ***\n"
        )

    return required


def extract_one(face_processor, adaface, filename: str):
    """Trả (thành_công: bool, status: str). Dùng cache nếu đã có sẵn."""
    image_id = image_id_from_filename(filename)
    out_path = cache_path(image_id)
    if os.path.exists(out_path):
        return True, "cached"

    img_path = image_path(filename)
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

    required_images = collect_required_images()
    print(
        f"\nĐã thu thập {len(required_images)} ảnh duy nhất cần trích xuất embedding."
    )

    print(
        "Khởi tạo FaceProcessor + AdaFaceExtractor (dùng đúng pipeline production)..."
    )
    face_processor = FaceProcessor(
        det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7
    )
    adaface = AdaFaceExtractor(device="cuda")

    n_ok, n_skip = 0, 0
    skip_rows = []

    for i, filename in enumerate(sorted(required_images), start=1):
        ok, status = extract_one(face_processor, adaface, filename)
        if ok:
            n_ok += 1
        else:
            skip_rows.append([filename, status])
            n_skip += 1

        if i % 200 == 0:
            print(f"  ... {i}/{len(required_images)} (ok={n_ok}, skip={n_skip})")

    with open(SKIPPED_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "reason"])
        writer.writerows(skip_rows)

    print("\n=== HOÀN TẤT TRÍCH XUẤT EMBEDDING CPLFW ===")
    print(f"Thành công: {n_ok}")
    print(f"Bị loại: {n_skip} (chi tiết: {SKIPPED_LOG})")
    if n_skip > 0:
        reasons = Counter(r[1] for r in skip_rows)
        print("Lý do bị loại:", dict(reasons))
    print(
        "\nTiếp theo: chạy scripts/03b_build_pairs_cplfw.py để tạo "
        "select_genuine.csv / select_impostor.csv (sẽ tự động BỎ QUA các "
        "cặp có ảnh bị loại ở bước này)."
    )


if __name__ == "__main__":
    main()
