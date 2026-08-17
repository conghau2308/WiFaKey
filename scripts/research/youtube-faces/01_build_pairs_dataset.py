"""
04b_build_pairs_ytf.py

Build select_genuine.csv / select_impostor.csv cho dataset YTF từ
manifest_selection.csv (được tạo bởi 04a_extract_embeddings_ytf.py),
ĐÚNG FORMAT mà run_ab_paired.py đang đọc (cột: name_enroll,
imagenum_enroll, name_verify, imagenum_verify) - giống hệt cách
03b_build_pairs_cplfw.py đã làm cho CPLFW.

LOGIC SINH CẶP:
  - GENUINE: với mỗi subject, mỗi ảnh "enroll" (tối đa 15 ảnh) được ghép
    với mỗi ảnh "verify" (1 ảnh / session khác). Vì enroll và verify luôn
    thuộc 2 session khác nhau (do cách 04a chọn), mọi cặp genuine đều là
    cross-session -> đúng mục tiêu multi-sample enrollment.
    -> Nếu subject có N session verify khác, sẽ có (số enroll) x N dòng
       genuine cho subject đó (tất cả cùng nhãn genuine).
  - IMPOSTOR: ghép ngẫu nhiên 1 ảnh enroll của subject A với 1 ảnh verify
    của subject B (A != B), lấy ngẫu nhiên không lặp, số lượng =
    IMPOSTOR_RATIO x số genuine pairs (mặc định 1:1, ĐANG LÀ GIẢ ĐỊNH vì
    bạn chưa xác nhận tỉ lệ cụ thể - đổi IMPOSTOR_RATIO nếu cần).

CHỈ CHẠY SAU KHI đã chạy xong 04a_extract_embeddings_ytf.py, vì script
này chỉ giữ lại các dòng có status="ok" trong manifest (ảnh đã có
embedding trong cache).

Cách chạy:
    python scripts/04b_build_pairs_ytf.py
"""

import os
import sys
import csv
import random
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

DATASET_NAME = "ytf"

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME)
MANIFEST_CSV = os.path.join(PROCESSED_DIR, "manifest_selection.csv")
PAIRS_DIR = os.path.join(PROCESSED_DIR, "pairs")
GENUINE_CSV = os.path.join(PAIRS_DIR, "select_genuine.csv")
IMPOSTOR_CSV = os.path.join(PAIRS_DIR, "select_impostor.csv")

# GIẢ ĐỊNH tỉ lệ 1:1 vì bạn chưa xác nhận - đổi nếu cần (vd 3.0 = 3x genuine)
IMPOSTOR_RATIO = 1.0
RANDOM_SEED = 42
MAX_IMPOSTOR_ATTEMPTS_MULTIPLIER = 20  # tránh vòng lặp vô hạn nếu pool nhỏ


def load_manifest():
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(
            f"Không tìm thấy {MANIFEST_CSV}. Chạy "
            f"scripts/04a_extract_embeddings_ytf.py trước."
        )
    with open(MANIFEST_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["status"] == "ok"]
    return rows


def build_pools(rows):
    """Trả về 3 dict: enroll_by_subject, verify_by_subject,
    impostor_extra_by_subject (subject chỉ có 1 session, chỉ dùng cho
    impostor). Mỗi giá trị là list các imagenum (int) đã extract thành
    công."""
    enroll_by_subject = defaultdict(list)
    verify_by_subject = defaultdict(list)
    impostor_extra_by_subject = defaultdict(list)
    for r in rows:
        subj = r["subject"]
        imagenum = int(r["imagenum"])
        role = r["role"]
        if role == "enroll":
            enroll_by_subject[subj].append(imagenum)
        elif role == "verify":
            verify_by_subject[subj].append(imagenum)
        elif role == "impostor_extra":
            impostor_extra_by_subject[subj].append(imagenum)
    return enroll_by_subject, verify_by_subject, impostor_extra_by_subject


def build_impostor_side_pools(
    enroll_by_subject, verify_by_subject, impostor_extra_by_subject
):
    """Gộp pool cho impostor pair: phía A (đóng vai enroll) = ảnh enroll
    bình thường + ảnh impostor_extra; phía B (đóng vai verify) = ảnh
    verify bình thường + ảnh impostor_extra. Nhờ vậy subject chỉ có 1
    session vẫn tham gia impostor pair ở CẢ HAI phía, thay vì bị bỏ hẳn
    như code cũ."""
    side_a = defaultdict(list)
    side_b = defaultdict(list)
    for subj, imgs in enroll_by_subject.items():
        side_a[subj].extend(imgs)
    for subj, imgs in verify_by_subject.items():
        side_b[subj].extend(imgs)
    for subj, imgs in impostor_extra_by_subject.items():
        side_a[subj].extend(imgs)
        side_b[subj].extend(imgs)
    return side_a, side_b


def build_genuine_pairs(enroll_by_subject, verify_by_subject):
    genuine_rows = []
    for subj, verify_list in verify_by_subject.items():
        enroll_list = enroll_by_subject.get(subj, [])
        if not enroll_list:
            continue
        for enroll_imagenum in enroll_list:
            for verify_imagenum in verify_list:
                genuine_rows.append(
                    {
                        "name_enroll": subj,
                        "imagenum_enroll": enroll_imagenum,
                        "name_verify": subj,
                        "imagenum_verify": verify_imagenum,
                    }
                )
    return genuine_rows


def build_impostor_pairs(side_a_by_subject, side_b_by_subject, n_target):
    random.seed(RANDOM_SEED)
    subjects_with_enroll = [s for s, v in side_a_by_subject.items() if v]
    subjects_with_verify = [s for s, v in side_b_by_subject.items() if v]

    if len(subjects_with_enroll) < 2 or len(subjects_with_verify) < 2:
        print(
            "CẢNH BÁO: không đủ subject để sinh impostor pair đa dạng "
            "(cần >=2 subject có ảnh enroll và >=2 subject có ảnh verify)."
        )
        return []

    impostor_rows = []
    seen_pairs = set()
    max_attempts = max(n_target * MAX_IMPOSTOR_ATTEMPTS_MULTIPLIER, 1000)
    attempts = 0

    while len(impostor_rows) < n_target and attempts < max_attempts:
        attempts += 1
        subj_a = random.choice(subjects_with_enroll)
        subj_b = random.choice(subjects_with_verify)
        if subj_a == subj_b:
            continue

        enroll_imagenum = random.choice(side_a_by_subject[subj_a])
        verify_imagenum = random.choice(side_b_by_subject[subj_b])

        key = (subj_a, enroll_imagenum, subj_b, verify_imagenum)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        impostor_rows.append(
            {
                "name_enroll": subj_a,
                "imagenum_enroll": enroll_imagenum,
                "name_verify": subj_b,
                "imagenum_verify": verify_imagenum,
            }
        )

    if len(impostor_rows) < n_target:
        print(
            f"CẢNH BÁO: chỉ sinh được {len(impostor_rows)}/{n_target} impostor "
            f"pair sau {attempts} lần thử (pool subject có thể quá nhỏ hoặc "
            f"đã hết cặp chưa trùng)."
        )

    return impostor_rows


def main():
    rows = load_manifest()
    print(f"Đã đọc {len(rows)} dòng manifest có status=ok.")

    enroll_by_subject, verify_by_subject, impostor_extra_by_subject = build_pools(rows)
    n_subjects_valid = len(
        set(enroll_by_subject.keys()) & set(verify_by_subject.keys())
    )
    print(f"Số subject có đủ cả enroll và verify sau extraction: {n_subjects_valid}")
    print(
        f"Số subject chỉ có 1 session (impostor_extra, không dùng cho "
        f"genuine): {len(impostor_extra_by_subject)}"
    )

    genuine_rows = build_genuine_pairs(enroll_by_subject, verify_by_subject)
    n_target_impostor = int(round(len(genuine_rows) * IMPOSTOR_RATIO))

    impostor_side_a, impostor_side_b = build_impostor_side_pools(
        enroll_by_subject, verify_by_subject, impostor_extra_by_subject
    )
    impostor_rows = build_impostor_pairs(
        impostor_side_a, impostor_side_b, n_target_impostor
    )

    os.makedirs(PAIRS_DIR, exist_ok=True)
    fieldnames = ["name_enroll", "imagenum_enroll", "name_verify", "imagenum_verify"]

    with open(GENUINE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(genuine_rows)

    with open(IMPOSTOR_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(impostor_rows)

    print("\n=== HOÀN TẤT BUILD PAIRS YTF ===")
    print(f"Genuine pairs : {len(genuine_rows)}  -> {GENUINE_CSV}")
    print(f"Impostor pairs: {len(impostor_rows)}  -> {IMPOSTOR_CSV}")
    print("\nTiếp theo: chạy " "'python experiments/run_ab_paired.py --dataset ytf'")
    print(
        "\nLƯU Ý: để phân tích ảnh hưởng SỐ LƯỢNG ảnh enrollment (vd so "
        "sánh 1 vs 5 vs 15 ảnh), bạn có thể lọc select_genuine.csv theo "
        "thứ tự imagenum trong từng nhóm (name_enroll, name_verify) và "
        "chỉ lấy K dòng đầu / random K dòng mỗi nhóm, thay vì tạo lại "
        "toàn bộ manifest - miễn K <= 15."
    )


if __name__ == "__main__":
    main()
