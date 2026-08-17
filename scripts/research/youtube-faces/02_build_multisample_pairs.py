"""
build_multisample_pairs_ytf.py

Build multisample_K{K}_genuine.csv / _impostor.csv cho dataset YTF, dùng
bởi research/commitment/run_multisample_benchmark.py (majority-vote
enrollment C.2). Đọc từ manifest_selection.csv do
04a_extract_embeddings_ytf.py tạo ra (KHÔNG đọc select_genuine.csv của
04b - đó là format khác, dùng cho run_ab_paired.py).

KHÁC VỚI select_genuine.csv/select_impostor.csv (04b):
  - 04b: 1 dòng = 1 cặp (1 ảnh enroll, 1 ảnh verify) -> dùng cho
    run_ab_paired.py (so sánh decoder variant, không liên quan vote).
  - Script này: 1 dòng = 1 TRIAL multisample-enrollment: K ảnh enroll
    (cùng 1 session) + 1 ảnh verify (khác session) -> dùng cho
    run_multisample_benchmark.py (so sánh CÓ vote vs KHÔNG vote).

SCHEMA output (đúng cột run_multisample_benchmark.py đang đọc):
    identity, verify_identity, enroll_cache_filenames, verify_cache_filename

  - enroll_cache_filenames: K tên file "{subject}_{imagenum:04d}.npy" nối
    bằng dấu ';', LUÔN sắp theo THỜI GIAN TĂNG DẦN (frame sớm -> muộn).
  - GENUINE: identity == verify_identity.
  - IMPOSTOR: identity != verify_identity.

Chạy thông thường (cũ):
    python scripts/build_multisample_pairs_ytf.py --k 1 3 5 10 15

Chạy với quy tắc CỐ ĐỊNH 15 embeddings (mới):
    python scripts/build_multisample_pairs_ytf.py --k 3 5 7 --strict-15-step
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

DATASET_NAME = "ytf"

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME)
MANIFEST_CSV = os.path.join(PROCESSED_DIR, "manifest_selection.csv")
PAIRS_DIR = os.path.join(PROCESSED_DIR, "pairs")

IMPOSTOR_PER_GENUINE = 1
RANDOM_SEED = 42
MAX_IMPOSTOR_ATTEMPTS_MULTIPLIER = 20


def cache_filename(subject: str, imagenum) -> str:
    return f"{subject}_{int(imagenum):04d}.npy"


def load_manifest():
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(
            f"Không tìm thấy {MANIFEST_CSV}. Chạy "
            f"scripts/04a_extract_embeddings_ytf.py trước."
        )
    with open(MANIFEST_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r["status"] == "ok"]


def build_pools(rows):
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

    for d in (enroll_by_subject, verify_by_subject, impostor_extra_by_subject):
        for subj in d:
            d[subj].sort()

    return enroll_by_subject, verify_by_subject, impostor_extra_by_subject


def pick_k_evenly(items_sorted, k, strict_15=False):
    """
    Nếu strict_15=True:
      - Bắt buộc danh sách phải có >= 15 phần tử (nếu không đủ -> trả về [] để bỏ qua).
      - k = 7: lấy các vị trí 1, 3, 5, 7, 9, 11, 13 (index 0, 2, 4, 6, 8, 10, 12)
      - k = 5: lấy các vị trí 1, 4, 7, 10, 13    (index 0, 3, 6, 9, 12)
      - k = 3: lấy các vị trí 1, 6, 11          (index 0, 5, 10)
    Nếu strict_15=False:
      - Giữ nguyên logic cũ (chia đều linh hoạt theo k bất kỳ).
    """
    n = len(items_sorted)

    if strict_15:
        if n < 15:
            return []  # Không đủ 15 embeddings -> Bỏ qua

        if k == 7:
            indices = [0, 2, 4, 6, 8, 10, 12]
        elif k == 5:
            indices = [0, 3, 6, 9, 12]
        elif k == 3:
            indices = [0, 5, 10]
        else:
            # Nếu truyền K khác 3, 5, 7 khi dùng strict-15, dùng 15 item đầu và pick evenly
            items_15 = items_sorted[:15]
            idx = sorted(set(round(i) for i in _linspace(0, 14, k)))
            return [items_15[i] for i in idx]

        return [items_sorted[i] for i in indices]

    # LOGIC CŨ (Default):
    if n <= k:
        return list(items_sorted)
    idx = sorted(set(round(i) for i in _linspace(0, n - 1, k)))
    if len(idx) < k:
        remaining = [i for i in range(n) if i not in idx]
        idx.extend(remaining[: k - len(idx)])
        idx = sorted(idx)
    return [items_sorted[i] for i in idx[:k]]


def _linspace(start, stop, num):
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def build_genuine_trials(enroll_by_subject, verify_by_subject, k, strict_15=False):
    rows = []
    skipped_not_enough_frames = []
    for subj, verify_list in verify_by_subject.items():
        enroll_full = enroll_by_subject.get(subj, [])

        # Kiểm tra điều kiện đầu vào
        if (strict_15 and len(enroll_full) < 15) or (
            not strict_15 and len(enroll_full) < k
        ):
            skipped_not_enough_frames.append((subj, len(enroll_full), k))
            continue

        enroll_subset = pick_k_evenly(enroll_full, k, strict_15=strict_15)
        if not enroll_subset:
            skipped_not_enough_frames.append((subj, len(enroll_full), k))
            continue

        enroll_filenames = ";".join(cache_filename(subj, i) for i in enroll_subset)
        for verify_imagenum in verify_list:
            rows.append(
                {
                    "identity": subj,
                    "verify_identity": subj,
                    "enroll_cache_filenames": enroll_filenames,
                    "verify_cache_filename": cache_filename(subj, verify_imagenum),
                }
            )
    return rows, skipped_not_enough_frames


def build_impostor_side_pools(
    enroll_by_subject, verify_by_subject, impostor_extra_by_subject
):
    side_a = defaultdict(list)
    side_b = defaultdict(list)
    for subj, imgs in enroll_by_subject.items():
        side_a[subj].extend(imgs)
    for subj, imgs in verify_by_subject.items():
        side_b[subj].extend(imgs)
    for subj, imgs in impostor_extra_by_subject.items():
        side_a[subj].extend(imgs)
        side_b[subj].extend(imgs)
    for d in (side_a, side_b):
        for subj in d:
            d[subj] = sorted(set(d[subj]))
    return side_a, side_b


def build_impostor_trials(
    side_a_by_subject, side_b_by_subject, k, n_target, strict_15=False
):
    random.seed(RANDOM_SEED + k)

    # Lọc danh sách subject thỏa mãn số lượng frame
    min_required = 15 if strict_15 else k
    subjects_a = [s for s, v in side_a_by_subject.items() if len(v) >= min_required]
    subjects_b = [s for s, v in side_b_by_subject.items() if v]

    if len(subjects_a) < 1 or len(subjects_b) < 2:
        print(
            f"  [K={k}] CẢNH BÁO: không đủ subject để sinh impostor trial "
            f"(subjects_a={len(subjects_a)}, subjects_b={len(subjects_b)})."
        )
        return []

    rows = []
    seen = set()
    max_attempts = max(n_target * MAX_IMPOSTOR_ATTEMPTS_MULTIPLIER, 1000)
    attempts = 0

    while len(rows) < n_target and attempts < max_attempts:
        attempts += 1
        subj_a = random.choice(subjects_a)
        subj_b = random.choice(subjects_b)
        if subj_a == subj_b:
            continue

        enroll_full = side_a_by_subject[subj_a]
        enroll_subset = tuple(pick_k_evenly(enroll_full, k, strict_15=strict_15))
        if not enroll_subset:
            continue

        verify_imagenum = random.choice(side_b_by_subject[subj_b])

        key = (subj_a, enroll_subset, subj_b, verify_imagenum)
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "identity": subj_a,
                "verify_identity": subj_b,
                "enroll_cache_filenames": ";".join(
                    cache_filename(subj_a, i) for i in enroll_subset
                ),
                "verify_cache_filename": cache_filename(subj_b, verify_imagenum),
            }
        )

    if len(rows) < n_target:
        print(
            f"  [K={k}] CẢNH BÁO: chỉ sinh được {len(rows)}/{n_target} "
            f"impostor trial sau {attempts} lần thử."
        )

    return rows


def write_csv(path, rows):
    fieldnames = [
        "identity",
        "verify_identity",
        "enroll_cache_filenames",
        "verify_cache_filename",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 15],
        help="Danh sách các giá trị K cần build (vd --k 1 3 5 10 15)",
    )
    ap.add_argument(
        "--strict-15-step",
        action="store_true",
        help="Yêu cầu tối thiểu 15 embeddings và trích xuất đúng theo bước cố định (1,3,5,7... hoặc 1,4,7... hoặc 1,6,11).",
    )
    args = ap.parse_args()

    rows = load_manifest()
    print(f"Đã đọc {len(rows)} dòng manifest có status=ok.")

    enroll_by_subject, verify_by_subject, impostor_extra_by_subject = build_pools(rows)
    print(
        f"Subject có enroll: {len(enroll_by_subject)} | "
        f"có verify (cross-session): {len(verify_by_subject)} | "
        f"chỉ 1 session (impostor_extra): {len(impostor_extra_by_subject)}"
    )

    side_a, side_b = build_impostor_side_pools(
        enroll_by_subject, verify_by_subject, impostor_extra_by_subject
    )

    os.makedirs(PAIRS_DIR, exist_ok=True)

    for k in sorted(args.k):
        genuine_rows, skipped = build_genuine_trials(
            enroll_by_subject, verify_by_subject, k, strict_15=args.strict_15_step
        )
        n_target_impostor = len(genuine_rows) * IMPOSTOR_PER_GENUINE
        impostor_rows = build_impostor_trials(
            side_a, side_b, k, n_target_impostor, strict_15=args.strict_15_step
        )

        # Đổi tên file output nếu bật strict-15 để tránh ghi đè file cũ
        suffix = "_strict15" if args.strict_15_step else ""
        genuine_path = os.path.join(PAIRS_DIR, f"multisample_K{k}{suffix}_genuine.csv")
        impostor_path = os.path.join(
            PAIRS_DIR, f"multisample_K{k}{suffix}_impostor.csv"
        )

        write_csv(genuine_path, genuine_rows)
        write_csv(impostor_path, impostor_rows)

        print(
            f"\n[K={k}{' (strict-15)' if args.strict_15_step else ''}] genuine: {len(genuine_rows)} -> {genuine_path}"
            f" | impostor: {len(impostor_rows)} -> {impostor_path}"
        )
        if skipped:
            req_text = "15" if args.strict_15_step else str(k)
            print(
                f"  Bỏ qua {len(skipped)} subject không đủ {req_text} ảnh enrollment."
            )


if __name__ == "__main__":
    main()
