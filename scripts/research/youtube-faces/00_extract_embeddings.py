"""
04a_extract_embeddings_ytf.py

Chọn frame enrollment (đa mẫu, "burst") + frame verify (khác session) từ
YouTube Faces DB (frame_images_DB) và trích xuất embedding, phục vụ thực
nghiệm multi-sample enrollment (đo ảnh hưởng số lượng ảnh enrollment lên
độ chính xác fuzzy commitment).

CẤU TRÚC DATASET KỲ VỌNG (sau khi giải nén frame_images_DB.tar.gz):
    frame_images_DB/
        <subject_name>/
            <session_id>/            <- mỗi thư mục con số = 1 video = 1 "phiên"
                <video_id>.<frame_number>.jpg
                ...
            <session_id_2>/
                ...
        ...

QUAN TRỌNG - HÃY KIỂM TRA TRƯỚC KHI CHẠY:
  Mình chưa tự tải/xem trực tiếp frame_images_DB nên không thể xác nhận
  100% cấu trúc thư mục và định dạng tên file. Script này sẽ IN RA cấu
  trúc của 3 subject đầu tiên (số session, số frame mỗi session, vài tên
  file mẫu) TRƯỚC khi chạy extraction thật, để bạn đối chiếu bằng mắt.
  Nếu sai định dạng, chỉ cần sửa hàm parse_frame_number() và
  list_sessions() - phần còn lại của pipeline không cần đổi.

CHIẾN LƯỢC CHỌN FRAME (đã thống nhất):
  - Với mỗi subject có >= MIN_SESSIONS_REQUIRED session (video):
      + Session có nhiều frame nhất -> làm "enrollment session".
        Lấy tối đa NUM_ENROLL_FRAMES (=15) frame, CHIA ĐỀU theo chỉ số
        trong toàn bộ session đó (spacing thích ứng - ưu tiên lấy ĐỦ 15
        ảnh; nếu session có ít hơn 15 frame khả dụng, lấy hết số frame
        đang có, không ép buộc khoảng cách theo giây).
      + Mỗi session còn lại -> lấy đúng 1 frame (frame ở giữa video) làm
        "verify frame" cho phiên xác thực chéo đó.
  - Subject chỉ có 1 session (không có gì để verify chéo): KHÔNG bỏ qua
    hoàn toàn nữa. Vẫn extract tối đa NUM_ENROLL_FRAMES frame (evenly
    spaced, giống cách chọn enrollment) nhưng gắn role riêng
    "impostor_extra". Lý do:
      + Không thể dùng 2 frame trong CÙNG 1 video làm genuine pair, vì
        các frame liên tiếp/gần nhau trong cùng 1 session giống nhau bất
        thường -> sẽ làm genuine score ảo cao (leak thông tin, không
        phản ánh đúng khả năng cross-session của hệ thống thật).
      + Nhưng hoàn toàn dùng tốt cho impostor pair, vì impostor pair so
        sánh 2 NGƯỜI KHÁC NHAU - không cần khác session mới hợp lệ.
    -> Các subject này sẽ được 04b_build_pairs_ytf.py gộp vào pool sinh
       impostor pair (ở cả 2 phía enroll/verify của cặp impostor), không
       bao giờ xuất hiện trong select_genuine.csv.
  - Subject không có frame hợp lệ nào -> BỎ QUA, ghi log riêng.

QUY ƯỚC ĐẶT TÊN (để tương thích run_ab_paired.py, giống cách CPLFW đã làm):
  - "name" = subject_name (tên định danh gốc, KHÔNG đổi giữa các session,
    vì genuine/impostor được xác định qua việc ở trong CSV nào, không
    phải qua so khớp "name").
  - "imagenum" = video_id * 1_000_000 + frame_number (số nguyên duy nhất
    trong phạm vi 1 subject, vẫn suy ngược ra được session + frame gốc).
  - Cache: "{name}_{imagenum:04d}.npy" - giống hệt convention CPLFW.

Cách chạy:
    python scripts/04a_extract_embeddings_ytf.py
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

DATASET_NAME = "ytf"

# CHỈNH LẠI cho khớp nơi bạn giải nén frame_images_DB.tar.gz
YTF_FRAMES_ROOT = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "ytf", "frame_images_DB"
)

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME)
CACHE_DIR = os.path.join(PROCESSED_DIR, "embeddings_cache")
MANIFEST_CSV = os.path.join(PROCESSED_DIR, "manifest_selection.csv")
SKIPPED_LOG = os.path.join(PROCESSED_DIR, "skipped_log.csv")
SKIPPED_SUBJECTS_LOG = os.path.join(PROCESSED_DIR, "skipped_subjects_log.csv")

NUM_ENROLL_FRAMES = 15  # số ảnh enrollment tối đa mỗi subject
MIN_SESSIONS_REQUIRED = 2  # phải có >=2 session mới có cross-session verify

IMG_EXTS = (".jpg", ".jpeg", ".png")

# Regex bắt số cuối cùng trước phần mở rộng, dùng làm frame_number.
# Khớp cả "1.1653.jpg" (frame_number=1653) lẫn "img_00234.jpg" (=234).
_FRAME_NUM_RE = re.compile(r"(\d+)\.\w+$")


def parse_frame_number(filename: str):
    """Trích frame_number từ tên file. Trả None nếu không parse được."""
    m = _FRAME_NUM_RE.search(filename)
    if m is None:
        return None
    return int(m.group(1))


def session_numeric_id(session_id_str: str, fallback_index: int) -> int:
    """Cố gắng parse session_id (tên thư mục) thành số nguyên (video_id).
    Nếu tên thư mục không phải số thuần, dùng fallback_index (thứ tự
    xuất hiện của session đó trong danh sách đã sort theo số)."""
    try:
        return int(session_id_str)
    except ValueError:
        return fallback_index


def numeric_sort_key(session_id_str: str):
    """Sort ưu tiên số: folder '1','2','4','10' sẽ theo đúng thứ tự
    1,2,4,10 thay vì string-sort ('1','10','2','4'). Việc thiếu số thứ tự
    (vd '0,1,4' không có '2,3') không ảnh hưởng - đây chỉ là tên định
    danh, không cần liên tục. Folder không phải số thuần rơi xuống cuối,
    sort theo string để vẫn ổn định (không phụ thuộc thứ tự os.listdir)."""
    try:
        return (0, int(session_id_str))
    except ValueError:
        return (1, session_id_str)


def list_sessions(subject_dir: str):
    """Trả về dict {session_id_str: [(frame_number, filename), ...]}
    đã sort theo frame_number tăng dần. Bỏ qua entry không phải thư mục
    (vd file .labeled_faces.txt nếu lỡ nằm chung cấp)."""
    sessions = {}
    for entry in sorted(os.listdir(subject_dir)):
        session_path = os.path.join(subject_dir, entry)
        if not os.path.isdir(session_path):
            continue
        frames = []
        for fname in os.listdir(session_path):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            frame_num = parse_frame_number(fname)
            if frame_num is None:
                continue
            frames.append((frame_num, fname))
        frames.sort(key=lambda x: x[0])
        if frames:
            sessions[entry] = frames
    return sessions


def pick_evenly_spaced(frames, k):
    """Chọn tối đa k phần tử từ danh sách frames (đã sort), CHIA ĐỀU theo
    chỉ số. Nếu len(frames) <= k, trả về toàn bộ (không đủ k thì lấy hết,
    không cố nhân bản cho đủ số lượng)."""
    n = len(frames)
    if n <= k:
        return list(frames)
    idx = sorted(set(int(round(i)) for i in np.linspace(0, n - 1, k)))
    # np.linspace + round có thể trùng index -> bù thêm cho đủ k
    if len(idx) < k:
        remaining = [i for i in range(n) if i not in idx]
        idx.extend(remaining[: k - len(idx)])
        idx = sorted(idx)
    return [frames[i] for i in idx[:k]]


def preview_structure(n_subjects: int = 3):
    """In cấu trúc vài subject đầu tiên để kiểm tra bằng mắt trước khi
    chạy extraction thật."""
    print("=== KIỂM TRA CẤU TRÚC THƯ MỤC (3 subject đầu tiên) ===")
    subject_names = sorted(
        d
        for d in os.listdir(YTF_FRAMES_ROOT)
        if os.path.isdir(os.path.join(YTF_FRAMES_ROOT, d))
    )
    for subj in subject_names[:n_subjects]:
        sessions = list_sessions(os.path.join(YTF_FRAMES_ROOT, subj))
        print(f"\nSubject: {subj}  ({len(sessions)} session)")
        for sid in sorted(sessions.keys(), key=numeric_sort_key):
            frames = sessions[sid]
            sample_names = [f for _, f in frames[:3]]
            print(f"  Session '{sid}': {len(frames)} frame, vd: {sample_names}")
    print(
        "\n*** Nếu cấu trúc trên KHÔNG khớp thực tế (vd tên file không "
        "trích được frame_number đúng), hãy DỪNG LẠI và sửa "
        "parse_frame_number()/list_sessions() trước khi chạy tiếp. ***\n"
    )
    return subject_names


def build_manifest(subject_names):
    """Với mỗi subject, chọn frame enrollment + verify (hoặc
    impostor_extra nếu chỉ có 1 session) theo chiến lược đã mô tả ở đầu
    file. Trả về (manifest_rows, skipped_subjects)."""
    manifest_rows = []
    skipped_subjects = []

    for subj in subject_names:
        subject_dir = os.path.join(YTF_FRAMES_ROOT, subj)
        sessions = list_sessions(subject_dir)

        if len(sessions) == 0:
            skipped_subjects.append([subj, 0, "no_frames_found"])
            continue

        session_ids_sorted = sorted(sessions.keys(), key=numeric_sort_key)
        session_video_id = {
            sid: session_numeric_id(sid, fallback_index=i + 1)
            for i, sid in enumerate(session_ids_sorted)
        }

        if len(sessions) < MIN_SESSIONS_REQUIRED:
            # Chỉ có 1 session -> không thể tạo genuine pair cross-session
            # (lấy 2 frame trong CÙNG 1 video làm genuine sẽ leak, vì các
            # frame liên tiếp giống nhau bất thường -> genuine score ảo
            # cao). Nhưng vẫn dùng tốt cho impostor pair (khác người thì
            # không cần khác session) -> gắn role riêng "impostor_extra",
            # không bao giờ lọt vào genuine pairs.
            only_sid = session_ids_sorted[0]
            frames = pick_evenly_spaced(sessions[only_sid], NUM_ENROLL_FRAMES)
            video_id = session_video_id[only_sid]
            for frame_number, fname in frames:
                imagenum = video_id * 1_000_000 + frame_number
                manifest_rows.append(
                    {
                        "subject": subj,
                        "role": "impostor_extra",
                        "session_id": only_sid,
                        "video_id": video_id,
                        "frame_number": frame_number,
                        "filename": fname,
                        "rel_path": os.path.join(subj, only_sid, fname),
                        "imagenum": imagenum,
                    }
                )
            continue

        # Session enrollment = session có nhiều frame nhất
        enroll_session_id = max(sessions, key=lambda sid: len(sessions[sid]))
        enroll_frames = pick_evenly_spaced(
            sessions[enroll_session_id], NUM_ENROLL_FRAMES
        )
        enroll_video_id = session_video_id[enroll_session_id]

        for frame_number, fname in enroll_frames:
            imagenum = enroll_video_id * 1_000_000 + frame_number
            manifest_rows.append(
                {
                    "subject": subj,
                    "role": "enroll",
                    "session_id": enroll_session_id,
                    "video_id": enroll_video_id,
                    "frame_number": frame_number,
                    "filename": fname,
                    "rel_path": os.path.join(subj, enroll_session_id, fname),
                    "imagenum": imagenum,
                }
            )

        # Mỗi session còn lại -> 1 frame verify (frame ở giữa video)
        verify_sessions = [
            sid for sid in session_ids_sorted if sid != enroll_session_id
        ]
        for sid in verify_sessions:
            frames = sessions[sid]
            frame_number, fname = frames[len(frames) // 2]
            video_id = session_video_id[sid]
            imagenum = video_id * 1_000_000 + frame_number
            manifest_rows.append(
                {
                    "subject": subj,
                    "role": "verify",
                    "session_id": sid,
                    "video_id": video_id,
                    "frame_number": frame_number,
                    "filename": fname,
                    "rel_path": os.path.join(subj, sid, fname),
                    "imagenum": imagenum,
                }
            )

    return manifest_rows, skipped_subjects


def cache_path(name: str, imagenum: int) -> str:
    return os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy")


def extract_one(face_processor, adaface, name: str, imagenum: int, rel_path: str):
    """Trả (thành_công: bool, status: str). Dùng cache nếu đã có sẵn."""
    out_path = cache_path(name, imagenum)
    if os.path.exists(out_path):
        return True, "cached"

    img_path = os.path.join(YTF_FRAMES_ROOT, rel_path)
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

    if not os.path.isdir(YTF_FRAMES_ROOT):
        raise FileNotFoundError(
            f"Không tìm thấy {YTF_FRAMES_ROOT}. Giải nén frame_images_DB.tar.gz "
            f"và/hoặc sửa lại YTF_FRAMES_ROOT ở đầu file."
        )

    subject_names = preview_structure(n_subjects=3)
    print(f"Tổng số subject phát hiện được: {len(subject_names)}")

    manifest_rows, skipped_subjects = build_manifest(subject_names)
    n_enroll = sum(1 for r in manifest_rows if r["role"] == "enroll")
    n_verify = sum(1 for r in manifest_rows if r["role"] == "verify")
    n_impostor_extra = sum(1 for r in manifest_rows if r["role"] == "impostor_extra")
    n_impostor_extra_subjects = len(
        {r["subject"] for r in manifest_rows if r["role"] == "impostor_extra"}
    )
    n_genuine_capable_subjects = len(
        {r["subject"] for r in manifest_rows if r["role"] in ("enroll", "verify")}
    )
    print(
        f"\nĐã chọn xong: {n_enroll} ảnh enrollment, {n_verify} ảnh verify "
        f"({n_genuine_capable_subjects} subject có thể dùng cho genuine). "
        f"{n_impostor_extra} ảnh impostor_extra từ {n_impostor_extra_subjects} "
        f"subject chỉ có 1 session (không dùng cho genuine, chỉ dùng cho "
        f"impostor). Bỏ qua {len(skipped_subjects)} subject không có frame "
        f"hợp lệ nào."
    )

    with open(SKIPPED_SUBJECTS_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "n_sessions", "reason"])
        writer.writerows(skipped_subjects)

    print("Khởi tạo FaceProcessor + AdaFaceExtractor (đúng pipeline production)...")
    face_processor = FaceProcessor(
        det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7
    )
    adaface = AdaFaceExtractor(device="cuda")

    n_ok, n_skip = 0, 0
    skip_rows = []

    for i, row in enumerate(manifest_rows, start=1):
        ok, status = extract_one(
            face_processor, adaface, row["subject"], row["imagenum"], row["rel_path"]
        )
        row["status"] = "ok" if ok else "skip"
        row["status_detail"] = status
        if ok:
            n_ok += 1
        else:
            skip_rows.append([row["subject"], row["rel_path"], status])
            n_skip += 1

        if i % 200 == 0:
            print(f"  ... {i}/{len(manifest_rows)} (ok={n_ok}, skip={n_skip})")

    # Ghi manifest đầy đủ (bao gồm cả row bị skip, để 04b tự lọc lại và
    # để bạn biết chính xác ảnh nào bị loại vì lý do gì)
    fieldnames = [
        "subject",
        "role",
        "session_id",
        "video_id",
        "frame_number",
        "filename",
        "rel_path",
        "imagenum",
        "status",
        "status_detail",
    ]
    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(SKIPPED_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "rel_path", "reason"])
        writer.writerows(skip_rows)

    print("\n=== HOÀN TẤT TRÍCH XUẤT EMBEDDING YTF ===")
    print(f"Thành công: {n_ok}")
    print(f"Bị loại: {n_skip} (chi tiết: {SKIPPED_LOG})")
    if n_skip > 0:
        reasons = Counter(r[2] for r in skip_rows)
        print("Lý do bị loại:", dict(reasons))
    print(f"Manifest đầy đủ: {MANIFEST_CSV}")
    print(
        "\nTiếp theo: chạy scripts/04b_build_pairs_ytf.py để tạo "
        "select_genuine.csv / select_impostor.csv."
    )


if __name__ == "__main__":
    main()
