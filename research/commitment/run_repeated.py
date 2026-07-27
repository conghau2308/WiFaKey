"""
run_repeated.py

Chạy LẶP LẠI N lần cùng 1 cấu hình (qua run_single_config.py) để đo độ ổn
định thật của GMR/FAR -- cần thiết vì đã quan sát nhiễu run-to-run đáng kể
(vd nsym=12 chạy 2 lần lệch 3.6 điểm %: 49.94% vs 53.58%), do enroll() mỗi
lần sinh true_secret + selection_indices ngẫu nhiên mới.

MỖI LẦN CHẠY LÀ 1 SUBPROCESS RIÊNG (gọi run_single_config.py qua subprocess,
không import/chạy trong cùng process) -- giữ đúng nguyên tắc cách ly VRAM
TF1.x đã ghi trong docstring gốc của run_single_config.py. KHÔNG sửa gì ở
run_single_config.py hay các handler -- chỉ là lớp điều phối bên ngoài.

Cách chạy:
    python research/commitment/run_repeated.py --variant rs_erasure --rs-nsym 8 --repeats 5 --max-pairs 200
    python research/commitment/run_repeated.py --variant rs_erasure --rs-nsym 10 --repeats 5 --results-csv research/commitment/logs/results_log.csv
    python research/commitment/run_repeated.py --variant v1 --repeats 5   # để so mức nhiễu nền của chính v1 (không có RS, thuần selection ngẫu nhiên)

In ra: từng lần chạy (GMR/FAR), rồi mean +/- std, min, max ở cuối. Nếu
truyền --results-csv, mỗi lần chạy con vẫn được log riêng vào đó như bình
thường (không đổi hành vi cũ của run_single_config.py) -- script này chỉ
gọi lại đúng CLI đã có, không viết logic log mới.
"""

import argparse
import re
import subprocess
import sys

import numpy as np

_RESULT_RE = re.compile(r"^RESULT,([^,]+),([\d.]+),([\d.]+),(\d+)$", re.MULTILINE)


def run_once(base_args: list[str]) -> tuple[str, float, float, int]:
    """Gọi run_single_config.py qua subprocess, parse dòng RESULT,... từ
    stdout. Trả về (label, gmr, far, n_errors). Raise nếu không tìm thấy
    dòng RESULT (vd lỗi crash giữa chừng)."""
    proc = subprocess.run(
        [sys.executable, "research/commitment/run_single_config.py", *base_args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # In lại toàn bộ stdout/stderr của lần chạy con để không mất thông tin
    # debug nếu có warning/lỗi (vd TF deprecation warnings vẫn thấy được).
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    match = _RESULT_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(
            f"Không tìm thấy dòng RESULT trong output (return code "
            f"{proc.returncode}) -- lần chạy có thể đã crash. Xem log ở "
            f"trên để biết chi tiết."
        )
    label, gmr, far, n_errors = match.groups()
    return label, float(gmr), float(far), int(n_errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    # Toàn bộ tham số còn lại được CHUYỂN THẲNG nguyên văn cho
    # run_single_config.py -- không định nghĩa lại ở đây để tránh lệch
    # convention khi run_single_config.py có thêm variant/tham số mới sau
    # này (vd --variant, --rs-nsym, --max-pairs, --results-csv, --symbol-agg
    # nếu bạn đã thêm, --force-cpu, ...).
    args, passthrough_args = ap.parse_known_args()

    if args.repeats < 2:
        print(
            "[WARN] --repeats < 2 thì không đo được std -- nên dùng >= 3, "
            "khuyến nghị 5.",
            file=sys.stderr,
        )

    gmrs, fars = [], []
    label = None
    for i in range(args.repeats):
        print(f"\n{'=' * 20} Lần chạy {i + 1}/{args.repeats} {'=' * 20}")
        label, gmr, far, n_errors = run_once(passthrough_args)
        gmrs.append(gmr)
        fars.append(far)

    gmrs_arr = np.array(gmrs)
    fars_arr = np.array(fars)

    print(f"\n{'=' * 60}")
    print(f"TỔNG HỢP {args.repeats} lần chạy -- {label}")
    print(f"{'=' * 60}")
    print(f"GMR từng lần : {[f'{g:.2f}' for g in gmrs]}")
    print(
        f"GMR mean +/- std : {gmrs_arr.mean():.2f}% +/- {gmrs_arr.std(ddof=1):.2f}% "
        f"(min={gmrs_arr.min():.2f}%, max={gmrs_arr.max():.2f}%)"
    )
    print(f"FAR từng lần : {[f'{f:.2f}' for f in fars]}")
    print(
        f"FAR mean +/- std : {fars_arr.mean():.4f}% +/- {fars_arr.std(ddof=1):.4f}% "
        f"(min={fars_arr.min():.4f}%, max={fars_arr.max():.4f}%)"
    )
    print(
        f"\nGợi ý đọc: nếu std GMR còn lớn (vd > 2 điểm %), nên tăng "
        f"--repeats hoặc tăng --max-pairs (nếu đang giới hạn) để kết quả "
        f"ổn định hơn trước khi dùng cho thesis."
    )


if __name__ == "__main__":
    main()
