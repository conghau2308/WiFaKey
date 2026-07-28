"""
compare_mcnemar.py

So sánh PAIRED (McNemar's exact test) giữa 2 file JSON per-pair đã dump từ
run_dump_per_pair.py -- ghép theo CÙNG 1 cặp khuôn mặt cụ thể (xem giải
thích ý nghĩa thống kê của phép ghép cặp này trong docstring của
run_dump_per_pair.py).

Không cần TF/GPU/dataset -- chỉ đọc 2 file JSON, chạy rất nhanh. Logic
mcnemar_exact() lấy đúng từ run_ab_paired.py (cùng công thức, cùng cách in
kết quả) để nhất quán với các so sánh McNemar khác đã có trong project.

Cách chạy:
    python research/commitment/compare_mcnemar.py \
        --file-a research/commitment/logs/per_pair_v1_lfw.json \
        --file-b research/commitment/logs/per_pair_rs8_lfw.json \
        --label-a v1 --label-b rs_erasure_nsym8

    python research/commitment/compare_mcnemar.py \
        --file-a research/commitment/logs/per_pair_v1_cplfw.json \
        --file-b research/commitment/logs/per_pair_rs8_cplfw.json
"""

import argparse
import json
import sys

import numpy as np
from scipy.stats import binomtest


def load_dump(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mcnemar_exact(success_a, success_b, label_a, label_b):
    """Giống hệt công thức trong run_ab_paired.py -- McNemar's exact test
    (binomial 2 phía trên các cặp DISCORDANT)."""
    a = np.array(success_a)
    b = np.array(success_b)

    if len(a) == 0:
        print(f"  [{label_a} vs {label_b}] Không có mẫu nào trong subset -- bỏ qua.")
        return 0, 0, None

    b_count = int(np.sum((a == 1) & (b == 0)))  # A đúng, B sai
    c_count = int(np.sum((a == 0) & (b == 1)))  # A sai, B đúng
    n_discordant = b_count + c_count

    if n_discordant == 0:
        print(
            f"  [{label_a} vs {label_b}] Không có cặp discordant nào -- 2 "
            f"variant THẬT SỰ giống hệt nhau trên tập test này."
        )
        return b_count, c_count, 1.0

    p_value = binomtest(
        min(b_count, c_count), n_discordant, 0.5, alternative="two-sided"
    ).pvalue
    sig = (
        "CÓ ý nghĩa (p<0.05)"
        if p_value < 0.05
        else "KHÔNG có ý nghĩa thống kê (p>=0.05)"
    )
    print(
        f"  [{label_a} vs {label_b}] {label_a} đúng/{label_b} sai: {b_count}   "
        f"{label_a} sai/{label_b} đúng: {c_count}   "
        f"McNemar exact p-value = {p_value:.4f}  -> {sig}"
    )
    return b_count, c_count, p_value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file-a", required=True)
    ap.add_argument("--file-b", required=True)
    ap.add_argument(
        "--label-a", default=None, help="Mặc định lấy 'label' trong file JSON A"
    )
    ap.add_argument(
        "--label-b", default=None, help="Mặc định lấy 'label' trong file JSON B"
    )
    args = ap.parse_args()

    dump_a = load_dump(args.file_a)
    dump_b = load_dump(args.file_b)

    label_a = args.label_a or dump_a["label"]
    label_b = args.label_b or dump_b["label"]

    if dump_a["tier"] != dump_b["tier"]:
        print(
            f"[WARN] 2 file dùng tier khác nhau ({dump_a['tier']} vs "
            f"{dump_b['tier']}) -- so sánh có thể KHÔNG hợp lệ (khác "
            f"dataset/tập test)!",
            file=sys.stderr,
        )

    results_a = dump_a["results"]
    results_b = dump_b["results"]

    keys_a = set(results_a.keys())
    keys_b = set(results_b.keys())
    common_keys = sorted(keys_a & keys_b)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    print(f"=== So sánh {label_a} vs {label_b} (tier={dump_a['tier']}) ===")
    print(f"Cặp chung (dùng để so sánh): {len(common_keys)}")
    if only_a:
        print(
            f"[WARN] {len(only_a)} cặp chỉ có ở file A (vd do --max-pairs "
            f"khác nhau, hoặc lỗi runtime khi dump) -- bị bỏ qua, không so "
            f"sánh được."
        )
    if only_b:
        print(
            f"[WARN] {len(only_b)} cặp chỉ có ở file B -- bị bỏ qua, không "
            f"so sánh được."
        )

    if not common_keys:
        print("[LỖI] Không có cặp chung nào -- không thể so sánh.", file=sys.stderr)
        sys.exit(1)

    # Sanity check: is_genuine phải khớp giữa 2 file cho cùng 1 key -- nếu
    # không khớp, khả năng cao 2 file dump từ 2 dataset/tier khác nhau dù
    # cùng key trùng hợp, hoặc CSV pairs bị đổi giữa 2 lần chạy.
    mismatched = [
        k
        for k in common_keys
        if results_a[k]["is_genuine"] != results_b[k]["is_genuine"]
    ]
    if mismatched:
        print(
            f"[LỖI] {len(mismatched)} cặp có is_genuine KHÔNG khớp giữa 2 "
            f"file -- dữ liệu có thể bị lẫn dataset/tier, KIỂM TRA LẠI "
            f"trước khi tin kết quả bên dưới.",
            file=sys.stderr,
        )

    genuine_keys = [k for k in common_keys if results_a[k]["is_genuine"]]
    impostor_keys = [k for k in common_keys if not results_a[k]["is_genuine"]]

    success_a_gen = [results_a[k]["success"] for k in genuine_keys]
    success_b_gen = [results_b[k]["success"] for k in genuine_keys]
    success_a_imp = [results_a[k]["success"] for k in impostor_keys]
    success_b_imp = [results_b[k]["success"] for k in impostor_keys]

    gmr_a = (
        100 * sum(success_a_gen) / len(success_a_gen) if success_a_gen else float("nan")
    )
    gmr_b = (
        100 * sum(success_b_gen) / len(success_b_gen) if success_b_gen else float("nan")
    )
    far_a = (
        100 * sum(success_a_imp) / len(success_a_imp) if success_a_imp else float("nan")
    )
    far_b = (
        100 * sum(success_b_imp) / len(success_b_imp) if success_b_imp else float("nan")
    )

    print(f"\nTrên {len(genuine_keys)} cặp genuine CHUNG:")
    print(f"  GMR {label_a}: {gmr_a:.2f}%   GMR {label_b}: {gmr_b:.2f}%")
    print("--- McNemar's exact test (genuine pairs, đo GMR/FRR) ---")
    mcnemar_exact(success_a_gen, success_b_gen, label_a, label_b)

    if impostor_keys:
        print(f"\nTrên {len(impostor_keys)} cặp impostor CHUNG:")
        print(f"  FAR {label_a}: {far_a:.2f}%   FAR {label_b}: {far_b:.2f}%")
        print("--- McNemar's exact test (impostor pairs, đo FAR) ---")
        mcnemar_exact(success_a_imp, success_b_imp, label_a, label_b)
    else:
        print("\n(Không có cặp impostor chung -- bỏ qua FAR/McNemar.)")


if __name__ == "__main__":
    main()
