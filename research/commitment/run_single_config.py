"""
run_single_config.py

Chạy ĐÚNG 1 cấu hình benchmark (1 handler duy nhất) rồi thoát hẳn process.
Tách ra riêng khỏi benchmark_real_decode.py vì TF1.x không đảm bảo giải
phóng hết RAM/VRAM giữa các session trong CÙNG 1 process (session.close()
không thu hồi hết bộ nhớ đã cấp phát, đặc biệt với allow_growth=True) — nên
chạy nhiều handler tuần tự trong 1 process có thể cộng dồn bộ nhớ tới khi
tràn, dù mỗi handler đã "del" xong.

Cách chạy (1 lệnh = 1 cấu hình = 1 process = giải phóng sạch khi xong):

    python research/commitment/run_single_config.py --variant baseline --kappa 0.3125 --max-pairs 50 --results-csv research/commitment/logs/results_log.csv
    python research/commitment/run_single_config.py --variant v1 --max-pairs 50 --results-csv research/commitment/logs/results_log.csv
    python research/commitment/run_single_config.py --variant v2 --pool-size 900 --reliability-scores research/commitment/reliability_tune.npy --max-pairs 50 --results-csv research/commitment/logs/results_log.csv

    python research/commitment/run_single_config.py --variant fixed_prefix --results-csv research/commitment/results_log.csv
    python research/commitment/run_single_config.py --variant v1_no_sort --results-csv research/commitment/results_log.csv

In kết quả ra 1 dòng CSV cuối cùng (dễ parse bởi orchestrator):
    RESULT,<label>,<gmr>,<far>,<errors>

Nếu truyền --results-csv, kết quả (kèm timestamp + toàn bộ tham số chạy)
sẽ được APPEND vào file đó — mỗi lần chạy 1 dòng mới, không ghi đè file cũ.
Nhờ vậy dù chạy rời rạc nhiều lần trong nhiều ngày, hay chạy qua orchestrator,
lịch sử kết quả vẫn được giữ lại đầy đủ, không cần chạy lại từ đầu để tra cứu.
"""

import argparse
import csv
import os
import sys
from datetime import datetime

# Ép stdout/stderr dùng UTF-8, tránh UnicodeEncodeError khi in tiếng Việt
# trên Windows console (mặc định dùng cp1252, không encode được các ký tự
# có dấu như "ỗ", "ả", "ệ"...). reconfigure() có từ Python 3.7+.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


def load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def load_pairs(pairs_csv: str, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def append_result_csv(csv_path: str, record: dict):
    """Append 1 dòng kết quả vào file CSV log. Tạo file + ghi header nếu
    file chưa tồn tại hoặc rỗng. Không bao giờ ghi đè dữ liệu cũ."""
    fieldnames = [
        "timestamp",
        "label",
        "variant",
        "gmr",
        "far",
        "errors",
        "n_genuine_pairs",
        "n_impostor_pairs",
        "max_pairs",
        "kappa",
        "pool_size",
        "reliability_scores",
        "force_cpu",
        "rs_nsym",
    ]
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

    parent_dir = os.path.dirname(os.path.abspath(csv_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def run_benchmark(handler, genuine_rows, impostor_rows, cache_dir: str, label: str):
    n_gen_ok, n_gen_total = 0, 0
    n_imp_accept, n_imp_total = 0, 0
    n_errors = 0

    for row in genuine_rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            helper_data, mask, key_hash = handler.enroll(e1)
            success = handler.verify(e2, helper_data, mask, key_hash)
            n_gen_total += 1
            if success:
                n_gen_ok += 1
        except Exception as e:
            n_errors += 1
            print(f"  [WARN] lỗi genuine pair ({row}): {e}", file=sys.stderr)

    for row in impostor_rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            helper_data, mask, key_hash = handler.enroll(e1)
            success = handler.verify(e2, helper_data, mask, key_hash)
            n_imp_total += 1
            if success:
                n_imp_accept += 1
        except Exception as e:
            n_errors += 1
            print(f"  [WARN] lỗi impostor pair ({row}): {e}", file=sys.stderr)

    gmr = 100 * n_gen_ok / n_gen_total if n_gen_total else float("nan")
    far = 100 * n_imp_accept / n_imp_total if n_imp_total else float("nan")

    print(
        f"\n=== {label} ===\n"
        f"  GMR (genuine accept)  : {n_gen_ok}/{n_gen_total} = {gmr:.2f}%\n"
        f"  FAR (impostor accept) : {n_imp_accept}/{n_imp_total} = {far:.2f}%\n"
        f"  Lỗi/exception         : {n_errors}"
    )
    print(f"RESULT,{label},{gmr:.4f},{far:.4f},{n_errors}")
    return gmr, far, n_errors, n_gen_total, n_imp_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variant",
        required=True,
        choices=[
            "baseline",
            "v1",
            "v2",
            "fixed_prefix",
            "v1_no_sort",
            "adaptive_quant",
            "reduced_key",
            "rs_erasure",
        ],
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--reliability-scores", default=None, help="Bắt buộc nếu --variant v2"
    )
    ap.add_argument(
        "--pool-size", type=int, default=None, help="Bắt buộc nếu --variant v2"
    )
    ap.add_argument(
        "--kappa", type=float, default=0.3125, help="Chỉ dùng cho --variant baseline"
    )
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument(
        "--force-cpu",
        action="store_true",
        help="Ép chạy CPU dù có GPU (tránh OOM VRAM)",
    )
    ap.add_argument(
        "--results-csv",
        default=None,
        help="Nếu set, APPEND kết quả (kèm timestamp + tham số) vào file CSV này. "
        "File sẽ được tạo mới nếu chưa tồn tại, không bao giờ bị ghi đè.",
    )
    ap.add_argument(
        "--rs-nsym",
        type=int,
        default=4,
        help="Số symbol ECC của Reed-Solomon (= sức sửa erasure tối đa). "
        "Chỉ dùng cho --variant rs_erasure. Mặc định 4 (RS(20,16), khớp "
        "đúng key_length=160 bit hiện tại với secret_bytes=16).",
    )
    args = ap.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")

    genuine_rows = load_pairs(
        os.path.join(pairs_dir, "tune_genuine.csv"), args.max_pairs
    )
    impostor_rows = load_pairs(
        os.path.join(pairs_dir, "tune_impostor.csv"), args.max_pairs
    )
    print(
        f"Genuine pairs dùng: {len(genuine_rows)} | Impostor pairs dùng: {len(impostor_rows)}"
    )

    if args.variant == "baseline":
        from wifakey_module.wifakey_handler import WiFaKeyHandler

        handler = WiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
        handler.kappa = args.kappa
        label = f"baseline_and_mask(kappa={args.kappa})"

    elif args.variant == "v1":
        from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler

        handler = SecureWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
        label = "v1_uniform_selection"

    elif args.variant == "fixed_prefix":
        from research.commitment.diagnostic_fixed_prefix import (
            FixedPrefixWiFaKeyHandler,
        )

        handler = FixedPrefixWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
        label = "diagnostic_fixed_prefix"

    elif args.variant == "v1_no_sort":
        from research.commitment.diagnostic_v1_no_sort import (
            NoSortSelectionWiFaKeyHandler,
        )

        handler = NoSortSelectionWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
        label = "diagnostic_v1_no_sort"

    elif args.variant == "adaptive_quant":
        from research.quantizer.v1_adaptive_quantization import (
            AdaptiveQuantizationWiFaKeyHandler,
        )

        handler = AdaptiveQuantizationWiFaKeyHandler(
            data_path=data_dir,
            weights_path=weights_path,
            biases_path=biases_path,
            per_dim_intervals_path="research/quantizer/per_dim_intervals.npy",
        )
        label = "adaptive_quantization"

    elif args.variant == "reduced_key":
        from research.commitment.v1_reduced_key_length import (
            ReducedKeyLengthWiFaKeyHandler,
        )

        handler = ReducedKeyLengthWiFaKeyHandler(
            data_path=data_dir,
            weights_path=weights_path,
            biases_path=biases_path,
            effective_key_length=128,
        )
        label = "reduced_key_128"
    
    elif args.variant == "rs_erasure":
        from research.commitment.v2_rs_erasure import RSErasureWiFaKeyHandler
        handler = RSErasureWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path,
            rs_nsym=args.rs_nsym,
            secret_bytes=(160 - args.rs_nsym * 8) // 8,  # tự suy ra để khớp key_length=160
        )
        label = f"rs_erasure_nsym{args.rs_nsym}"

    else:  # v2
        if not args.reliability_scores or not args.pool_size:
            print(
                "ERROR: --variant v2 cần --reliability-scores và --pool-size",
                file=sys.stderr,
            )
            sys.exit(1)
        from research.commitment.v2_reliability_selection import (
            ReliabilitySelectionWiFaKeyHandler,
        )

        handler = ReliabilitySelectionWiFaKeyHandler(
            data_path=data_dir,
            weights_path=weights_path,
            biases_path=biases_path,
            reliability_scores_path=args.reliability_scores,
            pool_size=args.pool_size,
        )
        label = f"v2_pool_M{args.pool_size}"

    gmr, far, n_errors, n_gen_total, n_imp_total = run_benchmark(
        handler, genuine_rows, impostor_rows, cache_dir, label
    )

    if args.results_csv:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "variant": args.variant,
            "gmr": f"{gmr:.4f}",
            "far": f"{far:.4f}",
            "errors": n_errors,
            "n_genuine_pairs": n_gen_total,
            "n_impostor_pairs": n_imp_total,
            "max_pairs": args.max_pairs if args.max_pairs is not None else "",
            "kappa": args.kappa if args.variant == "baseline" else "",
            "pool_size": args.pool_size if args.variant == "v2" else "",
            "reliability_scores": (
                args.reliability_scores if args.variant == "v2" else ""
            ),
            "force_cpu": args.force_cpu,
            "rs_nsym": args.rs_nsym if args.variant == "rs_erasure" else "",
        }
        try:
            append_result_csv(args.results_csv, record)
            print(f"Đã lưu kết quả vào: {os.path.abspath(args.results_csv)}")
        except Exception as e:
            print(
                f"[WARN] Không ghi được kết quả vào --results-csv: {e}",
                file=sys.stderr,
            )

    # Không cần del/cleanup thủ công gì thêm — process thoát ngay sau đây,
    # OS thu hồi toàn bộ RAM/VRAM chắc chắn 100%.


if __name__ == "__main__":
    main()
