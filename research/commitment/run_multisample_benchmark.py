"""
run_multisample_benchmark.py

Benchmark thật (qua Neural-MS decode) cho majority-vote enrollment (C.2)
trên DemogPairs, đọc multisample_K{K}_genuine.csv / _impostor.csv (sinh bởi
build_multisample_pairs_demogpairs.py).

QUAN TRỌNG — thiết kế để so sánh CÔNG BẰNG, không lẫn với số 42.45% đo trên
LFW (khác dataset/domain): script này chạy CẢ 2 chế độ trên CÙNG 1 tập dữ
liệu DemogPairs vừa build:
  --mode vote   : enroll bằng majority-vote trên K ảnh (MultisampleWiFaKeyHandler)
  --mode single : enroll CHỈ bằng 1 ảnh đầu tiên trong K ảnh đó (bỏ qua vote)
                  -> đây là baseline "không multisample" nhưng cùng identity,
                  cùng ảnh verify, cùng pool dữ liệu -- cô lập đúng 1 biến
                  duy nhất: có vote hay không.

Cách chạy (mỗi lệnh 1 process riêng, tránh cộng dồn RAM/VRAM giữa các
handler như đã rút kinh nghiệm trước đó):

    python research/commitment/run_multisample_benchmark.py --mode single --k 3 --results-csv research/commitment/results_multisample.csv
    python research/commitment/run_multisample_benchmark.py --mode vote   --k 3 --results-csv research/commitment/results_multisample.csv
"""

import argparse
import csv
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from research.commitment.v1_multisample import MultisampleWiFaKeyHandler

IMPOSTOR_PER_GENUINE = 20

def load_embedding(cache_dir: str, cache_filename: str) -> np.ndarray:
    return np.load(os.path.join(cache_dir, cache_filename))


def load_rows(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_result_csv(csv_path: str, record: dict):
    fieldnames = list(record.keys())
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    parent_dir = os.path.dirname(os.path.abspath(csv_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def get_enroll_embeddings(row: dict, cache_dir: str, mode: str):
    filenames = row["enroll_cache_filenames"].split(";")
    if mode == "single":
        filenames = filenames[:1]  # CHỈ ảnh đầu tiên -- baseline không vote
    return [load_embedding(cache_dir, fn) for fn in filenames]


def run_genuine(handler, rows, cache_dir: str, mode: str):
    ok, total, errors = 0, 0, 0
    for row in rows:
        try:
            enroll_embs = get_enroll_embeddings(row, cache_dir, mode)
            verify_emb = load_embedding(cache_dir, row["verify_cache_filename"])
            helper_data, mask, key_hash = handler.enroll_multisample(enroll_embs)
            success = handler.verify(verify_emb, helper_data, mask, key_hash)
            total += 1
            if success:
                ok += 1
        except Exception as e:
            errors += 1
            print(f"  [WARN] genuine ({row.get('identity')}): {e}", file=sys.stderr)
    return ok, total, errors


def run_impostor(handler, rows, cache_dir: str, mode: str):
    accept, total, errors = 0, 0, 0
    for row in rows:
        try:
            enroll_embs = get_enroll_embeddings(row, cache_dir, mode)
            verify_emb = load_embedding(cache_dir, row["verify_cache_filename"])
            helper_data, mask, key_hash = handler.enroll_multisample(enroll_embs)
            success = handler.verify(verify_emb, helper_data, mask, key_hash)
            total += 1
            if success:
                accept += 1
        except Exception as e:
            errors += 1
            print(
                f"  [WARN] impostor ({row.get('verify_identity')}): {e}",
                file=sys.stderr,
            )
    return accept, total, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["vote", "single"])
    ap.add_argument(
        "--k", type=int, required=True, help="Khớp với K đã dùng lúc build pairs"
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument(
        "--pairs-dir",
        default=None,
        help="Mặc định: <project-root>/datasets/processed/demogpairs/pairs",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Mặc định: <project-root>/datasets/processed/demogpairs/embeddings_cache",
    )
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--results-csv", default=None)
    args = ap.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", "demogpairs", "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", "demogpairs", "embeddings_cache"
    )
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")

    genuine_rows = load_rows(
        os.path.join(pairs_dir, f"multisample_K{args.k}_genuine.csv")
    )
    impostor_rows = load_rows(
        os.path.join(pairs_dir, f"multisample_K{args.k}_impostor.csv")
    )
    if args.max_pairs:
        genuine_rows = genuine_rows[: args.max_pairs]
        impostor_rows = impostor_rows[: args.max_pairs]

    print(
        f"mode={args.mode}, K={args.k} | genuine trials: {len(genuine_rows)} | impostor trials: {len(impostor_rows)}"
    )

    handler = MultisampleWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    gen_ok, gen_total, gen_err = run_genuine(
        handler, genuine_rows, cache_dir, args.mode
    )
    imp_accept, imp_total, imp_err = run_impostor(
        handler, impostor_rows, cache_dir, args.mode
    )

    gmr = 100 * gen_ok / gen_total if gen_total else float("nan")
    far = 100 * imp_accept / imp_total if imp_total else float("nan")

    label = f"{args.mode}_K{args.k}"
    print(
        f"\n=== {label} ===\n"
        f"  GMR: {gen_ok}/{gen_total} = {gmr:.2f}%  (lỗi: {gen_err})\n"
        f"  FAR: {imp_accept}/{imp_total} = {far:.2f}%  (lỗi: {imp_err})"
    )
    print(f"RESULT,{label},{gmr:.4f},{far:.4f}")

    if args.results_csv:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "impostor_per_genuine": IMPOSTOR_PER_GENUINE,
            "mode": args.mode,
            "k": args.k,
            "gmr": f"{gmr:.4f}",
            "far": f"{far:.4f}",
            "n_genuine": gen_total,
            "n_impostor": imp_total,
            "errors": gen_err + imp_err,
        }
        try:
            append_result_csv(args.results_csv, record)
            print(f"Đã lưu vào: {os.path.abspath(args.results_csv)}")
        except Exception as e:
            print(f"[WARN] Không ghi được --results-csv: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
