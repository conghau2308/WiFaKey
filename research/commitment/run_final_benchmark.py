"""
run_final_benchmark.py

Benchmark toàn tập, hỗ trợ cả baseline (SecureWiFaKeyHandler) và
EmpiricalMultiStartHandler. Xuất JSON per‑pair tương thích với compare_mcnemar.py.

Cách dùng:
  # Baseline (BPSK cứng)
  python research/commitment/run_final_benchmark.py --baseline --label bpsk --output results/baseline_lfw.json

  # Empirical + Multi‑start
  python research/commitment/run_final_benchmark.py --lookup <path> --K 5 --sigma 0.2 --label emp_multi --output results/emp_multi_lfw.json
"""

import argparse
import csv
import json
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.commitment.v4_empirical_multistart_handler import (
    EmpiricalMultiStartHandler,
)


def load_embedding(cache_dir, name, imagenum):
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def load_pairs(pairs_csv, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def pairs_iter(pairs_csv, cache_dir, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2, row
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Dùng SecureWiFaKeyHandler (BPSK cứng) thay vì Empirical+Multi",
    )
    ap.add_argument("--lookup", default=None, help="Path reliability_lookup.npz")
    ap.add_argument("--K", type=int, default=5, help="Multi‑start K (0 = tắt)")
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument(
        "--label", default=None, help="Tên phương pháp (nếu không truyền sẽ tự tạo)"
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--output", required=True, help="File JSON kết quả")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "embeddings_cache"
    )

    genuine_csv = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")
    impostor_csv = os.path.join(pairs_dir, f"{args.tier}_impostor.csv")

    # Khởi tạo handler phù hợp
    if args.baseline:
        print("Khởi tạo handler BASELINE (BPSK cứng)...")
        handler = SecureWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
        default_label = "bpsk_baseline"
    else:
        print("Khởi tạo handler Empirical + Multi‑start...")
        handler = EmpiricalMultiStartHandler(
            lookup_path=args.lookup,
            multi_start_K=args.K,
            multi_start_sigma=args.sigma,
            data_path=data_dir,
            weights_path=weights_path,
            biases_path=biases_path,
        )
        default_label = f"emp_multi_K{args.K}_s{args.sigma}"

    label = args.label or default_label

    # Định dạng JSON tương thích compare_mcnemar.py
    output_data = {"label": label, "tier": args.tier, "results": {}}
    results_dict = output_data["results"]

    # Genuine pairs
    print("Đánh giá genuine...")
    for feat_enroll, feat_verify, row in pairs_iter(
        genuine_csv, cache_dir, args.max_pairs
    ):
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)
        ok = handler.verify(feat_verify, helper, sel_mask, key_hash)
        pair_id = f"{row['name_enroll']}_{row['imagenum_enroll']}_{row['name_verify']}_{row['imagenum_verify']}"
        results_dict[pair_id] = {"is_genuine": True, "success": bool(ok)}

    # Impostor pairs
    print("Đánh giá impostor...")
    for feat_enroll, feat_verify, row in pairs_iter(
        impostor_csv, cache_dir, args.max_pairs
    ):
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)
        ok = handler.verify(feat_verify, helper, sel_mask, key_hash)
        pair_id = f"{row['name_enroll']}_{row['imagenum_enroll']}_{row['name_verify']}_{row['imagenum_verify']}"
        results_dict[pair_id] = {"is_genuine": False, "success": bool(ok)}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Đã lưu kết quả vào {args.output}")

    # In tóm tắt nhanh
    gen_entries = [v for v in results_dict.values() if v["is_genuine"]]
    imp_entries = [v for v in results_dict.values() if not v["is_genuine"]]
    n_gen = len(gen_entries)
    n_imp = len(imp_entries)
    gmr = sum(e["success"] for e in gen_entries) / n_gen * 100 if n_gen else 0
    far = sum(e["success"] for e in imp_entries) / n_imp * 100 if n_imp else 0
    print(f"GMR: {gmr:.2f}% ({sum(e['success'] for e in gen_entries)}/{n_gen})")
    print(f"FAR: {far:.4f}% ({sum(e['success'] for e in imp_entries)}/{n_imp})")

    handler.sess.close()


if __name__ == "__main__":
    main()
