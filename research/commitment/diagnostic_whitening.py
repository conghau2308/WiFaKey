"""
diagnostic_whitening_v2.py

Phiên bản tiết kiệm bộ nhớ: chạy tuần tự baseline và whitening,
mỗi lần chỉ dùng một TF session duy nhất. Tránh OOM trên GPU 2GB.
"""

import argparse
import csv
import hashlib
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from wifakey_module.wifakey_lib import Modulation


# Loaders (giữ nguyên)
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


def genuine_pairs_iter(pairs_csv, cache_dir, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


def compute_whitening_matrix(embedding_list):
    X = np.stack(embedding_list, axis=0)
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    U, S, Vt = np.linalg.svd(cov, full_matrices=False)
    inv_sqrt_S = np.diag(1.0 / np.sqrt(np.maximum(S, 1e-6)))
    W = U @ inv_sqrt_S @ U.T
    return W, mean


def evaluate_gmr(handler, pairs_iter, max_pairs, whiten_params=None, emp_mod=None):
    """
    Chạy đánh giá GMR với handler hiện tại.
    whiten_params: tuple (W, mean) hoặc None.
    """
    n_total = 0
    n_pass = 0

    for feat_enroll, feat_verify in pairs_iter:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        # Tiền xử lý whitening nếu có
        if whiten_params is not None:
            W, mean = whiten_params
            vec_enroll = np.dot(feat_enroll - mean, W)
            vec_verify = np.dot(feat_verify - mean, W)
        else:
            vec_enroll = feat_enroll
            vec_verify = feat_verify

        # Enrollment
        helper, sel_mask, key_hash = handler.enroll(vec_enroll)

        # Verify
        if emp_mod is None:
            ok = handler.verify(vec_verify, helper, sel_mask, key_hash)
        else:
            # Empirical LLR verify
            b_full = handler._binarize_full(vec_verify).astype(np.uint8)
            idx = np.where(sel_mask == 1)[0]
            b_sel = b_full[idx]
            noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
            _, margin_all = binarize_with_perbit_confidence(
                np.dot(vec_verify, handler.M_matrix), handler.intervals
            )
            margin_sel = margin_all[idx]
            llr_emp = emp_mod.modulate(noisy, context={"margin": margin_sel})
            llr_emp = llr_emp.reshape(1, handler.N, handler.Z)
            y_pred = handler.sess.run(
                handler.decoder_output, feed_dict={handler.xa: llr_emp}
            )
            y_pred = y_pred.flatten()
            decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
            ok = hashlib.sha256(decoded_key.tobytes()).digest() == key_hash

        if ok:
            n_pass += 1

    return n_total, n_pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--num-whiten-samples", type=int, default=1000)
    ap.add_argument("--empirical-lookup", default=None)
    ap.add_argument(
        "--cpu", action="store_true", help="Chạy trên CPU để tiết kiệm VRAM"
    )
    args = ap.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")

    # Whitening matrix (từ LFW)
    lfw_pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    lfw_cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )
    lfw_csv = os.path.join(lfw_pairs_dir, "tune_genuine.csv")

    print(f"Lấy {args.num_whiten_samples} embedding từ LFW để tính whitening...")
    whiten_embs = []
    count = 0
    for fe, fv in genuine_pairs_iter(
        lfw_csv, lfw_cache_dir, max_pairs=args.num_whiten_samples // 2
    ):
        whiten_embs.append(fe)
        whiten_embs.append(fv)
        count += 1
        if count >= args.num_whiten_samples // 2:
            break
    W, mean = compute_whitening_matrix(whiten_embs)

    # Test set
    test_pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "pairs"
    )
    test_cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "embeddings_cache"
    )
    test_csv = os.path.join(test_pairs_dir, f"{args.tier}_genuine.csv")

    # Empirical LLR nếu có
    emp_mod = None
    if args.empirical_lookup:
        from research.modulation.v2_empirical_llr import EmpiricalLLR

        emp_mod = EmpiricalLLR(lookup_path=args.empirical_lookup)

    # ======== 1. Baseline (không whitening) ========
    print("=== BASELINE (không whitening) ===")
    handler1 = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )
    test_iter1 = genuine_pairs_iter(test_csv, test_cache_dir, max_pairs=args.max_pairs)
    n1, pass1 = evaluate_gmr(
        handler1, test_iter1, args.max_pairs, whiten_params=None, emp_mod=emp_mod
    )
    gmr1 = 100.0 * pass1 / n1 if n1 > 0 else 0.0
    print(f"GMR (no whiten): {pass1}/{n1} ({gmr1:.2f}%)")
    handler1.sess.close()

    # ======== 2. Whitening ========
    print("=== WHITENING ===")
    handler2 = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )
    test_iter2 = genuine_pairs_iter(test_csv, test_cache_dir, max_pairs=args.max_pairs)
    n2, pass2 = evaluate_gmr(
        handler2, test_iter2, args.max_pairs, whiten_params=(W, mean), emp_mod=emp_mod
    )
    gmr2 = 100.0 * pass2 / n2 if n2 > 0 else 0.0
    print(f"GMR (whiten)   : {pass2}/{n2} ({gmr2:.2f}%)")
    handler2.sess.close()

    # Tổng kết
    print("----------------------------")
    print(f"Baseline       : {gmr1:.2f}%")
    print(f"+ Whitening    : {gmr2:.2f}%")


if __name__ == "__main__":
    main()
