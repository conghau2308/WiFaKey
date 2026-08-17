"""
diagnostic_hotspot_multistart.py

Đánh giá hiệu quả của Adaptive Hotspot Multi‑start Decoding.
So sánh GMR giữa:
    - Empirical LLR + Multi‑start thường (sigma cố định)
    - Empirical LLR + Multi‑start có trọng số (sigma biến thiên theo hotspot map)

Cách dùng:
    python diagnostic_hotspot_multistart.py --lookup <path> --hotspot hotspot_map.npy --K 5 --sigma 0.2 --alpha 2.0 --max-pairs 200
"""

import argparse
import csv
import hashlib
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR
from wifakey_module.wifakey_lib import Modulation


# --- Loaders (giữ nguyên) ---
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


# --- Hàm giải mã đa khởi động với trọng số thích ứng ---
def multi_start_decode_adaptive(
    handler, llr_flat, key_hash, K, sigma_base, alpha, hotspot_weights
):
    """
    hotspot_weights: mảng 832 phần tử, giá trị trong khoảng [0, 1] hoặc đã chuẩn hóa.
    sigma[i] = sigma_base * (1 + alpha * hotspot_weights[i])
    """
    # Thử lần đầu không nhiễu
    if _try_decode(handler, llr_flat, key_hash):
        return True
    if K <= 0:
        return False

    llr_clean = llr_flat.reshape(1, handler.N, handler.Z)
    # Tính ma trận sigma cho từng vị trí (shape 1, N, Z)
    sigma_map = sigma_base * (
        1.0 + alpha * hotspot_weights.reshape(1, handler.N, handler.Z)
    )

    for _ in range(K):
        noise = np.random.normal(0, 1, size=llr_clean.shape).astype(np.float32)
        noise_scaled = noise * sigma_map
        llr_noisy = llr_clean + noise_scaled
        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr_noisy}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            return True
    return False


def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, handler.N, handler.Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


# --- Hàm chính ---
def run_diagnostic(
    handler, pairs_iter, max_pairs, lookup_path, hotspot_path, K, sigma, alpha
):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    hotspot_1536 = np.load(hotspot_path).astype(np.float32)  # shape (1536,)

    n_total = 0
    pass_baseline = 0
    pass_uniform = 0
    pass_adaptive = 0

    for feat_enroll, feat_verify in pairs_iter:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # Baseline BPSK (dùng verify gốc của handler)
        ok_bpsk = handler.verify(feat_verify, helper, sel_mask, key_hash)

        # Chuẩn bị LLR và hotspot weights
        b_full = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx]
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()

        # Trọng số cho 832 bit được chọn
        hotspot_sel = hotspot_1536[idx]

        # Multi‑start thường (sigma cố định)
        ok_uniform = multi_start_decode_adaptive(
            handler, llr, key_hash, K, sigma, 0.0, hotspot_sel
        )

        # Multi‑start thích ứng
        ok_adaptive = multi_start_decode_adaptive(
            handler, llr, key_hash, K, sigma, alpha, hotspot_sel
        )

        if ok_bpsk:
            pass_baseline += 1
        if ok_uniform:
            pass_uniform += 1
        if ok_adaptive:
            pass_adaptive += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"GMR BPSK baseline           : {pass_baseline}/{n_total} ({100*pass_baseline/n_total:.2f}%)"
    )
    print(
        f"GMR Empirical + Multi (uniform) : {pass_uniform}/{n_total} ({100*pass_uniform/n_total:.2f}%)"
    )
    print(
        f"GMR Empirical + Multi (adaptive): {pass_adaptive}/{n_total} ({100*pass_adaptive/n_total:.2f}%)"
    )

    return pass_baseline, pass_uniform, pass_adaptive


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookup", required=True, help="Path reliability_lookup.npz")
    ap.add_argument("--hotspot", required=True, help="Path hotspot_map.npy")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Hệ số ảnh hưởng của hotspot (0 = uniform)",
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
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
    pairs_csv = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")

    print(f"Đọc genuine pairs từ: {pairs_csv}")
    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(
        handler,
        pairs_iter,
        args.max_pairs,
        args.lookup,
        args.hotspot,
        args.K,
        args.sigma,
        args.alpha,
    )

    handler.sess.close()


if __name__ == "__main__":
    main()
