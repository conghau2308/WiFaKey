"""
diagnostic_bit_denoiser.py (ĐÃ SỬA LỖI UnboundLocalError)

Đánh giá hiệu quả của Bit Denoising Network trong pipeline an toàn.
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

FULL_BITS = 1536
N, m, Z = 52, 42, 16


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


# Đổi tên hàm này để tránh xung đột với biến cùng tên
def iterate_pairs(pairs_csv, cache_dir, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2, row
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


def run_diagnostic(handler, pairs_iter, max_pairs, lookup_path, model_path):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)

    # Load mô hình denoiser
    tf.reset_default_graph()
    denoiser_graph = tf.Graph()
    with denoiser_graph.as_default():
        x_ph = tf.placeholder(tf.float32, [1, FULL_BITS], name="noisy_bits")
        h1 = tf.layers.dense(x_ph, 512, activation=tf.nn.relu, name="h1")
        h2 = tf.layers.dense(h1, 512, activation=tf.nn.relu, name="h2")
        logits = tf.layers.dense(h2, FULL_BITS, activation=None, name="logits")
        pred_bits = tf.nn.sigmoid(logits)
        saver = tf.train.Saver()

    sess_denoiser = tf.Session(graph=denoiser_graph)
    saver.restore(sess_denoiser, model_path)

    n_total = 0
    pass_baseline = 0
    pass_denoised = 0

    for feat_enroll, feat_verify, row in pairs_iter:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # --- Baseline (không denoise) ---
        ok_baseline = handler.verify(feat_verify, helper, sel_mask, key_hash)

        # --- Có denoise ---
        projected = np.dot(feat_verify, handler.M_matrix)
        bits_v, margin_v = binarize_with_perbit_confidence(projected, handler.intervals)
        bits_v_float = bits_v.astype(np.float32).reshape(1, -1)

        cleaned_bits = sess_denoiser.run(pred_bits, feed_dict={x_ph: bits_v_float})
        cleaned_bits = (cleaned_bits > 0.5).astype(np.uint8).flatten()

        idx = np.where(sel_mask == 1)[0]
        b_selected = cleaned_bits[idx]
        noisy = np.logical_xor(b_selected, helper).astype(np.uint8)

        margin_sel = margin_v[idx]
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()

        llr = llr.reshape(1, N, Z)
        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        ok_denoised = hashlib.sha256(decoded_key.tobytes()).digest() == key_hash

        if ok_baseline:
            pass_baseline += 1
        if ok_denoised:
            pass_denoised += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"GMR Baseline (Empirical LLR) : {pass_baseline}/{n_total} ({100*pass_baseline/n_total:.2f}%)"
    )
    print(
        f"GMR + Bit Denoising          : {pass_denoised}/{n_total} ({100*pass_denoised/n_total:.2f}%)"
    )

    sess_denoiser.close()
    handler.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True, help="Path reliability_lookup.npz")
    ap.add_argument("--model", required=True, help="Path bit denoiser checkpoint")
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

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    # Sử dụng hàm iterate_pairs đã sửa
    pair_iter = iterate_pairs(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(handler, pair_iter, args.max_pairs, args.lookup, args.model)


if __name__ == "__main__":
    main()
