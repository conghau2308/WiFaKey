"""
diagnostic_neural_llr.py

Đánh giá mô hình Neural LLR Correction trên tập test.
So sánh với BPSK baseline và (tuỳ chọn) empirical LLR.
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
from wifakey_module.wifakey_lib import Modulation


# Loaders
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


def build_model(margin_input, llr_input, is_training=False):
    # Xây dựng lại chính xác kiến trúc đã dùng trong huấn luyện
    features = tf.stack([margin_input, llr_input], axis=-1)
    batch_size = tf.shape(features)[0]
    flat = tf.reshape(features, [-1, 2])
    h1 = tf.layers.dense(flat, 64, activation=tf.nn.relu, name="h1")
    h1 = tf.layers.batch_normalization(h1, training=is_training, name="bn1")
    h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu, name="h2")
    h2 = tf.layers.batch_normalization(h2, training=is_training, name="bn2")
    h3 = tf.layers.dense(h2, 1, activation=None, name="h3")
    output = tf.reshape(h3, [batch_size, -1])
    return output


def run_diagnostic(handler, pairs_iter, max_pairs, model_path, empirical_lookup=None):
    tf.reset_default_graph()
    margin_ph = tf.placeholder(tf.float32, [1, 832], name="margin")
    llr_ph = tf.placeholder(tf.float32, [1, 832], name="llr_raw")
    pred_llr = build_model(margin_ph, llr_ph, False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    n_total = 0
    pass_bpsk = 0
    pass_neural = 0
    pass_empirical = 0

    # Load empirical LLR nếu có
    if empirical_lookup is not None:
        from research.modulation.v2_empirical_llr import EmpiricalLLR

        emp_llr = EmpiricalLLR(lookup_path=empirical_lookup)
    else:
        emp_llr = None

    for feat_enroll, feat_verify in pairs_iter:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        # Enrollment an toàn (không oracle key) – dùng verify gốc để đánh giá BPSK
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # --- BPSK baseline (dùng verify của handler) ---
        ok_bpsk = handler.verify(feat_verify, helper, sel_mask, key_hash)

        # --- Neural correction ---
        # Lấy LLR thô BPSK
        b_full = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
        llr_bpsk = (2 * noisy.astype(np.float32) - 1).reshape(1, -1)  # 1x832

        # Margin
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx].reshape(1, -1)

        # Dùng mô hình để hiệu chỉnh
        corrected_llr = sess.run(
            pred_llr, feed_dict={margin_ph: margin_sel, llr_ph: llr_bpsk}
        )  # shape 1x832

        # Giải mã
        y_llr = corrected_llr.reshape(1, handler.N, handler.Z)
        y_pred = handler.sess.run(handler.decoder_output, feed_dict={handler.xa: y_llr})
        y_pred = y_pred.flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        # Hash check
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            pass_neural += 1

        # --- Empirical LLR (nếu có) ---
        if emp_llr is not None:
            # Cần margin cho verify, đã có margin_sel
            # emp_llr.modulate yêu cầu context['margin'], noisy bits dạng uint8
            llr_emp = emp_llr.modulate(noisy, context={"margin": margin_sel.flatten()})
            llr_emp = llr_emp.reshape(1, handler.N, handler.Z)
            y_pred_emp = handler.sess.run(
                handler.decoder_output, feed_dict={handler.xa: llr_emp}
            )
            y_pred_emp = y_pred_emp.flatten()
            decoded_key_emp = (y_pred_emp > 0).astype(int)[: handler.key_length]
            if hashlib.sha256(decoded_key_emp.tobytes()).digest() == key_hash:
                pass_empirical += 1

        if ok_bpsk:
            pass_bpsk += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"GMR BPSK baseline      : {pass_bpsk}/{n_total} ({100*pass_bpsk/n_total:.2f}%)"
    )
    print(
        f"GMR Neural Correction  : {pass_neural}/{n_total} ({100*pass_neural/n_total:.2f}%)"
    )
    if emp_llr is not None:
        print(
            f"GMR Empirical LLR      : {pass_empirical}/{n_total} ({100*pass_empirical/n_total:.2f}%)"
        )

    sess.close()
    handler.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        required=True,
        help="Đường dẫn tới checkpoint đã train (ví dụ checkpoints/neural_llr_corrector/model)",
    )
    ap.add_argument(
        "--empirical-lookup",
        default=None,
        help="Đường dẫn tới reliability_lookup.npz (để so sánh empirical LLR)",
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "embeddings_cache"
    )
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    pairs_csv = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir)
    run_diagnostic(
        handler,
        pairs_iter,
        args.max_pairs,
        model_path=args.model_path,
        empirical_lookup=args.empirical_lookup,
    )


if __name__ == "__main__":
    main()
