"""
diagnostic_neural_llr_v2.py

Đánh giá Neural LLR Correction (v2) dùng Empirical LLR làm đầu vào.
So sánh BPSK baseline, Empirical LLR, và Neural Correction.
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

# Loaders (giữ nguyên)


def build_model(margin_input, llr_input, is_training=False):
    features = tf.stack([margin_input, llr_input], axis=-1)
    batch_size = tf.shape(features)[0]
    flat = tf.reshape(features, [-1, 2])
    h1 = tf.layers.dense(flat, 128, activation=tf.nn.relu, name="h1")
    h1 = tf.layers.batch_normalization(h1, training=is_training, name="bn1")
    h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu, name="h2")
    h2 = tf.layers.batch_normalization(h2, training=is_training, name="bn2")
    h3 = tf.layers.dense(h2, 1, activation=None, name="h3")
    output = tf.reshape(h3, [batch_size, -1])
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--empirical-lookup", default=None, help="File reliability_lookup.npz (nếu có)"
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

    # Load empirical LLR module nếu có
    if args.empirical_lookup is not None:
        from research.modulation.v2_empirical_llr import EmpiricalLLR

        emp_mod = EmpiricalLLR(lookup_path=args.empirical_lookup)
    else:
        emp_mod = None
        # Tự build empirical lookup từ 400 mẫu LFW? Hoặc dùng lại code cũ.
        # Để đơn giản, ta sẽ không so sánh Empirical LLR nếu không có lookup.
        print("Không có --empirical-lookup, sẽ bỏ qua Empirical LLR baseline.")

    # Load model neural correction
    tf.reset_default_graph()
    margin_ph = tf.placeholder(tf.float32, [1, 832], name="margin")
    llr_ph = tf.placeholder(tf.float32, [1, 832], name="llr_emp")
    pred_llr = build_model(margin_ph, llr_ph, False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess_nn = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess_nn, args.model_path)

    # Load dữ liệu test
    # (dùng iterator)
    from train_llr_corrector_v2 import genuine_pairs_iter  # hoặc copy lại

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, max_pairs=args.max_pairs)

    n_total = 0
    pass_bpsk = 0
    pass_emp = 0
    pass_neural = 0

    for feat_enroll, feat_verify in pairs_iter:
        if args.max_pairs is not None and n_total >= args.max_pairs:
            break
        n_total += 1

        # Enrollment
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # --- BPSK baseline ---
        ok_bpsk = handler.verify(feat_verify, helper, sel_mask, key_hash)

        # --- Empirical LLR (nếu có) ---
        ok_emp = False
        if emp_mod is not None:
            b_full = handler._binarize_full(feat_verify).astype(np.uint8)
            idx = np.where(sel_mask == 1)[0]
            b_sel = b_full[idx]
            noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
            _, margin_all = binarize_with_perbit_confidence(
                np.dot(feat_verify, handler.M_matrix), handler.intervals
            )
            margin_sel = margin_all[idx]
            llr_emp = emp_mod.modulate(noisy, context={"margin": margin_sel})
            llr_emp = llr_emp.reshape(1, handler.N, handler.Z)
            y_pred_emp = handler.sess.run(
                handler.decoder_output, feed_dict={handler.xa: llr_emp}
            )
            y_pred_emp = y_pred_emp.flatten()
            decoded_key_emp = (y_pred_emp > 0).astype(int)[: handler.key_length]
            if hashlib.sha256(decoded_key_emp.tobytes()).digest() == key_hash:
                ok_emp = True

        # --- Neural Correction ---
        b_full = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx]
        # Cần tính LLR đầu vào cho neural: nếu có emp_mod thì dùng nó, nếu không thì dùng BPSK
        if emp_mod is not None:
            llr_input = emp_mod.modulate(noisy, context={"margin": margin_sel})
        else:
            llr_input = 2 * noisy.astype(np.float32) - 1
        llr_input = llr_input.reshape(1, -1)
        margin_sel = margin_sel.reshape(1, -1)

        corrected_llr = sess_nn.run(
            pred_llr, feed_dict={margin_ph: margin_sel, llr_ph: llr_input}
        )
        y_llr = corrected_llr.reshape(1, handler.N, handler.Z)
        y_pred_nn = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: y_llr}
        )
        y_pred_nn = y_pred_nn.flatten()
        decoded_key_nn = (y_pred_nn > 0).astype(int)[: handler.key_length]
        ok_neural = hashlib.sha256(decoded_key_nn.tobytes()).digest() == key_hash

        if ok_bpsk:
            pass_bpsk += 1
        if ok_emp:
            pass_emp += 1
        if ok_neural:
            pass_neural += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"GMR BPSK baseline      : {pass_bpsk}/{n_total} ({100*pass_bpsk/n_total:.2f}%)"
    )
    if emp_mod is not None:
        print(
            f"GMR Empirical LLR      : {pass_emp}/{n_total} ({100*pass_emp/n_total:.2f}%)"
        )
    print(
        f"GMR Neural Correction  : {pass_neural}/{n_total} ({100*pass_neural/n_total:.2f}%)"
    )

    sess_nn.close()
    handler.sess.close()


if __name__ == "__main__":
    main()
