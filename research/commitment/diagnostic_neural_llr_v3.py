"""
diagnostic_neural_llr_v3.py

Đánh giá mô hình Neural LLR Correction (phiên bản linh hoạt).
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


def build_model(
    margin_input,
    llr_input,
    is_training=False,
    num_layers=3,
    hidden_units=128,
    activation="relu",
    dropout=0.0,
):
    features = tf.stack([margin_input, llr_input], axis=-1)
    batch_size = tf.shape(features)[0]
    flat = tf.reshape(features, [-1, 2])
    x = flat
    for i in range(num_layers):
        x = tf.layers.dense(x, hidden_units, activation=None, name=f"fc{i}")
        x = tf.layers.batch_normalization(x, training=is_training, name=f"bn{i}")
        if activation == "relu":
            x = tf.nn.relu(x)
        elif activation == "leaky_relu":
            x = tf.nn.leaky_relu(x, alpha=0.1)
        elif activation == "swish":
            x = x * tf.nn.sigmoid(x)
        if dropout > 0:
            x = tf.layers.dropout(x, rate=dropout, training=is_training)
    output_flat = tf.layers.dense(x, 1, activation=None, name="output")
    output = tf.reshape(output_flat, [batch_size, -1])
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Đường dẫn đến checkpoint")
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--hidden-units", type=int, default=128)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--empirical-lookup", default=None)
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
    test_pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "pairs"
    )
    test_cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "embeddings_cache"
    )
    test_csv = os.path.join(test_pairs_dir, f"{args.tier}_genuine.csv")

    # Load handler
    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    # Empirical LLR
    emp_mod = None
    if args.empirical_lookup:
        from research.modulation.v2_empirical_llr import EmpiricalLLR

        emp_mod = EmpiricalLLR(lookup_path=args.empirical_lookup)

    # Load mô hình neural correction
    tf.reset_default_graph()
    margin_ph = tf.placeholder(tf.float32, [1, 832], name="margin")
    llr_ph = tf.placeholder(tf.float32, [1, 832], name="llr_emp")
    pred_llr = build_model(
        margin_ph,
        llr_ph,
        False,
        num_layers=args.num_layers,
        hidden_units=args.hidden_units,
        activation=args.activation,
        dropout=args.dropout,
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess_nn = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess_nn, args.model_path)

    # Đánh giá
    n_total = 0
    pass_bpsk = 0
    pass_emp = 0
    pass_neural = 0

    test_iter = genuine_pairs_iter(test_csv, test_cache_dir, max_pairs=args.max_pairs)
    for feat_enroll, feat_verify in test_iter:
        if args.max_pairs is not None and n_total >= args.max_pairs:
            break
        n_total += 1

        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # BPSK baseline
        ok_bpsk = handler.verify(feat_verify, helper, sel_mask, key_hash)

        # Empirical LLR (nếu có)
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
            llr_e = emp_mod.modulate(noisy, context={"margin": margin_sel})
            llr_e = llr_e.reshape(1, handler.N, handler.Z)
            y_pred_e = handler.sess.run(
                handler.decoder_output, feed_dict={handler.xa: llr_e}
            ).flatten()
            decoded_key_e = (y_pred_e > 0).astype(int)[: handler.key_length]
            ok_emp = hashlib.sha256(decoded_key_e.tobytes()).digest() == key_hash

        # Neural Correction
        b_full = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx]
        # LLR đầu vào cho neural: dùng empirical nếu có, nếu không dùng BPSK
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
        ).flatten()
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
