"""
diagnostic_final.py

Đánh giá tổng hợp: BPSK, Empirical LLR, Neural LLR Correction,
và kết hợp Multi‑start Decoding.

Cách dùng:
  python research/commitment/diagnostic_final.py \
    --empirical-lookup experiments/out_step3/reliability_lookup.npz \
    --model-path checkpoints/neural_llr_v3/L3_H128_relu/model \
    --K 5 --sigma 0.2 --max-pairs 200
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


# --- Loaders ---
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


# --- Neural model builder (giống hệt lúc train) ---
def build_neural_model(
    margin_ph,
    llr_ph,
    is_training=False,
    num_layers=3,
    hidden_units=128,
    activation="relu",
    dropout=0.0,
):
    features = tf.stack([margin_ph, llr_ph], axis=-1)  # (batch, 832, 2)
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


# --- Hàm chính ---
def run_diagnostic(
    handler,
    pairs_iter,
    max_pairs,
    empirical_lookup=None,
    model_path=None,
    model_args=None,
    K=5,
    sigma=0.2,
):
    # Chuẩn bị Empirical LLR nếu có
    emp_mod = None
    if empirical_lookup is not None:
        from research.modulation.v2_empirical_llr import EmpiricalLLR

        emp_mod = EmpiricalLLR(lookup_path=empirical_lookup)

    # Chuẩn bị Neural model nếu có
    sess_nn = None
    margin_ph_nn = None
    llr_ph_nn = None
    pred_llr_nn = None
    if model_path is not None and model_args is not None:
        tf.reset_default_graph()
        margin_ph_nn = tf.placeholder(tf.float32, [1, 832], name="margin_nn")
        llr_ph_nn = tf.placeholder(tf.float32, [1, 832], name="llr_nn")
        pred_llr_nn = build_neural_model(margin_ph_nn, llr_ph_nn, False, **model_args)
        config_nn = tf.ConfigProto()
        config_nn.gpu_options.allow_growth = True
        sess_nn = tf.Session(config=config_nn)
        saver_nn = tf.train.Saver()
        saver_nn.restore(sess_nn, model_path)

    n_total = 0
    pass_bpsk = 0
    pass_emp_single = 0
    pass_emp_multi = 0
    pass_nn_single = 0
    pass_nn_multi = 0

    for feat_enroll, feat_verify in pairs_iter:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # --- BPSK baseline ---
        ok_bpsk = handler.verify(feat_verify, helper, sel_mask, key_hash)

        # --- Chuẩn bị dữ liệu chung cho verify ---
        b_full = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx].astype(np.float32)

        # --- Empirical LLR ---
        if emp_mod is not None:
            # single
            llr_emp = emp_mod.modulate(noisy, context={"margin": margin_sel})
            ok_emp_single = _decode_and_check(handler, llr_emp, key_hash)
            # multi
            ok_emp_multi = ok_emp_single
            if not ok_emp_single:
                ok_emp_multi = _multi_start_decode(handler, llr_emp, key_hash, K, sigma)
            if ok_emp_single:
                pass_emp_single += 1
            if ok_emp_multi:
                pass_emp_multi += 1

        # --- Neural Correction ---
        if sess_nn is not None:
            # Cần LLR đầu vào cho neural: dùng empirical nếu có, nếu không dùng BPSK
            if emp_mod is not None:
                llr_input = emp_mod.modulate(noisy, context={"margin": margin_sel})
            else:
                llr_input = 2 * noisy.astype(np.float32) - 1
            llr_input = llr_input.reshape(1, -1)
            margin_reshaped = margin_sel.reshape(1, -1)
            corrected_llr = sess_nn.run(
                pred_llr_nn,
                feed_dict={margin_ph_nn: margin_reshaped, llr_ph_nn: llr_input},
            ).flatten()

            # single
            ok_nn_single = _decode_and_check(handler, corrected_llr, key_hash)
            # multi
            ok_nn_multi = ok_nn_single
            if not ok_nn_single:
                ok_nn_multi = _multi_start_decode(
                    handler, corrected_llr, key_hash, K, sigma
                )
            if ok_nn_single:
                pass_nn_single += 1
            if ok_nn_multi:
                pass_nn_multi += 1

        if ok_bpsk:
            pass_bpsk += 1

    # In kết quả
    print(f"Tổng cặp test: {n_total}")
    print(
        f"GMR BPSK baseline        : {pass_bpsk}/{n_total} ({100*pass_bpsk/n_total:.2f}%)"
    )
    if emp_mod is not None:
        print(
            f"GMR Empirical LLR (single): {pass_emp_single}/{n_total} ({100*pass_emp_single/n_total:.2f}%)"
        )
        print(
            f"GMR Empirical LLR + Multi : {pass_emp_multi}/{n_total} ({100*pass_emp_multi/n_total:.2f}%)"
        )
    if sess_nn is not None:
        print(
            f"GMR Neural Correction (single): {pass_nn_single}/{n_total} ({100*pass_nn_single/n_total:.2f}%)"
        )
        print(
            f"GMR Neural Correction + Multi : {pass_nn_multi}/{n_total} ({100*pass_nn_multi/n_total:.2f}%)"
        )

    if sess_nn is not None:
        sess_nn.close()


def _decode_and_check(handler, llr_flat, key_hash):
    """Giải mã LLR phẳng (832,) và kiểm tra hash."""
    llr = llr_flat.reshape(1, handler.N, handler.Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def _multi_start_decode(handler, llr_clean_flat, key_hash, K, sigma):
    """Thử tối đa K lần, mỗi lần thêm nhiễu nhẹ, trả về True nếu có lần khớp."""
    llr_clean = llr_clean_flat.reshape(1, handler.N, handler.Z)
    for _ in range(K):
        noise = np.random.normal(0, sigma, size=llr_clean.shape).astype(np.float32)
        llr_noisy = llr_clean + noise
        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr_noisy}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--empirical-lookup",
        default=None,
        help="Đường dẫn reliability_lookup.npz (nếu có)",
    )
    ap.add_argument(
        "--model-path",
        default=None,
        help="Đường dẫn checkpoint Neural Correction (nếu có)",
    )
    ap.add_argument(
        "--num-layers", type=int, default=3, help="Số lớp ẩn của Neural model"
    )
    ap.add_argument("--hidden-units", type=int, default=128, help="Số nơ-ron mỗi lớp")
    ap.add_argument(
        "--activation", default="relu", choices=["relu", "leaky_relu", "swish"]
    )
    ap.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate lúc suy luận (thường là 0)",
    )
    ap.add_argument("--K", type=int, default=5, help="Số lần thử trong multi‑start")
    ap.add_argument(
        "--sigma", type=float, default=0.2, help="Độ lệch chuẩn nhiễu Gaussian"
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="Chạy handler LDPC trên CPU")
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

    print("Khởi tạo handler LDPC (một session duy nhất)...")
    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    # Gói tham số model
    model_args = None
    if args.model_path:
        model_args = {
            "num_layers": args.num_layers,
            "hidden_units": args.hidden_units,
            "activation": args.activation,
            "dropout": args.dropout,
        }

    test_iter = genuine_pairs_iter(test_csv, test_cache_dir, max_pairs=args.max_pairs)
    run_diagnostic(
        handler,
        test_iter,
        args.max_pairs,
        empirical_lookup=args.empirical_lookup,
        model_path=args.model_path,
        model_args=model_args,
        K=args.K,
        sigma=args.sigma,
    )

    handler.sess.close()


if __name__ == "__main__":
    main()
