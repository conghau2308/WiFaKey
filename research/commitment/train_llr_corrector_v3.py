"""
train_llr_corrector_v3.py

Huấn luyện Neural LLR Correction với kiến trúc tùy chỉnh.
Tạo dữ liệu từ tập train (LFW ± CPLFW) rồi huấn luyện.
Dữ liệu được lưu vào .npz để tái sử dụng.
"""

import argparse
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
import csv
import hashlib
import pickle


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


def margin_to_llr_empirical(margin, margin_bp, p_bp, eps=1e-6):
    p = np.interp(margin, margin_bp, p_bp, left=p_bp[0], right=p_bp[-1])
    p = np.clip(p, eps, 0.5 - eps)
    return np.log((1.0 - p) / p).astype(np.float32)


def build_empirical_lookup(handler, pairs_iter, num_pairs, n_bins=100):
    all_margins = []
    all_errors = []
    count = 0
    for feat_enroll, feat_verify in pairs_iter:
        if count >= num_pairs:
            break
        b_full_enroll = handler._binarize_full(feat_enroll).astype(np.uint8)
        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full_enroll), size=handler.feature_length, replace=False
        )
        selection_indices.sort()
        selection_mask = np.zeros(len(b_full_enroll), dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected_enroll = b_full_enroll[selection_indices]

        b_full_verify = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_selected_verify = b_full_verify[idx]

        error_bits = (b_selected_enroll != b_selected_verify).astype(np.float32)

        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx]

        all_margins.append(margin_sel)
        all_errors.append(error_bits)
        count += 1

    all_margins = np.concatenate(all_margins)
    all_errors = np.concatenate(all_errors)
    bin_edges = np.linspace(all_margins.min(), all_margins.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    p_bp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (all_margins >= bin_edges[i]) & (all_margins < bin_edges[i + 1])
        if mask.sum() > 0:
            p_bp[i] = all_errors[mask].mean()
        else:
            p_bp[i] = 0.5
    p_bp = np.clip(p_bp, 1e-6, 0.5 - 1e-6)
    return bin_centers.astype(np.float32), p_bp.astype(np.float32)


def generate_training_data(
    handler,
    lfw_csv,
    lfw_cache_dir,
    cplfw_csv=None,
    cplfw_cache_dir=None,
    empirical_lookup=None,
):
    """Sinh dữ liệu huấn luyện từ LFW (và CPLFW nếu có)."""
    margins = []
    llr_emp = []
    targets = []

    # Empirical LLR builder if no lookup
    if empirical_lookup is None:
        print("Xây dựng empirical lookup từ 400 cặp LFW...")
        lfw_iter_lookup = genuine_pairs_iter(lfw_csv, lfw_cache_dir, max_pairs=400)
        margin_bp, p_bp = build_empirical_lookup(handler, lfw_iter_lookup, 400)
        print(f"Đã tạo lookup với {len(margin_bp)} bins.")
    else:
        from research.modulation.v2_empirical_llr import EmpiricalLLR

        emp_mod = EmpiricalLLR(lookup_path=empirical_lookup)
        margin_bp, p_bp = None, None  # will use emp_mod directly

    # Process LFW
    lfw_iter = genuine_pairs_iter(lfw_csv, lfw_cache_dir, max_pairs=881)
    count = 0
    for feat_enroll, feat_verify in lfw_iter:
        b_full_enroll = handler._binarize_full(feat_enroll).astype(np.uint8)
        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full_enroll), size=handler.feature_length, replace=False
        )
        selection_indices.sort()
        selection_mask = np.zeros(len(b_full_enroll), dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected_enroll = b_full_enroll[selection_indices]

        random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected_enroll, codeword).astype(np.uint8)

        b_full_verify = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_selected_verify = b_full_verify[idx]
        noisy = np.logical_xor(b_selected_verify, helper_data).astype(np.uint8)

        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx]

        if empirical_lookup is None:
            llr_mag = margin_to_llr_empirical(margin_sel, margin_bp, p_bp)
            sign = 2 * noisy.astype(np.float32) - 1
            llr = sign * llr_mag
        else:
            llr = emp_mod.modulate(noisy, context={"margin": margin_sel})

        target = 2 * codeword.astype(np.float32) - 1  # ±1

        margins.append(margin_sel)
        llr_emp.append(llr)
        targets.append(target)
        count += 1
        if count % 200 == 0:
            print(f"  Đã xử lý {count} cặp LFW")
    print(f"LFW: {len(margins)} cặp")

    # Process CPLFW if available
    if cplfw_csv is not None:
        cplfw_iter = genuine_pairs_iter(cplfw_csv, cplfw_cache_dir, max_pairs=1694)
        cplfw_count = 0
        for feat_enroll, feat_verify in cplfw_iter:
            # same logic as above, but we can reuse code
            b_full_enroll = handler._binarize_full(feat_enroll).astype(np.uint8)
            rng = np.random.default_rng()
            selection_indices = rng.choice(
                len(b_full_enroll), size=handler.feature_length, replace=False
            )
            selection_indices.sort()
            selection_mask = np.zeros(len(b_full_enroll), dtype=np.uint8)
            selection_mask[selection_indices] = 1
            b_selected_enroll = b_full_enroll[selection_indices]

            random_key = np.random.randint(
                0, 2, size=(1, handler.key_length), dtype=int
            )
            codeword = (
                handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
            )
            helper_data = np.logical_xor(b_selected_enroll, codeword).astype(np.uint8)

            b_full_verify = handler._binarize_full(feat_verify).astype(np.uint8)
            idx = np.where(selection_mask == 1)[0]
            b_selected_verify = b_full_verify[idx]
            noisy = np.logical_xor(b_selected_verify, helper_data).astype(np.uint8)

            _, margin_all = binarize_with_perbit_confidence(
                np.dot(feat_verify, handler.M_matrix), handler.intervals
            )
            margin_sel = margin_all[idx]

            if empirical_lookup is None:
                llr_mag = margin_to_llr_empirical(margin_sel, margin_bp, p_bp)
                sign = 2 * noisy.astype(np.float32) - 1
                llr = sign * llr_mag
            else:
                llr = emp_mod.modulate(noisy, context={"margin": margin_sel})

            target = 2 * codeword.astype(np.float32) - 1

            margins.append(margin_sel)
            llr_emp.append(llr)
            targets.append(target)
            cplfw_count += 1
            if cplfw_count % 200 == 0:
                print(f"  Đã xử lý {cplfw_count} cặp CPLFW")
        print(f"CPLFW: {cplfw_count} cặp")

    margins = np.array(margins, dtype=np.float32)
    llr_emp = np.array(llr_emp, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    return margins, llr_emp, targets


def build_model(margin_input, llr_input, is_training, args):
    """Xây dựng MLP theo tham số dòng lệnh."""
    features = tf.stack([margin_input, llr_input], axis=-1)  # (batch, 832, 2)
    batch_size = tf.shape(features)[0]
    flat = tf.reshape(features, [-1, 2])

    x = flat
    for i in range(args.num_layers):
        x = tf.layers.dense(x, args.hidden_units, activation=None, name=f"fc{i}")
        if args.batch_norm:
            x = tf.layers.batch_normalization(x, training=is_training, name=f"bn{i}")
        # Activation
        if args.activation == "relu":
            x = tf.nn.relu(x)
        elif args.activation == "leaky_relu":
            x = tf.nn.leaky_relu(x, alpha=0.1)
        elif args.activation == "swish":
            x = x * tf.nn.sigmoid(x)
        # Dropout
        if args.dropout > 0:
            x = tf.layers.dropout(x, rate=args.dropout, training=is_training)

    # Output layer (linear)
    output_flat = tf.layers.dense(x, 1, activation=None, name="output")
    output = tf.reshape(output_flat, [batch_size, -1])
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--empirical-lookup", default=None, help="Path to reliability_lookup.npz"
    )
    ap.add_argument("--add-cplfw", action="store_true")
    ap.add_argument("--num-layers", type=int, default=3, help="Số lớp ẩn")
    ap.add_argument("--hidden-units", type=int, default=128, help="Số nơ-ron mỗi lớp")
    ap.add_argument(
        "--activation", choices=["relu", "leaky_relu", "swish"], default="relu"
    )
    ap.add_argument("--batch-norm", action="store_true", default=True)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--model-dir", default="./checkpoints/neural_llr_v3")
    ap.add_argument(
        "--data-cache",
        default="./checkpoints/neural_training_data.npz",
        help="Nếu có file này, bỏ qua bước tạo dữ liệu và dùng luôn",
    )
    ap.add_argument(
        "--device",
        default="/gpu:0",
        help="Thiết bị huấn luyện ('/cpu:0' hoặc '/gpu:0')",
    )
    ap.add_argument(
        "--force-cpu-handler",
        action="store_true",
        help="Chạy handler trên CPU khi tạo dữ liệu",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")

    lfw_pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    lfw_cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )
    lfw_csv = os.path.join(lfw_pairs_dir, "tune_genuine.csv")

    cplfw_csv = None
    cplfw_cache_dir = None
    if args.add_cplfw:
        cplfw_pairs_dir = os.path.join(root, "datasets", "processed", "cplfw", "pairs")
        cplfw_cache_dir = os.path.join(
            root, "datasets", "processed", "cplfw", "embeddings_cache"
        )
        cplfw_csv = os.path.join(cplfw_pairs_dir, "select_genuine.csv")

    # Bước 1: Tạo dữ liệu (nếu chưa có cache)
    if os.path.exists(args.data_cache):
        print(f"Sử dụng dữ liệu từ {args.data_cache}")
        data = np.load(args.data_cache)
        margins = data["margins"]
        llr_emp = data["llr_emp"]
        targets = data["targets"]
    else:
        if args.force_cpu_handler:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Khởi tạo handler để sinh dữ liệu...")
        handler = SecureWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
        margins, llr_emp, targets = generate_training_data(
            handler,
            lfw_csv,
            lfw_cache_dir,
            cplfw_csv,
            cplfw_cache_dir,
            empirical_lookup=args.empirical_lookup,
        )
        handler.sess.close()
        np.savez_compressed(
            args.data_cache, margins=margins, llr_emp=llr_emp, targets=targets
        )
        print(f"Đã lưu dữ liệu vào {args.data_cache}")
        if args.force_cpu_handler:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    print(f"Dữ liệu: {margins.shape[0]} cặp, {margins.shape[1]} bit/cặp")

    # Xáo trộn
    idx = np.random.permutation(margins.shape[0])
    margins = margins[idx]
    llr_emp = llr_emp[idx]
    targets = targets[idx]

    # Bước 2: Xây dựng mô hình và huấn luyện
    with tf.device(args.device):
        tf.reset_default_graph()
        margin_ph = tf.placeholder(tf.float32, [None, 832], name="margin")
        llr_ph = tf.placeholder(tf.float32, [None, 832], name="llr_emp")
        target_ph = tf.placeholder(tf.float32, [None, 832], name="target")
        is_training = tf.placeholder_with_default(False, shape=[], name="is_training")

        pred_llr = build_model(margin_ph, llr_ph, is_training, args)

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=(target_ph + 1.0) / 2.0, logits=pred_llr
            )
        )
        optimizer = tf.train.AdamOptimizer(args.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        n_samples = margins.shape[0]
        batches_per_epoch = int(np.ceil(n_samples / args.batch_size))
        print(f"Bắt đầu huấn luyện {args.epochs} epochs, batch size {args.batch_size}")
        for epoch in range(args.epochs):
            perm = np.random.permutation(n_samples)
            margins_epoch = margins[perm]
            llr_epoch = llr_emp[perm]
            targets_epoch = targets[perm]
            total_loss = 0
            for b in range(batches_per_epoch):
                start = b * args.batch_size
                end = min(start + args.batch_size, n_samples)
                feed = {
                    margin_ph: margins_epoch[start:end],
                    llr_ph: llr_epoch[start:end],
                    target_ph: targets_epoch[start:end],
                    is_training: True,
                }
                _, batch_loss = sess.run([train_op, loss], feed_dict=feed)
                total_loss += batch_loss * (end - start)
            avg_loss = total_loss / n_samples
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Tạo tên thư mục dựa trên config
        config_str = f"L{args.num_layers}_H{args.hidden_units}_{args.activation}"
        if args.add_cplfw:
            config_str += "_cplfw"
        if args.dropout > 0:
            config_str += f"_dp{args.dropout}"
        model_dir = os.path.join(args.model_dir, config_str)
        os.makedirs(model_dir, exist_ok=True)
        saver.save(sess, os.path.join(model_dir, "model"))
        print(f"Đã lưu model vào {model_dir}")

        sess.close()


if __name__ == "__main__":
    main()
