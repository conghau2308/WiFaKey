"""
train_llr_corrector.py

Huấn luyện mô hình Neural LLR Correction (hướng L) trên tập con LFW.
Mô hình ánh xạ margin và LLR thô -> LLR hiệu chỉnh, dùng TF1.x.
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
from wifakey_module.wifakey_lib import Modulation
import csv


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


def build_training_data(handler, pairs_iter, num_pairs, erasure_count=0):
    """
    Duyệt qua các cặp genuine, trả về mảng:
        margins: (N_samples, 832)  margin cho từng bit được chọn
        llr_raw: (N_samples, 832)  LLR BPSK thô (±1)
        targets: (N_samples, 832)  bit codeword đúng dạng ±1
    """
    all_margins = []
    all_llr_raw = []
    all_targets = []

    count = 0
    for feat_enroll, feat_verify in pairs_iter:
        if count >= num_pairs:
            break
        # Enrollment (cần oracle key để lấy target)
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

        # Verify: lấy b_full, margin, LLR thô
        b_full_verify = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_selected_verify = b_full_verify[idx]
        noisy = np.logical_xor(b_selected_verify, helper_data).astype(np.uint8)
        llr_raw = 2 * noisy.astype(np.float32) - 1  # 832

        # Margin cho các bit được chọn
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx]  # 832

        # Target: bit codeword đúng (0/1 -> -1/+1)
        target = 2 * codeword.astype(np.float32) - 1  # 832

        all_margins.append(margin_sel)
        all_llr_raw.append(llr_raw)
        all_targets.append(target)
        count += 1

    return (
        np.array(all_margins, dtype=np.float32),
        np.array(all_llr_raw, dtype=np.float32),
        np.array(all_targets, dtype=np.float32),
    )


def build_model(margin_input, llr_input, is_training):
    """MLP nhỏ: margin và llr -> LLR mới"""
    # Ghép margin và llr thành vector 2 chiều cho mỗi bit
    features = tf.stack([margin_input, llr_input], axis=-1)  # (batch, 832, 2)
    batch_size = tf.shape(features)[0]
    flat = tf.reshape(features, [-1, 2])  # (batch*832, 2)

    # Lớp 1
    h1 = tf.layers.dense(flat, 64, activation=tf.nn.relu, name="h1")
    h1 = tf.layers.batch_normalization(h1, training=is_training, name="bn1")
    # Lớp 2
    h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu, name="h2")
    h2 = tf.layers.batch_normalization(h2, training=is_training, name="bn2")
    # Lớp 3 (tuyến tính)
    h3 = tf.layers.dense(h2, 1, activation=None, name="h3")  # (batch*832, 1)
    output = tf.reshape(h3, [batch_size, -1])  # (batch, 832)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--train-pairs", type=int, default=400, help="Số cặp dùng để train")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--model-dir", default="./checkpoints/neural_llr_corrector")
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

    print("Khởi tạo handler...")
    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    print("Xây dựng dữ liệu huấn luyện...")
    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, args.train_pairs)
    margins, llr_raw, targets = build_training_data(
        handler, pairs_iter, args.train_pairs
    )
    print(
        f"Đã lấy {margins.shape[0]} cặp, mỗi cặp 832 bit. Tổng số mẫu = {margins.size}"
    )

    # Xáo trộn
    idx = np.random.permutation(margins.shape[0])
    margins = margins[idx]
    llr_raw = llr_raw[idx]
    targets = targets[idx]

    # Build model
    tf.reset_default_graph()
    margin_ph = tf.placeholder(tf.float32, [None, 832], name="margin")
    llr_ph = tf.placeholder(tf.float32, [None, 832], name="llr_raw")
    target_ph = tf.placeholder(tf.float32, [None, 832], name="target")
    is_training = tf.placeholder_with_default(False, shape=[], name="is_training")

    pred_llr = build_model(margin_ph, llr_ph, is_training)

    # Loss: sigmoid cross entropy với target (coi target là +1/-1, pred là logit)
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

    # Huấn luyện
    sess.run(tf.global_variables_initializer())
    n_samples = margins.shape[0]
    batches_per_epoch = int(np.ceil(n_samples / args.batch_size))
    print(f"Bắt đầu huấn luyện {args.epochs} epochs, batch size {args.batch_size}")
    for epoch in range(args.epochs):
        perm = np.random.permutation(n_samples)
        margins_epoch = margins[perm]
        llr_epoch = llr_raw[perm]
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

    # Lưu model
    os.makedirs(args.model_dir, exist_ok=True)
    saver.save(sess, os.path.join(args.model_dir, "model"))
    print(f"Đã lưu model vào {args.model_dir}")

    sess.close()
    handler.sess.close()


if __name__ == "__main__":
    main()
