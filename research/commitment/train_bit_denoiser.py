"""
train_bit_denoiser.py

Huấn luyện mạng khử nhiễu bit (Bit Denoising Network).
Mạng nhận vector 1536-bit từ ảnh verify, dự đoán vector 1536-bit sạch (giống ảnh enroll).
"""

import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from collections import defaultdict
import csv
from wifakey_module.wifakey_handler import WiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

N, m, Z = 52, 42, 16
FULL_BITS = 1536

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)
PAIRS_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
)
OUTPUT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "bit_denoiser_model"
)

BATCH_SIZE = 64
N_EPOCHS = 100
LEARNING_RATE = 1e-3
VALID_FRAC = 0.15


def load_embedding(name, imagenum):
    return np.load(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def load_all_identities():
    """Trả về dict: identity -> list of (name, imagenum)"""
    id_to_images = defaultdict(list)
    with open(os.path.join(PAIRS_DIR, "tune_genuine.csv"), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_images[row["name_enroll"]].append(
                (row["name_enroll"], int(row["imagenum_enroll"]))
            )
            id_to_images[row["name_verify"]].append(
                (row["name_verify"], int(row["imagenum_verify"]))
            )
    # Loại bỏ trùng lặp
    for ident in id_to_images:
        id_to_images[ident] = list(set(id_to_images[ident]))
    return id_to_images


def build_training_data(handler, id_to_images):
    """Tạo tất cả các cặp (bits_verify, bits_enroll) có thể từ cùng một identity."""
    X, Y = [], []
    for ident, images in id_to_images.items():
        if len(images) < 2:
            continue
        # Lấy embedding cho tất cả ảnh của identity này
        embs = []
        for name, imagenum in images:
            emb = load_embedding(name, imagenum)
            # Binarize
            bits, _ = binarize_with_perbit_confidence(
                np.dot(emb, handler.M_matrix), handler.intervals
            )
            embs.append(bits.astype(np.float32))
        # Tạo tất cả các cặp (enroll, verify)
        for i in range(len(embs)):
            for j in range(len(embs)):
                if i == j:
                    continue
                X.append(embs[j])  # verify
                Y.append(embs[i])  # enroll (target sạch)
    return np.array(X), np.array(Y)


def build_model():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, FULL_BITS], name="noisy_bits")
        y = tf.placeholder(tf.float32, [None, FULL_BITS], name="clean_bits")

        # Mạng đơn giản: 2 lớp ẩn
        h1 = tf.layers.dense(x, 512, activation=tf.nn.relu, name="h1")
        h2 = tf.layers.dense(h1, 512, activation=tf.nn.relu, name="h2")
        logits = tf.layers.dense(h2, FULL_BITS, activation=None, name="logits")

        # Loss: sigmoid cross-entropy
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        )

        # Accuracy (bit-level)
        pred_bits = tf.nn.sigmoid(logits)
        bit_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(pred_bits), y), tf.float32)
        )

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    return {
        "graph": graph,
        "x": x,
        "y": y,
        "loss": loss,
        "bit_accuracy": bit_accuracy,
        "train_op": train_op,
        "init": init,
        "saver": saver,
        "logits": logits,
    }


def main():
    print("Khởi tạo handler nhẹ...")
    # Dùng handler chỉ để lấy M_matrix và intervals, không cần session
    handler = WiFaKeyHandler()
    # Tắt session thật để tiết kiệm RAM
    handler.sess = None

    print("Xây dựng dữ liệu huấn luyện...")
    id_to_images = load_all_identities()
    X, Y = build_training_data(handler, id_to_images)
    print(f"Tổng số cặp huấn luyện: {X.shape[0]}")

    # Chia train/val
    n_val = int(X.shape[0] * VALID_FRAC)
    indices = np.random.permutation(X.shape[0])
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    model = build_model()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=model["graph"], config=config) as sess:
        sess.run(model["init"])

        best_val_acc = 0.0
        for epoch in range(N_EPOCHS):
            # Huấn luyện
            perm = np.random.permutation(len(train_idx))
            for start in range(0, len(train_idx), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(train_idx))
                batch_idx = perm[start:end]
                feed = {model["x"]: X_train[batch_idx], model["y"]: Y_train[batch_idx]}
                sess.run(model["train_op"], feed_dict=feed)

            # Đánh giá validation
            feed_val = {model["x"]: X_val, model["y"]: Y_val}
            val_loss, val_acc = sess.run(
                [model["loss"], model["bit_accuracy"]], feed_dict=feed_val
            )
            print(
                f"Epoch {epoch+1:3d}: val_loss={val_loss:.4f}, val_bit_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model["saver"].save(sess, OUTPUT_MODEL_PATH)
                print(f"  -> Lưu model tốt nhất (acc={best_val_acc:.4f})")

    print(f"Huấn luyện hoàn tất. Model tốt nhất với accuracy={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
