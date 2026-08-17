"""
train_embedding_denoiser.py

Huấn luyện Embedding Denoising Network (512 -> 256 -> 128 -> 256 -> 512).
Sử dụng tên biến cố định để tương thích với script đánh giá.
"""

import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from collections import defaultdict
import csv

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536
DIM = 512
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
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "embedding_denoiser_model"
)

BATCH_SIZE = 32
N_EPOCHS = 200
LEARNING_RATE = 1e-3
VALID_FRAC = 0.15


def load_embedding(name, imagenum):
    return np.load(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def load_all_identities():
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
    for ident in id_to_images:
        id_to_images[ident] = list(set(id_to_images[ident]))
    return id_to_images


def build_training_data(handler, id_to_images):
    X, Y = [], []
    for ident, images in id_to_images.items():
        if len(images) < 2:
            continue
        # Lấy projected vectors cho tất cả ảnh
        projs = []
        for name, imagenum in images:
            emb = load_embedding(name, imagenum)
            proj = np.dot(emb, handler.M_matrix).astype(np.float32)
            projs.append(proj)
        # Tạo cặp (proj_verify, proj_enroll)
        for i in range(len(projs)):
            for j in range(len(projs)):
                if i == j:
                    continue
                X.append(projs[j])  # verify (noisy)
                Y.append(projs[i])  # enroll (clean)
    return np.array(X), np.array(Y)


def build_model():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, DIM], name="noisy_proj")
        y = tf.placeholder(tf.float32, [None, DIM], name="clean_proj")

        # Tên lớp được cố định rõ ràng
        h1 = tf.layers.dense(x, 256, activation=tf.nn.relu, name="enc1")
        h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu, name="enc2")
        h3 = tf.layers.dense(h2, 256, activation=tf.nn.relu, name="dec1")
        pred = tf.layers.dense(
            h3, DIM, activation=None, name="pred_proj"
        )  # Tên này sẽ được lưu

        loss = tf.reduce_mean(tf.square(y - pred))
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    return {
        "graph": graph,
        "x": x,
        "y": y,
        "loss": loss,
        "train_op": train_op,
        "init": init,
        "saver": saver,
    }


def main():
    print("Khởi tạo handler nhẹ...")
    handler = WiFaKeyHandler()
    handler.sess = None  # Chỉ dùng M_matrix

    print("Xây dựng dữ liệu...")
    id_to_images = load_all_identities()
    X, Y = build_training_data(handler, id_to_images)
    print(f"Số cặp: {X.shape[0]}")

    # Chia train/val
    n_val = int(len(X) * VALID_FRAC)
    idx = np.random.permutation(len(X))
    X_train, Y_train = X[idx[n_val:]], Y[idx[n_val:]]
    X_val, Y_val = X[idx[:n_val]], Y[idx[:n_val]]

    model = build_model()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=model["graph"], config=config) as sess:
        sess.run(model["init"])
        best_val_loss = float("inf")
        for epoch in range(N_EPOCHS):
            # Train
            perm = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), BATCH_SIZE):
                batch_idx = perm[start : start + BATCH_SIZE]
                feed = {model["x"]: X_train[batch_idx], model["y"]: Y_train[batch_idx]}
                sess.run(model["train_op"], feed_dict=feed)
            # Val
            val_loss = sess.run(
                model["loss"], feed_dict={model["x"]: X_val, model["y"]: Y_val}
            )
            print(f"Epoch {epoch+1:3d}: val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model["saver"].save(sess, MODEL_PATH)
                print("  -> Lưu model tốt nhất")
    print("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
