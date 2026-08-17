"""
train_llr_corrector_v2.py

Huấn luyện Neural LLR Correction (hướng L) với đầu vào là Empirical LLR
thay vì BPSK thô. Huấn luyện trên toàn bộ LFW (881 cặp), tùy chọn thêm CPLFW.

Nếu không có file reliability_lookup.npz, script sẽ tự xây dựng empirical LLR
từ tập train (không cần file ngoài).

ách chạy: nếu đã có reliability_lookup.npz, truyền đường dẫn vào --empirical-lookup:

python research/commitment/train_llr_corrector_v2.py --empirical-lookup experiments/out_step3/reliability_lookup.npz --model-dir checkpoints/neural_llr_corrector_v2

Nếu không có thì chạy:
python research/commitment/train_llr_corrector_v2.py --epochs 50 --batch-size 32 --model-dir checkpoints/neural_llr_corrector_v2

thêm cplfw:
python research/commitment/train_llr_corrector_v2.py --add-cplfw --model-dir checkpoints/neural_llr_corrector_v2
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


def margin_to_llr_empirical(margin, margin_bp, p_bp, eps=1e-6):
    """Nội suy tuyến tính từ bảng (margin_bp, p_bp) -> LLR."""
    p = np.interp(margin, margin_bp, p_bp, left=p_bp[0], right=p_bp[-1])
    p = np.clip(p, eps, 0.5 - eps)
    return np.log((1.0 - p) / p).astype(np.float32)


def build_empirical_lookup(handler, pairs_iter, num_pairs, n_bins=100):
    """Quét tập train để xây dựng bảng tra cứu empirical LLR."""
    all_margins = []
    all_errors = []  # 1 nếu bit lỗi, 0 nếu đúng
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

        # Verify
        b_full_verify = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_selected_verify = b_full_verify[idx]

        # Lỗi bit (so sánh trực tiếp bit sinh trắc, không qua mã hóa)
        error_bits = (b_selected_enroll != b_selected_verify).astype(np.float32)

        # Margin
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx]

        all_margins.append(margin_sel)
        all_errors.append(error_bits)
        count += 1

    all_margins = np.concatenate(all_margins)
    all_errors = np.concatenate(all_errors)

    # Chia bin theo margin
    bin_edges = np.linspace(all_margins.min(), all_margins.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    p_bp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (all_margins >= bin_edges[i]) & (all_margins < bin_edges[i + 1])
        if mask.sum() > 0:
            p_bp[i] = all_errors[mask].mean()
        else:
            p_bp[i] = 0.5  # giá trị mặc định nếu bin rỗng
    # Đảm bảo p_bp nằm trong (0,0.5)
    p_bp = np.clip(p_bp, 1e-6, 0.5 - 1e-6)
    # Lưu bảng (có thể tái sử dụng)
    return bin_centers.astype(np.float32), p_bp.astype(np.float32)


def build_training_data(
    handler, pairs_iter, num_pairs, empirical_lookup=None, margin_bp=None, p_bp=None
):
    """
    Tạo dữ liệu huấn luyện: margin, LLR empirical (đầu vào), target codeword ±1.
    Nếu empirical_lookup là None, sẽ dùng margin_bp, p_bp để tự tính.
    """
    all_margins = []
    all_llr_emp = []
    all_targets = []

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

        random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected_enroll, codeword).astype(np.uint8)

        # Verify
        b_full_verify = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_selected_verify = b_full_verify[idx]
        noisy = np.logical_xor(b_selected_verify, helper_data).astype(np.uint8)

        # Margin
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[idx]

        # Empirical LLR
        if empirical_lookup is not None:
            from research.modulation.v2_empirical_llr import EmpiricalLLR

            emp_mod = EmpiricalLLR(lookup_path=empirical_lookup)
            llr_emp = emp_mod.modulate(noisy, context={"margin": margin_sel})
        else:
            # Tự tính từ bảng margin_bp, p_bp
            llr_mag = margin_to_llr_empirical(margin_sel, margin_bp, p_bp)
            sign = 2 * noisy.astype(np.float32) - 1
            llr_emp = sign * llr_mag

        # Target: codeword bits dạng ±1
        target = 2 * codeword.astype(np.float32) - 1

        all_margins.append(margin_sel)
        all_llr_emp.append(llr_emp)
        all_targets.append(target)
        count += 1

    return (
        np.array(all_margins, dtype=np.float32),
        np.array(all_llr_emp, dtype=np.float32),
        np.array(all_targets, dtype=np.float32),
    )


def build_model(margin_input, llr_input, is_training):
    features = tf.stack([margin_input, llr_input], axis=-1)  # (batch, 832, 2)
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
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument(
        "--empirical-lookup",
        default=None,
        help="Đường dẫn đến reliability_lookup.npz (nếu có)",
    )
    ap.add_argument(
        "--add-cplfw", action="store_true", help="Thêm dữ liệu CPLFW vào tập huấn luyện"
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--model-dir", default="./checkpoints/neural_llr_corrector_v2")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")

    # LFW paths
    lfw_pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "pairs"
    )
    lfw_cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "embeddings_cache"
    )
    lfw_csv = os.path.join(lfw_pairs_dir, f"{args.tier}_genuine.csv")

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    # Chuẩn bị dữ liệu huấn luyện
    # Số cặp LFW tối đa: 881
    lfw_iter = genuine_pairs_iter(lfw_csv, lfw_cache_dir, max_pairs=881)
    num_lfw = 881

    # Nếu có thêm CPLFW
    if args.add_cplfw:
        cplfw_pairs_dir = os.path.join(root, "datasets", "processed", "cplfw", "pairs")
        cplfw_cache_dir = os.path.join(
            root, "datasets", "processed", "cplfw", "embeddings_cache"
        )
        cplfw_csv = os.path.join(cplfw_pairs_dir, "select_genuine.csv")
        cplfw_iter = genuine_pairs_iter(cplfw_csv, cplfw_cache_dir, max_pairs=1694)
        num_cplfw = 1694
    else:
        cplfw_iter = None
        num_cplfw = 0

    # Nếu không có empirical lookup, tự xây dựng từ tập train (LFW)
    margin_bp, p_bp = None, None
    if args.empirical_lookup is None:
        print("Xây dựng empirical lookup từ tập LFW...")
        lfw_iter_for_lookup = genuine_pairs_iter(
            lfw_csv, lfw_cache_dir, max_pairs=400
        )  # dùng 400 cặp để làm lookup
        margin_bp, p_bp = build_empirical_lookup(handler, lfw_iter_for_lookup, 400)
        print(f"Đã tạo lookup với {len(margin_bp)} bins.")
        # Reset iterator cho LFW
        lfw_iter = genuine_pairs_iter(lfw_csv, lfw_cache_dir, max_pairs=881)

    # Tạo dữ liệu huấn luyện từ LFW
    print("Tạo dữ liệu huấn luyện từ LFW...")
    margins_lfw, llr_emp_lfw, targets_lfw = build_training_data(
        handler,
        lfw_iter,
        num_lfw,
        empirical_lookup=args.empirical_lookup,
        margin_bp=margin_bp,
        p_bp=p_bp,
    )
    margins, llr_emp, targets = margins_lfw, llr_emp_lfw, targets_lfw

    # Nếu có CPLFW
    if cplfw_iter is not None:
        print("Thêm dữ liệu CPLFW...")
        margins_cplfw, llr_emp_cplfw, targets_cplfw = build_training_data(
            handler,
            cplfw_iter,
            num_cplfw,
            empirical_lookup=args.empirical_lookup,
            margin_bp=margin_bp,
            p_bp=p_bp,
        )
        margins = np.concatenate([margins, margins_cplfw], axis=0)
        llr_emp = np.concatenate([llr_emp, llr_emp_cplfw], axis=0)
        targets = np.concatenate([targets, targets_cplfw], axis=0)

    print(f"Tổng số mẫu huấn luyện: {margins.shape[0]} cặp, {margins.size} bit.")

    # Xáo trộn
    idx = np.random.permutation(margins.shape[0])
    margins = margins[idx]
    llr_emp = llr_emp[idx]
    targets = targets[idx]

    # Xây dựng mô hình
    tf.reset_default_graph()
    margin_ph = tf.placeholder(tf.float32, [None, 832], name="margin")
    llr_ph = tf.placeholder(tf.float32, [None, 832], name="llr_emp")
    target_ph = tf.placeholder(tf.float32, [None, 832], name="target")
    is_training = tf.placeholder_with_default(False, shape=[], name="is_training")

    pred_llr = build_model(margin_ph, llr_ph, is_training)

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

    os.makedirs(args.model_dir, exist_ok=True)
    saver.save(sess, os.path.join(args.model_dir, "model"))
    print(f"Đã lưu model vào {args.model_dir}")

    sess.close()
    handler.sess.close()


if __name__ == "__main__":
    main()
