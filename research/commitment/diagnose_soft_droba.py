"""
diagnose_soft_droba.py

Soft DROBA kết hợp với Margin Selection.
Dùng trọng số từ Fisher ratio và entropy để ưu tiên bit khi chọn 832.
"""

import os, sys, csv, hashlib, numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR

N, m, Z = 52, 42, 16
FULL_BITS = 1536
FEATURE_LENGTH = 832
N_DIMS = 512


def load_embedding(name, imagenum):
    path = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "embeddings_cache",
        f"{name}_{int(imagenum):04d}.npy",
    )
    return np.load(path)


def load_genuine_pairs(max_pairs=None):
    pairs_csv = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "pairs",
        "tune_genuine.csv",
    )
    pairs = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                (
                    row["name_enroll"],
                    int(row["imagenum_enroll"]),
                    row["name_verify"],
                    int(row["imagenum_verify"]),
                )
            )
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def compute_dimension_weights(handler, pairs):
    """Tính trọng số cho từng chiều dựa trên Fisher ratio và entropy."""
    # Thu thập projected values
    all_proj = []
    for name_e, img_e, _, _ in pairs:
        emb = load_embedding(name_e, img_e)
        all_proj.append(np.dot(emb, handler.M_matrix))
    all_proj = np.array(all_proj)  # (N, 512)

    # Tính Fisher ratio cho từng chiều
    intra_vars = np.zeros(N_DIMS)
    for name_e, img_e, name_v, img_v in pairs:
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)
        proj_e = np.dot(emb_e, handler.M_matrix)
        proj_v = np.dot(emb_v, handler.M_matrix)
        intra_vars += (proj_e - proj_v) ** 2
    intra_vars /= len(pairs)
    inter_vars = np.var(all_proj, axis=0)
    fisher = inter_vars / (intra_vars + 1e-8)

    # Tính entropy cho từng bit (trên toàn bộ 1536 bit)
    p_bit1 = np.zeros(FULL_BITS)
    bit_counts = np.zeros(FULL_BITS)
    for name_e, img_e, _, _ in pairs:
        emb = load_embedding(name_e, img_e)
        b_full = handler._binarize_full(emb).astype(np.uint8)
        bit_counts += b_full
    p_bit1 = bit_counts / len(pairs)
    p = np.clip(p_bit1, 1e-6, 1 - 1e-6)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    # Tính entropy trung bình cho mỗi chiều (3 bit)
    dim_entropy = np.mean(entropy.reshape(N_DIMS, 3), axis=1)

    # Kết hợp thành trọng số: w_d = fisher^alpha * dim_entropy^beta
    # Chuẩn hóa về [0.5, 1.5] để tránh thay đổi quá lớn
    w_raw = np.sqrt(fisher) * dim_entropy  # có thể điều chỉnh alpha, beta
    w = 0.5 + (w_raw - w_raw.min()) / (
        w_raw.max() - w_raw.min()
    )  # chuẩn hóa về [0.5, 1.5]

    print(f"Trọng số chiều: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")
    return w


def margin_selection_weighted(margin, dim_weights):
    """Chọn 832 bit với margin được nhân trọng số chiều."""
    # margin: (1536,)
    # dim_weights: (512,) -> lặp 3 lần cho 3 bit
    weights_1536 = np.repeat(dim_weights, 3)
    score = margin * weights_1536
    selection_indices = np.argpartition(-score, FEATURE_LENGTH)[:FEATURE_LENGTH]
    selection_indices.sort()
    return selection_indices


def run_test(handler, test_pairs, dim_weights, emp_mod, use_droba=False):
    pass_count = 0
    for name_e, img_e, name_v, img_v in test_pairs:
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)
        proj_e = np.dot(emb_e, handler.M_matrix)
        proj_v = np.dot(emb_v, handler.M_matrix)

        bits_e, margin_e = binarize_with_perbit_confidence(proj_e, handler.intervals)
        bits_v, margin_v = binarize_with_perbit_confidence(proj_v, handler.intervals)

        if use_droba:
            sel_idx = margin_selection_weighted(margin_e, dim_weights)
        else:
            # Baseline: margin selection thuần
            sel_idx = np.argpartition(-margin_e, FEATURE_LENGTH)[:FEATURE_LENGTH]
            sel_idx.sort()

        b_sel_e = bits_e[sel_idx]
        rng = np.random.default_rng()
        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper = np.logical_xor(b_sel_e, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        b_sel_v = bits_v[sel_idx]
        noisy = np.logical_xor(b_sel_v, helper).astype(np.uint8)
        margin_sel = margin_v[sel_idx]
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        llr = llr.reshape(1, N, Z)
        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            pass_count += 1
    return pass_count


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    emp_mod = EmpiricalLLR(
        lookup_path=os.path.join(
            _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
        )
    )

    all_pairs = load_genuine_pairs()
    n_train = int(len(all_pairs) * 0.7)
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train : n_train + 200]

    print("1. Tính trọng số DROBA cho từng chiều...")
    dim_weights = compute_dimension_weights(handler, train_pairs)

    print("\n2. Đánh giá Baseline (Margin Selection thuần)...")
    pass_baseline = run_test(handler, test_pairs, dim_weights, emp_mod, use_droba=False)
    gmr_baseline = 100 * pass_baseline / len(test_pairs)
    print(f"   GMR Baseline: {pass_baseline}/{len(test_pairs)} ({gmr_baseline:.2f}%)")

    print("\n3. Đánh giá Soft DROBA...")
    pass_droba = run_test(handler, test_pairs, dim_weights, emp_mod, use_droba=True)
    gmr_droba = 100 * pass_droba / len(test_pairs)
    print(f"   GMR Soft DROBA: {pass_droba}/{len(test_pairs)} ({gmr_droba:.2f}%)")

    print(f"\nChênh lệch: {gmr_droba - gmr_baseline:+.2f} điểm %")
    handler.sess.close()


if __name__ == "__main__":
    main()
