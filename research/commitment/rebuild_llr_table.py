"""
rebuild_llr_table.py

Xây dựng bảng Empirical LLR mới dựa trên ngưỡng debiased.
- Thu thập margin và lỗi bit từ tập train.
- Xây dựng bảng tra cứu margin → P(lỗi).
- Đánh giá GMR với bảng LLR mới.
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
N_DIMS = 512
N_THRESHOLDS = 3
FEATURE_LENGTH = 832


# --- Loaders ---
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


def binarize_with_custom_thresholds(projected, custom_thresholds):
    """Binarize với ngưỡng tùy chỉnh cho từng chiều."""
    bits = np.zeros(FULL_BITS, dtype=np.uint8)
    margin = np.zeros(FULL_BITS, dtype=np.float32)
    for d in range(N_DIMS):
        thr = np.sort(custom_thresholds[d])[::-1]  # Giảm dần để khớp code gốc
        for t_idx in range(N_THRESHOLDS):
            pos = d * 3 + t_idx
            bits[pos] = projected[d] >= thr[t_idx]
            margin[pos] = abs(projected[d] - thr[t_idx])
    return bits, margin


def build_empirical_lookup(margins, errors, n_bins=100):
    """Xây dựng bảng tra cứu margin → P(lỗi)."""
    all_margins = np.concatenate(margins)
    all_errors = np.concatenate(errors)

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


def margin_to_llr(margin, margin_bp, p_bp, eps=1e-6):
    """Tra cứu bảng để chuyển margin → LLR."""
    p = np.interp(margin, margin_bp, p_bp, left=p_bp[0], right=p_bp[-1])
    p = np.clip(p, eps, 0.5 - eps)
    return np.log((1.0 - p) / p).astype(np.float32)


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    # ---- 1. Tính ngưỡng mới (quantile) từ tập train ----
    print("1. Tính ngưỡng debiased...")
    train_pairs = load_genuine_pairs(max_pairs=400)
    all_projected = []
    for name_e, img_e, name_v, img_v in train_pairs:
        emb = load_embedding(name_e, img_e)
        all_projected.append(np.dot(emb, handler.M_matrix))
        emb_v = load_embedding(name_v, img_v)
        all_projected.append(np.dot(emb_v, handler.M_matrix))
    all_projected = np.array(all_projected)

    new_thresholds = np.zeros((N_DIMS, N_THRESHOLDS))
    for d in range(N_DIMS):
        vals = all_projected[:, d]
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        new_thresholds[d] = [q25, q50, q75]

    # ---- 2. Thu thập margin và lỗi bit với ngưỡng mới ----
    print("2. Thu thập margin và lỗi bit...")
    margin_list = []
    error_list = []

    for name_e, img_e, name_v, img_v in train_pairs:
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)
        proj_e = np.dot(emb_e, handler.M_matrix)
        proj_v = np.dot(emb_v, handler.M_matrix)

        bits_e, margin_e = binarize_with_custom_thresholds(proj_e, new_thresholds)
        bits_v, margin_v = binarize_with_custom_thresholds(proj_v, new_thresholds)

        errors = (bits_e != bits_v).astype(np.float32)
        margin_list.append(margin_v)
        error_list.append(errors)

    # ---- 3. Xây dựng bảng LLR mới ----
    print("3. Xây dựng bảng LLR mới...")
    margin_bp, p_bp = build_empirical_lookup(margin_list, error_list)
    lookup_path = os.path.join(
        os.path.dirname(__file__), "reliability_lookup_debiased.npz"
    )
    np.savez_compressed(
        lookup_path,
        margin_breakpoints=margin_bp,
        p_breakpoints=p_bp,
        eps=np.float32(1e-6),
    )
    print(f"   Đã lưu bảng LLR mới vào {lookup_path}")

    # ---- 4. Đánh giá GMR với bảng LLR mới ----
    print("4. Đánh giá GMR với bảng LLR mới...")
    test_pairs = load_genuine_pairs(max_pairs=200)
    pass_debiased = 0
    n_test = 0

    for name_e, img_e, name_v, img_v in test_pairs:
        n_test += 1
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)
        proj_e = np.dot(emb_e, handler.M_matrix)
        proj_v = np.dot(emb_v, handler.M_matrix)

        # Binarize với ngưỡng mới
        bits_e_new, margin_e_new = binarize_with_custom_thresholds(
            proj_e, new_thresholds
        )
        bits_v_new, margin_v_new = binarize_with_custom_thresholds(
            proj_v, new_thresholds
        )

        # Margin selection
        selection_indices = np.argpartition(-margin_e_new, FEATURE_LENGTH)[
            :FEATURE_LENGTH
        ]
        selection_indices.sort()
        b_selected_e = bits_e_new[selection_indices]

        # Tạo key và helper_data
        rng = np.random.default_rng()
        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper = np.logical_xor(b_selected_e, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        # Verify với bảng LLR mới
        b_selected_v = bits_v_new[selection_indices]
        noisy = np.logical_xor(b_selected_v, helper).astype(np.uint8)
        margin_sel = margin_v_new[selection_indices]

        # Dùng bảng LLR mới (thay vì EmpiricalLLR module)
        llr_mag = margin_to_llr(margin_sel, margin_bp, p_bp)
        sign = 2 * noisy.astype(np.float32) - 1
        llr = (sign * llr_mag).reshape(1, N, Z)

        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            pass_debiased += 1

    gmr = 100 * pass_debiased / n_test
    print(f"\nGMR với bảng LLR mới: {pass_debiased}/{n_test} ({gmr:.2f}%)")

    handler.sess.close()


if __name__ == "__main__":
    main()
