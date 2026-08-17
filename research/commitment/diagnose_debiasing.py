"""
diagnose_debiasing.py (ĐÃ TỐI ƯU – tránh OOM)

Đo thiên lệch bit, đề xuất ngưỡng mới, mô phỏng debiasing.
Chỉ dùng MỘT handler duy nhất, binarize thủ công với ngưỡng mới.
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
    """
    Binarize với ngưỡng tùy chỉnh cho từng chiều.
    custom_thresholds: (N_DIMS, N_THRESHOLDS) – mỗi hàng là 3 ngưỡng (đã sort tăng dần).
    Trả về bits (1536,) và margin (1536,).
    """
    bits = np.zeros(FULL_BITS, dtype=np.uint8)
    margin = np.zeros(FULL_BITS, dtype=np.float32)
    for d in range(N_DIMS):
        thr = np.sort(custom_thresholds[d])[::-1]  # Giảm dần để khớp code gốc
        for t_idx in range(N_THRESHOLDS):
            pos = d * 3 + t_idx
            bits[pos] = projected[d] >= thr[t_idx]
            margin[pos] = abs(projected[d] - thr[t_idx])
    return bits, margin


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    # ---- 1. Thu thập projected values để ước lượng ngưỡng mới ----
    pairs_est = load_genuine_pairs(max_pairs=400)
    all_projected = []
    for name_e, img_e, name_v, img_v in pairs_est:
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)
        all_projected.append(np.dot(emb_e, handler.M_matrix))
        all_projected.append(np.dot(emb_v, handler.M_matrix))
    all_projected = np.array(all_projected)  # (N, 512)

    # ---- 2. Đo thiên lệch hiện tại ----
    original_thresholds = np.sort(handler.intervals)
    p_bit1_original = np.zeros(FULL_BITS)
    for d in range(N_DIMS):
        for t_idx, thr in enumerate(original_thresholds):
            pos = d * 3 + t_idx
            p_bit1_original[pos] = np.mean(all_projected[:, d] >= thr)
    p = np.clip(p_bit1_original, 1e-12, 1 - 1e-12)
    entropy_original = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    print(f"Entropy trung bình hiện tại: {np.mean(entropy_original):.4f} bit")

    # ---- 3. Đề xuất ngưỡng mới (quantile 25, 50, 75 cho từng chiều) ----
    new_thresholds = np.zeros((N_DIMS, N_THRESHOLDS))
    for d in range(N_DIMS):
        vals = all_projected[:, d]
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        new_thresholds[d] = [q25, q50, q75]

    # Đo entropy sau debiasing
    p_bit1_new = np.zeros(FULL_BITS)
    for d in range(N_DIMS):
        thr_sorted = np.sort(new_thresholds[d])[::-1]
        for t_idx, thr in enumerate(thr_sorted):
            pos = d * 3 + t_idx
            p_bit1_new[pos] = np.mean(all_projected[:, d] >= thr)
    p_new = np.clip(p_bit1_new, 1e-12, 1 - 1e-12)
    entropy_new = -(p_new * np.log2(p_new) + (1 - p_new) * np.log2(1 - p_new))
    print(f"Entropy trung bình sau debiasing: {np.mean(entropy_new):.4f} bit")

    # ---- 4. Đánh giá GMR ----
    emp_mod = EmpiricalLLR(
        lookup_path=os.path.join(
            _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
        )
    )
    test_pairs = load_genuine_pairs(max_pairs=200)
    pass_original = 0
    pass_debiased = 0
    n_test = 0

    for name_e, img_e, name_v, img_v in test_pairs:
        n_test += 1
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)
        proj_e = np.dot(emb_e, handler.M_matrix)
        proj_v = np.dot(emb_v, handler.M_matrix)

        # --- Pipeline gốc ---
        helper_orig, sel_mask_orig, key_hash_orig = handler.enroll(emb_e)
        ok_orig = handler.verify(emb_v, helper_orig, sel_mask_orig, key_hash_orig)
        if ok_orig:
            pass_original += 1

        # --- Pipeline debiased ---
        bits_e_new, margin_e_new = binarize_with_custom_thresholds(
            proj_e, new_thresholds
        )
        bits_v_new, margin_v_new = binarize_with_custom_thresholds(
            proj_v, new_thresholds
        )

        # Margin selection
        selection_indices = np.argpartition(-margin_e_new, handler.feature_length)[
            : handler.feature_length
        ]
        selection_indices.sort()
        b_selected_e = bits_e_new[selection_indices]

        rng = np.random.default_rng()
        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_new = np.logical_xor(b_selected_e, codeword).astype(np.uint8)
        key_hash_new = hashlib.sha256(random_key.flatten().tobytes()).digest()

        b_selected_v = bits_v_new[selection_indices]
        noisy = np.logical_xor(b_selected_v, helper_new).astype(np.uint8)
        margin_sel = margin_v_new[selection_indices]
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        llr = llr.reshape(1, N, Z)
        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash_new:
            pass_debiased += 1

    print(f"\nGMR gốc: {pass_original}/{n_test} ({100*pass_original/n_test:.2f}%)")
    print(
        f"GMR sau debiasing: {pass_debiased}/{n_test} ({100*pass_debiased/n_test:.2f}%)"
    )
    handler.sess.close()


if __name__ == "__main__":
    main()
