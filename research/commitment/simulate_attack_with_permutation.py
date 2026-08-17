"""
simulate_attack_with_permutation.py

So sánh hiệu quả tấn công:
- Không hoán vị: attacker dùng prior toàn cục thật.
- Có hoán vị bí mật: attacker buộc dùng prior ngẫu nhiên (p=0.5).
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
KEY_LENGTH = 160


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


def estimate_prior_global(handler, pairs):
    """Ước tính prior thật cho từng vị trí 1536."""
    count_selected = np.zeros(FULL_BITS)
    count_bit1 = np.zeros(FULL_BITS)
    count_total = 0

    for name_e, img_e, name_v, img_v in pairs:
        emb = load_embedding(name_e, img_e)
        b_full = handler._binarize_full(emb).astype(np.uint8)
        projected = np.dot(emb, handler.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, handler.intervals)

        selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
        for pos in selection_indices:
            count_selected[pos] += 1
            if b_full[pos] == 1:
                count_bit1[pos] += 1
        count_total += 1

    p_selected = count_selected / count_total
    p_bit1 = np.divide(
        count_bit1,
        count_selected,
        out=np.full(FULL_BITS, 0.5),
        where=count_selected > 0,
    )
    p_bit1 = np.clip(p_bit1, 0.01, 0.99)
    return p_selected, p_bit1


def generate_secret_permutation(seed):
    """Tạo hoán vị bí mật từ seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(FULL_BITS)
    return perm


def try_decode_with_llr(handler, noisy_bits, margin, key_hash, emp_mod):
    """Giải mã với Empirical LLR."""
    llr = emp_mod.modulate(noisy_bits, context={"margin": margin}).flatten()
    llr = llr.reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[:KEY_LENGTH]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def combinatorial_attack(
    handler, helper_data, sel_mask, key_hash, prior_1536, emp_mod, max_attempts=50000
):
    """Tấn công tổ hợp (giống simulate_attack_v2.py)."""
    selected_positions = np.where(sel_mask == 1)[0]
    p = prior_1536[selected_positions]
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    sorted_local_indices = np.argsort(entropy)

    base_bits = (p > 0.5).astype(np.uint8)
    margin_fake = np.zeros(FEATURE_LENGTH)
    margin_fake[sorted_local_indices] = np.linspace(2.0, 0.1, FEATURE_LENGTH)

    # Thử base candidate
    noisy_base = np.logical_xor(base_bits, helper_data).astype(np.uint8)
    if try_decode_with_llr(handler, noisy_base, margin_fake, key_hash, emp_mod):
        return 1

    attempt_count = 1
    flip_candidates = sorted_local_indices[:30]

    from itertools import combinations

    for L in range(1, min(6, len(flip_candidates) + 1)):
        for combo in combinations(flip_candidates, L):
            attempt_count += 1
            if attempt_count > max_attempts:
                return None
            candidate = base_bits.copy()
            for idx in combo:
                candidate[idx] ^= 1
            noisy = np.logical_xor(candidate, helper_data).astype(np.uint8)
            if try_decode_with_llr(handler, noisy, margin_fake, key_hash, emp_mod):
                return attempt_count
    return None


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    emp_mod = EmpiricalLLR(
        lookup_path=os.path.join(
            _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
        )
    )

    # Ước tính prior toàn cục (từ tập train)
    train_pairs = load_genuine_pairs(max_pairs=400)
    p_selected, p_bit1 = estimate_prior_global(handler, train_pairs)

    # Tạo prior ngẫu nhiên (mô phỏng attacker không biết ánh xạ khi có hoán vị)
    prior_random = np.full(FULL_BITS, 0.5)

    # Test trên tập test
    test_pairs = load_genuine_pairs(max_pairs=100)
    results_no_perm = []
    results_with_perm = []

    for name_e, img_e, name_v, img_v in test_pairs:
        emb_e = load_embedding(name_e, img_e)
        helper, sel_mask, key_hash = handler.enroll(emb_e)

        # --- Tấn công không hoán vị (dùng prior thật) ---
        attempts = combinatorial_attack(
            handler, helper, sel_mask, key_hash, p_bit1, emp_mod, max_attempts=20000
        )
        if attempts is not None:
            results_no_perm.append(attempts)

        # --- Tấn công có hoán vị (attacker dùng prior ngẫu nhiên) ---
        attempts_perm = combinatorial_attack(
            handler,
            helper,
            sel_mask,
            key_hash,
            prior_random,
            emp_mod,
            max_attempts=20000,
        )
        if attempts_perm is not None:
            results_with_perm.append(attempts_perm)

    print(f"Số cặp test: {len(test_pairs)}")
    print(f"\n=== KHÔNG HOÁN VỊ (attacker dùng prior thật) ===")
    print(f"Thành công: {len(results_no_perm)}/{len(test_pairs)}")
    if results_no_perm:
        print(f"Số lần thử TB: {np.mean(results_no_perm):.2f}")

    print(f"\n=== CÓ HOÁN VỊ (attacker dùng prior ngẫu nhiên) ===")
    print(f"Thành công: {len(results_with_perm)}/{len(test_pairs)}")
    if results_with_perm:
        print(f"Số lần thử TB: {np.mean(results_with_perm):.2f}")

    handler.sess.close()


if __name__ == "__main__":
    main()
