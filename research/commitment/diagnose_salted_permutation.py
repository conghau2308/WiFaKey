"""
diagnose_salted_permutation.py (ĐÃ SỬA LỖI)

Đánh giá tác động của Public Salted Permutation lên GMR.
So sánh:
  - Baseline: Margin Selection + Empirical LLR (không salt)
  - Salted:   Margin Selection + Empirical LLR + Salted Permutation
"""

import os
import sys
import csv
import hashlib
import numpy as np
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


# --- Loaders (giữ nguyên) ---
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


def generate_permutation(seed_bytes):
    """Tạo một hoán vị của 1536 phần tử từ seed."""
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
    perm = rng.permutation(FULL_BITS)
    return perm


def apply_permutation(bits, perm):
    """Áp dụng hoán vị lên mảng bits."""
    return bits[perm]


# --- Handler với salted permutation ---
class SaltedPermutationHandler(SecureWiFaKeyHandler):
    """Handler thực hiện salted permutation trước khi chọn bit."""

    def __init__(self, salt_bytes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.salt_bytes = salt_bytes
        self.permutation = generate_permutation(salt_bytes)

    def enroll(self, feature_vector_float):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        b_full_permuted = apply_permutation(b_full, self.permutation)
        projected = np.dot(feature_vector_float, self.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, self.intervals)
        margin_permuted = apply_permutation(margin, self.permutation)

        selection_indices = np.argpartition(-margin_permuted, FEATURE_LENGTH)[
            :FEATURE_LENGTH
        ]
        selection_indices.sort()
        selection_mask = np.zeros(FULL_BITS, dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected = b_full_permuted[selection_indices]

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()
        return helper_data, selection_mask, key_hash

    def verify(
        self, feature_vector_float, helper_data, selection_mask, stored_key_hash
    ):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        b_full_permuted = apply_permutation(b_full, self.permutation)
        idx = np.where(selection_mask == 1)[0]
        b_sel = b_full_permuted[idx]
        noisy = np.logical_xor(b_sel, helper_data).astype(np.uint8)

        projected = np.dot(feature_vector_float, self.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, self.intervals)
        margin_permuted = apply_permutation(margin, self.permutation)
        margin_sel = margin_permuted[idx]

        emp_mod = EmpiricalLLR(
            lookup_path=os.path.join(
                _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
            )
        )
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        llr = llr.reshape(1, N, Z)
        y_pred = self.sess.run(self.decoder_output, feed_dict={self.xa: llr}).flatten()
        decoded_key = (y_pred > 0).astype(int)[: self.key_length]
        recon_hash = hashlib.sha256(decoded_key.tobytes()).digest()
        return recon_hash == stored_key_hash


# --- Hàm chính ---
def run_test(handler, test_pairs, cache_dir):
    """Chạy đánh giá GMR trên tập test (test_pairs là list các dict)."""
    pass_count = 0
    for row in test_pairs:
        emb_e = load_embedding(
            cache_dir, row["name_enroll"], int(row["imagenum_enroll"])
        )
        emb_v = load_embedding(
            cache_dir, row["name_verify"], int(row["imagenum_verify"])
        )
        helper, sel_mask, key_hash = handler.enroll(emb_e)
        if handler.verify(emb_v, helper, sel_mask, key_hash):
            pass_count += 1
    return pass_count


def main():
    # Đường dẫn
    pairs_dir = os.path.join(
        _PROJECT_ROOT, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    cache_dir = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "embeddings_cache",
    )
    pairs_csv = os.path.join(pairs_dir, "tune_genuine.csv")
    test_pairs = load_pairs(pairs_csv, max_pairs=200)

    # 1. Baseline (không salt)
    print("=== BASELINE (không salted permutation) ===")
    handler_baseline = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    pass_baseline = run_test(handler_baseline, test_pairs, cache_dir)
    gmr_baseline = 100 * pass_baseline / len(test_pairs)
    print(f"Baseline GMR: {pass_baseline}/{len(test_pairs)} ({gmr_baseline:.2f}%)")
    handler_baseline.sess.close()

    # 2. Salted Permutation
    print("\n=== SALTED PERMUTATION ===")
    rng = np.random.default_rng(42)
    salt_bytes = rng.bytes(32)
    handler_salted = SaltedPermutationHandler(
        salt_bytes, data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    pass_salted = run_test(handler_salted, test_pairs, cache_dir)
    gmr_salted = 100 * pass_salted / len(test_pairs)
    print(f"Salted GMR: {pass_salted}/{len(test_pairs)} ({gmr_salted:.2f}%)")
    handler_salted.sess.close()

    print("\n=== KẾT LUẬN ===")
    if gmr_salted >= gmr_baseline - 1.0:
        print("Salted Permutation KHÔNG làm giảm GMR đáng kể.")
        print("Có thể tích hợp an toàn vào pipeline cuối cùng.")
    else:
        print("CẢNH BÁO: Salted Permutation làm giảm GMR! Cần điều tra thêm.")


if __name__ == "__main__":
    main()
