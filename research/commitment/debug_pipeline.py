"""
debug_pipeline.py

Mô phỏng pipeline với embedding giả (toàn 0.5) để so sánh với Rust.
"""

import numpy as np
import os
import sys
import hashlib

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR
from wifakey_module.wifakey_lib import Encode


def main():
    # 1. Embedding giả
    embedding = np.full(512, 0.5, dtype=np.float64)

    # 2. BioHashing
    secret = b"test-secret-12345678"
    seed = int.from_bytes(secret, "big") % (2**32)
    rng = np.random.default_rng(seed)
    M_bio = rng.normal(0, 1, size=(512, 512)).astype(np.float32)
    M_bio = M_bio / np.linalg.norm(M_bio, axis=1, keepdims=True)
    v_proj = np.dot(embedding, M_bio.T)
    norm = np.linalg.norm(v_proj)
    if norm > 0:
        v_proj = v_proj / norm

    # 3. Load M_matrix và intervals
    M = np.load(os.path.join(_PROJECT_ROOT, "wifakey_module", "data", "M_matrix.npy"))
    intervals = np.load(
        os.path.join(
            _PROJECT_ROOT, "wifakey_module", "data", "binarization_intervals.npy"
        )
    )

    # 4. Binarization
    bits, margin = binarize_with_perbit_confidence(np.dot(v_proj, M), intervals)

    # 5. Salted Permutation
    salt = b"test-salt-87654321"
    seed_salt = int.from_bytes(salt, "big") % (2**32)
    rng_salt = np.random.default_rng(seed_salt)
    perm = rng_salt.permutation(1536)
    bits_perm = bits[perm]
    margin_perm = margin[perm]

    # 6. Margin Selection
    idx_sel = np.argpartition(-margin_perm, 832)[:832]
    idx_sel.sort()
    b_sel = bits_perm[idx_sel]
    margin_sel = margin_perm[idx_sel]

    # 7. Sinh khóa ngẫu nhiên
    K = np.random.bytes(20)  # 20 bytes = 160 bits
    K_bits = np.unpackbits(np.frombuffer(K, dtype=np.uint8))

    # 8. LDPC Encode
    encoder = Encode.Proto_LDPC(52, 42, 16)
    C = encoder.encode_LDPC(K_bits.reshape(1, -1)).flatten().astype(np.uint8)

    # 9. Helper data
    helper = b_sel ^ C

    # === VERIFY ===
    # 10. Lặp lại các bước với cùng embedding
    v_proj2 = np.dot(embedding, M_bio.T)
    norm2 = np.linalg.norm(v_proj2)
    if norm2 > 0:
        v_proj2 = v_proj2 / norm2
    bits2, margin2 = binarize_with_perbit_confidence(np.dot(v_proj2, M), intervals)
    bits_perm2 = bits2[perm]
    margin_perm2 = margin2[perm]
    b_sel2 = bits_perm2[idx_sel]
    margin_sel2 = margin_perm2[idx_sel]

    noisy = b_sel2 ^ helper

    # 11. Empirical LLR
    emp_mod = EmpiricalLLR(
        lookup_path=os.path.join(
            _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
        )
    )
    llr = emp_mod.modulate(noisy, context={"margin": margin_sel2})

    # 12. LDPC Decode
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    llr_reshaped = llr.reshape(1, 52, 16)
    output = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr_reshaped}
    )
    K_prime_bits = (output.flatten()[:160] > 0).astype(np.uint8)
    K_prime = np.packbits(K_prime_bits).tobytes()

    print("Original K (hex):", K.hex())
    print("Decoded K (hex):", K_prime.hex())
    print("Match:", K == K_prime)

    handler.sess.close()


if __name__ == "__main__":
    main()
