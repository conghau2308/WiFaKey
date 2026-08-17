"""
debug_pipeline_compare.py

In margin và LLR của 5 bit đầu tiên từ pipeline Python, để so sánh với Rust.
"""

import numpy as np
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR


def main():
    # Embedding, secret, salt giống hệt Rust
    embedding = np.full(512, 0.5, dtype=np.float64)
    user_secret = b"test-secret-12345678"
    service_salt = b"test-salt-87654321"

    # BioHashing
    seed = int.from_bytes(user_secret, "big") % (2**32)
    rng = np.random.default_rng(seed)
    M = rng.normal(0, 1, size=(512, 512)).astype(np.float32)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    v_proj = np.dot(embedding, M.T)
    norm = np.linalg.norm(v_proj)
    if norm > 0:
        v_proj = v_proj / norm

    # Load M_matrix và intervals
    M_mat = np.load(
        os.path.join(_PROJECT_ROOT, "wifakey_module", "data", "M_matrix.npy")
    )
    intervals = np.load(
        os.path.join(
            _PROJECT_ROOT, "wifakey_module", "data", "binarization_intervals.npy"
        )
    )

    # Binarization
    projected = np.dot(v_proj, M_mat)
    bits, margin = binarize_with_perbit_confidence(projected, intervals)

    # Salted Permutation
    seed_salt = int.from_bytes(service_salt, "big") % (2**32)
    rng_salt = np.random.default_rng(seed_salt)
    perm = rng_salt.permutation(1536)
    bits_perm = bits[perm]
    margin_perm = margin[perm]

    # Margin Selection (dùng cách chọn top 832 để tạo mask giống Rust)
    idx_sel = np.argpartition(-margin_perm, 832)[:832]
    idx_sel.sort()
    b_sel = bits_perm[idx_sel]
    margin_sel = margin_perm[idx_sel]

    # In margin
    print("Python margin (first 5):", margin_sel[:5])

    # Empirical LLR (dùng noisy bit = 0 để kiểm tra magnitude)
    emp_mod = EmpiricalLLR(
        lookup_path=os.path.join(
            _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
        )
    )
    print("Python LLR (noisy=0, first 5):")
    for m in margin_sel[:5]:
        llr_val = emp_mod.modulate(
            np.array([0], dtype=np.uint8), context={"margin": np.array([m])}
        )[0]
        print(f"  margin={m:.10f} -> LLR={llr_val:.6f}")


if __name__ == "__main__":
    main()
