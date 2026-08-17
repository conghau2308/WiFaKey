"""
debug_intermediate.py (HMAC-based)
So sánh giá trị trung gian với Rust.
"""

import numpy as np, os, sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(
    0,
    os.path.join(
        _PROJECT_ROOT, "build-wifakey-core", "wifakey-core", "python-reference"
    ),
)

from biohashing import biohash_project
from salted_permutation import generate_permutation as gen_perm_salt
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)


def main():
    embedding = np.full(512, 0.5, dtype=np.float64)
    user_secret = b"test-secret-12345678"
    service_salt = b"test-salt-87654321"

    v_proj = biohash_project(embedding, user_secret)
    print("Python v_proj (first 5):", v_proj[:5])

    M_mat = np.load(
        os.path.join(_PROJECT_ROOT, "wifakey_module", "data", "M_matrix.npy")
    )
    intervals = np.load(
        os.path.join(
            _PROJECT_ROOT, "wifakey_module", "data", "binarization_intervals.npy"
        )
    )
    projected = np.dot(v_proj, M_mat)
    print("Python projected (first 5):", projected[:5])
    bits, margin = binarize_with_perbit_confidence(projected, intervals)
    print("Python margin (first 5):", margin[:5])

    perm = gen_perm_salt(service_salt, 1536)
    margin_perm = margin[perm]
    idx_sel = np.argpartition(-margin_perm, 832)[:832]
    idx_sel.sort()
    margin_sel = margin_perm[idx_sel]
    print("Python margin_sel (first 5):", margin_sel[:5])


if __name__ == "__main__":
    main()
