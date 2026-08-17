"""
export_test_vectors.py (HMAC-based)
Xuất test vectors cho Rust.
"""

import numpy as np, os, sys, json, hashlib

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
from wifakey_module.wifakey_lib import Encode


def main():
    embedding = np.full(512, 0.5, dtype=np.float64)
    user_secret = b"test-secret-12345678"
    service_salt = b"test-salt-87654321"

    v_proj = biohash_project(embedding, user_secret)
    M_mat = np.load(
        os.path.join(_PROJECT_ROOT, "wifakey_module", "data", "M_matrix.npy")
    )
    intervals = np.load(
        os.path.join(
            _PROJECT_ROOT, "wifakey_module", "data", "binarization_intervals.npy"
        )
    )
    projected = np.dot(v_proj, M_mat)
    bits, margin = binarize_with_perbit_confidence(projected, intervals)
    perm = gen_perm_salt(service_salt, 1536)
    bits_perm, margin_perm = bits[perm], margin[perm]
    idx_sel = np.argpartition(-margin_perm, 832)[:832]
    idx_sel.sort()
    b_sel, margin_sel = bits_perm[idx_sel], margin_perm[idx_sel]

    K = np.random.bytes(20)
    K_bits = np.unpackbits(np.frombuffer(K, dtype=np.uint8))
    encoder = Encode.Proto_LDPC(52, 42, 16)
    C = encoder.encode_LDPC(K_bits.reshape(1, -1)).flatten().astype(np.uint8)
    helper = b_sel ^ C
    key_hash = hashlib.sha256(K).digest()

    out = {
        "v_proj": v_proj.tolist(),
        "perm": perm.tolist(),
        "b_sel": b_sel.tolist(),
        "margin_sel": margin_sel.tolist(),
        "helper_data": helper.tolist(),
        "key_hash": key_hash.hex(),
        "original_k": K.hex(),
    }
    with open(os.path.join(os.path.dirname(__file__), "test_vectors.json"), "w") as f:
        json.dump(out, f)
    print("Đã lưu test_vectors.json (HMAC-based)")


if __name__ == "__main__":
    main()
