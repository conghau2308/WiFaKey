"""
diagnose_biohashing_v2.py

So sánh BioHashing với pipeline TỐT NHẤT (Margin Selection + Empirical LLR).
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


def transform_embedding(emb, proj_matrix):
    vec = np.dot(emb, proj_matrix.T)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def margin_enroll(handler, emb, proj_matrix=None):
    """Enrollment với margin selection, trả về (helper, sel_mask, key_hash)."""
    if proj_matrix is not None:
        emb = transform_embedding(emb, proj_matrix)
    b_full = handler._binarize_full(emb).astype(np.uint8)
    projected = np.dot(emb, handler.M_matrix)
    _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
    # Margin selection: chọn 832 bit có margin cao nhất
    selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
    selection_indices.sort()
    selection_mask = np.zeros(FULL_BITS, dtype=np.uint8)
    selection_mask[selection_indices] = 1
    b_selected = b_full[selection_indices]
    random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
    codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
    helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
    key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()
    return helper_data, selection_mask, key_hash


def margin_verify(
    handler, emb, helper_data, selection_mask, key_hash, emp_mod, proj_matrix=None
):
    """Verification với margin selection + empirical LLR."""
    if proj_matrix is not None:
        emb = transform_embedding(emb, proj_matrix)
    b_full = handler._binarize_full(emb).astype(np.uint8)
    idx = np.where(selection_mask == 1)[0]
    b_sel = b_full[idx]
    noisy = np.logical_xor(b_sel, helper_data).astype(np.uint8)
    projected = np.dot(emb, handler.M_matrix)
    _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
    margin_sel = margin[idx]
    llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
    llr = llr.reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def run_test(handler, test_pairs, emp_mod, proj_matrix=None):
    pass_count = 0
    for name_e, img_e, name_v, img_v in test_pairs:
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)
        helper, sel_mask, key_hash = margin_enroll(handler, emb_e, proj_matrix)
        ok = margin_verify(
            handler, emb_v, helper, sel_mask, key_hash, emp_mod, proj_matrix
        )
        if ok:
            pass_count += 1
    return pass_count


def main():
    # Tạo ma trận BioHashing
    rng = np.random.default_rng(12345)
    user_secret = rng.bytes(32)
    seed = int.from_bytes(user_secret, "big") % (2**32)
    proj_rng = np.random.default_rng(seed)
    M = proj_rng.normal(0, 1, size=(N_DIMS, N_DIMS)).astype(np.float32)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)

    test_pairs = load_genuine_pairs(max_pairs=200)

    # Baseline: Margin Selection + Empirical LLR (không BioHashing)
    print("=== BASELINE (Margin + Empirical LLR) ===")
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    emp_mod = EmpiricalLLR(
        lookup_path=os.path.join(
            _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
        )
    )
    pass_baseline = run_test(handler, test_pairs, emp_mod, proj_matrix=None)
    gmr_baseline = 100 * pass_baseline / len(test_pairs)
    print(f"Baseline GMR: {pass_baseline}/{len(test_pairs)} ({gmr_baseline:.2f}%)")
    handler.sess.close()

    # BioHashing: Margin Selection + Empirical LLR + BioHashing
    print("\n=== BIOHASHING (Margin + Empirical + BioHashing) ===")
    handler2 = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    pass_biohash = run_test(handler2, test_pairs, emp_mod, proj_matrix=M)
    gmr_biohash = 100 * pass_biohash / len(test_pairs)
    print(f"BioHashing GMR: {pass_biohash}/{len(test_pairs)} ({gmr_biohash:.2f}%)")
    handler2.sess.close()


if __name__ == "__main__":
    main()
