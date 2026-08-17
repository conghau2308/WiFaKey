"""
diagnose_droba.py (HOÀN CHỈNH)

Đánh giá GMR của DROBA: phân bổ bit theo Fisher Ratio.
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


def load_pairs(pairs_csv, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def iterate_pairs(pairs_csv, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(row["name_enroll"], int(row["imagenum_enroll"]))
            e2 = load_embedding(row["name_verify"], int(row["imagenum_verify"]))
            yield e1, e2, row
        except:
            continue


def compute_fisher_ratios(pairs_csv, max_pairs=400):
    intra_vars = np.zeros(N_DIMS)
    count = 0
    for e1, e2, _ in iterate_pairs(pairs_csv, max_pairs):
        intra_vars += (e1 - e2) ** 2
        count += 1
    intra_vars /= max(count, 1)
    all_emb = []
    for e1, e2, _ in iterate_pairs(pairs_csv, max_pairs):
        all_emb.append(e1)
        all_emb.append(e2)
    all_emb = np.array(all_emb)
    inter_vars = np.var(all_emb, axis=0)
    fisher = inter_vars / (intra_vars + 1e-8)
    return fisher


def create_bit_mask(fisher, total_bits=832):
    sorted_idx = np.argsort(-fisher)
    # Giải: 3*X + 2*Y + 1*Z = total_bits, X+Y+Z=512
    # => Y = total_bits - 512 - 2*X, Z = 512 - X - Y
    X, Y, Z = 0, 0, 512
    for x in range(0, 513):
        y = total_bits - 512 - 2 * x
        if y < 0 or y > 512 - x:
            continue
        z = 512 - x - y
        if z >= 0:
            X, Y, Z = x, y, z
            break
    # Map bit allocation
    bit_alloc = np.ones(N_DIMS, dtype=int)  # mặc định 1 bit
    for i in range(X):
        bit_alloc[sorted_idx[i]] = 3
    for i in range(X, X + Y):
        bit_alloc[sorted_idx[i]] = 2
    # Tạo mask 1536
    mask = np.zeros(FULL_BITS, dtype=np.uint8)
    for d in range(N_DIMS):
        n = bit_alloc[d]
        if n == 3:
            mask[d * 3 : d * 3 + 3] = 1
        elif n == 2:
            mask[d * 3] = 1
            mask[d * 3 + 2] = 1
        else:  # 1
            mask[d * 3] = 1
    return mask, bit_alloc


def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def run_diagnostic(handler, pairs_csv, max_pairs, lookup_path, bit_mask):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    kept_indices = np.where(bit_mask == 1)[0]
    n_kept = len(kept_indices)
    print(f"Số bit được giữ: {n_kept}")

    n_total, pass_baseline, pass_droba = 0, 0, 0
    for emb_enroll, emb_verify, row in iterate_pairs(pairs_csv, max_pairs):
        if max_pairs and n_total >= max_pairs:
            break
        n_total += 1

        # --- Enrollment với mask DROBA ---
        b_full_e = handler._binarize_full(emb_enroll).astype(np.uint8)
        # Dùng kept_indices thay vì random selection
        b_selected_e = b_full_e[kept_indices]
        # Tạo key và helper_data
        rng = np.random.default_rng()
        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected_e, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        # --- Verify với mask DROBA ---
        b_full_v = handler._binarize_full(emb_verify).astype(np.uint8)
        b_selected_v = b_full_v[kept_indices]
        noisy = np.logical_xor(b_selected_v, helper_data).astype(np.uint8)

        _, margin_all = binarize_with_perbit_confidence(
            np.dot(emb_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_all[kept_indices]

        # Empirical LLR và decode
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        ok = _try_decode(handler, llr, key_hash)
        if ok:
            pass_droba += 1

        # Baseline (giữ nguyên code cũ, chọn ngẫu nhiên 832 bit)
        helper_base, sel_mask_base, key_hash_base = handler.enroll(emb_enroll)
        ok_base = handler.verify(emb_verify, helper_base, sel_mask_base, key_hash_base)
        if ok_base:
            pass_baseline += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"GMR Baseline (random 832 bit): {pass_baseline}/{n_total} ({100*pass_baseline/n_total:.2f}%)"
    )
    print(
        f"GMR DROBA   (Fisher-based)   : {pass_droba}/{n_total} ({100*pass_droba/n_total:.2f}%)"
    )
    handler.sess.close()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = _PROJECT_ROOT
    data_dir = os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    pairs_dir = os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    pairs_csv = os.path.join(pairs_dir, "tune_genuine.csv")

    # Tính Fisher
    fisher = compute_fisher_ratios(pairs_csv, max_pairs=400)
    # Tạo mask
    mask, bit_alloc = create_bit_mask(fisher, total_bits=832)
    print("Phân bổ bit:", np.unique(bit_alloc, return_counts=True))

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )
    run_diagnostic(handler, pairs_csv, args.max_pairs, args.lookup, mask)


if __name__ == "__main__":
    main()
