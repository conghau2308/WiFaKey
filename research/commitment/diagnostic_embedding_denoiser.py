"""
diagnostic_embedding_denoiser.py (FIXED RESTORE – khớp train_embedding_denoiser.py)

So sánh 5 cấu hình:
  1. BPSK gốc (không denoise)
  2. BPSK + Denoised Embedding
  3. Empirical LLR gốc (không denoise)
  4. Empirical LLR + Denoised Embedding
  5. Empirical LLR + Denoised Embedding + Multi‑start (K=5, σ=0.2)
"""

import argparse, csv, hashlib, os, sys, numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR

DIM = 512
FULL_BITS = 1536
N, m, Z = 52, 42, 16


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


def iterate_pairs(pairs_csv, cache_dir, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2, row
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def multi_start_decode(handler, llr_flat, key_hash, K, sigma):
    if _try_decode(handler, llr_flat, key_hash):
        return True
    if K <= 0:
        return False
    llr_clean = llr_flat.reshape(1, N, Z)
    for _ in range(K):
        noise = np.random.normal(0, sigma, size=llr_clean.shape).astype(np.float32)
        llr_noisy = llr_clean + noise
        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr_noisy}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            return True
    return False


def run_diagnostic(handler, pairs_iter, max_pairs, lookup_path, model_path, K, sigma):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)

    # --- Load embedding denoiser (KHÔNG dùng variable_scope để khớp checkpoint cũ) ---
    tf.reset_default_graph()
    denoiser_graph = tf.Graph()
    with denoiser_graph.as_default():
        x_ph = tf.placeholder(tf.float32, [1, DIM], name="noisy_proj")
        h1 = tf.layers.dense(x_ph, 256, activation=tf.nn.relu, name="enc1")
        h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu, name="enc2")
        h3 = tf.layers.dense(h2, 256, activation=tf.nn.relu, name="dec1")
        pred_proj = tf.layers.dense(h3, DIM, activation=None, name="pred_proj")
        saver = tf.train.Saver()

    sess_denoiser = tf.Session(graph=denoiser_graph)
    saver.restore(sess_denoiser, model_path)

    n_total = 0
    pass_bpsk_raw = 0
    pass_bpsk_denoised = 0
    pass_emp_raw = 0
    pass_emp_denoised = 0
    pass_emp_denoised_multi = 0

    for feat_enroll, feat_verify, row in pairs_iter:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # --- Projected vectors ---
        proj_verify = np.dot(feat_verify, handler.M_matrix).astype(np.float32)
        proj_denoised = sess_denoiser.run(
            pred_proj, feed_dict={x_ph: proj_verify.reshape(1, -1)}
        ).flatten()

        # --- 1. BPSK gốc ---
        ok_bpsk_raw = handler.verify(feat_verify, helper, sel_mask, key_hash)

        # --- 2. BPSK + Denoised Embedding ---
        b_full_denoised = handler._binarize_full(proj_denoised).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel_denoised = b_full_denoised[idx]
        noisy_bpsk = np.logical_xor(b_sel_denoised, helper).astype(np.uint8)
        llr_bpsk = (2 * noisy_bpsk.astype(np.float32) - 1).reshape(1, N, Z)
        y_pred_bpsk = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr_bpsk}
        ).flatten()
        decoded_key_bpsk = (y_pred_bpsk > 0).astype(int)[: handler.key_length]
        ok_bpsk_denoised = (
            hashlib.sha256(decoded_key_bpsk.tobytes()).digest() == key_hash
        )

        # --- Margin & noisy bits cho Empirical LLR ---
        bits_raw, margin_raw = binarize_with_perbit_confidence(
            proj_verify, handler.intervals
        )
        margin_sel_raw = margin_raw[idx]
        noisy_raw = np.logical_xor(bits_raw[idx], helper).astype(np.uint8)

        bits_den, margin_den = binarize_with_perbit_confidence(
            proj_denoised, handler.intervals
        )
        margin_sel_den = margin_den[idx]
        noisy_den = np.logical_xor(bits_den[idx], helper).astype(np.uint8)

        # --- 3. Empirical LLR gốc ---
        llr_emp_raw = emp_mod.modulate(
            noisy_raw, context={"margin": margin_sel_raw}
        ).flatten()
        ok_emp_raw = _try_decode(handler, llr_emp_raw, key_hash)

        # --- 4. Empirical LLR + Denoised Embedding ---
        llr_emp_den = emp_mod.modulate(
            noisy_den, context={"margin": margin_sel_den}
        ).flatten()
        ok_emp_denoised = _try_decode(handler, llr_emp_den, key_hash)

        # --- 5. Empirical LLR + Denoised + Multi‑start ---
        ok_emp_denoised_multi = multi_start_decode(
            handler, llr_emp_den, key_hash, K, sigma
        )

        if ok_bpsk_raw:
            pass_bpsk_raw += 1
        if ok_bpsk_denoised:
            pass_bpsk_denoised += 1
        if ok_emp_raw:
            pass_emp_raw += 1
        if ok_emp_denoised:
            pass_emp_denoised += 1
        if ok_emp_denoised_multi:
            pass_emp_denoised_multi += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"1. BPSK gốc                      : {pass_bpsk_raw}/{n_total} ({100*pass_bpsk_raw/n_total:.2f}%)"
    )
    print(
        f"2. BPSK + Denoised Embed         : {pass_bpsk_denoised}/{n_total} ({100*pass_bpsk_denoised/n_total:.2f}%)"
    )
    print(
        f"3. Empirical LLR gốc             : {pass_emp_raw}/{n_total} ({100*pass_emp_raw/n_total:.2f}%)"
    )
    print(
        f"4. Empirical LLR + Denoised      : {pass_emp_denoised}/{n_total} ({100*pass_emp_denoised/n_total:.2f}%)"
    )
    print(
        f"5. Empirical LLR + Denoised + Multi‑start (K={K}): {pass_emp_denoised_multi}/{n_total} ({100*pass_emp_denoised_multi/n_total:.2f}%)"
    )

    sess_denoiser.close()
    handler.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True, help="Path reliability_lookup.npz")
    ap.add_argument("--model", required=True, help="Path embedding denoiser checkpoint")
    ap.add_argument("--K", type=int, default=5, help="Số lần thử multi‑start")
    ap.add_argument("--sigma", type=float, default=0.2, help="Độ lệch chuẩn nhiễu")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "embeddings_cache"
    )
    pairs_csv = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    pair_iter = iterate_pairs(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(
        handler, pair_iter, args.max_pairs, args.lookup, args.model, args.K, args.sigma
    )


if __name__ == "__main__":
    main()
