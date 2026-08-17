"""
diagnose_tta_simulation.py

Mô phỏng Test‑Time Augmentation (TTA) trên embedding:
- Tạo M phiên bản embedding bằng cách thêm nhiễu Gaussian nhẹ.
- Lấy trung bình để có embedding ổn định hơn trước khi binarize.
- So sánh GMR với Empirical LLR gốc (không TTA).
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
    llr = llr_flat.reshape(1, handler.N, handler.Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def run_diagnostic(handler, pairs_iter, max_pairs, lookup_path, M, sigma_tta):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)

    n_total = 0
    pass_baseline = 0
    pass_tta = 0

    for feat_enroll, feat_verify, row in pairs_iter:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # --- Baseline: Empirical LLR gốc ---
        proj_verify = np.dot(feat_verify, handler.M_matrix)
        bits_raw, margin_raw = binarize_with_perbit_confidence(
            proj_verify, handler.intervals
        )
        idx = np.where(sel_mask == 1)[0]
        noisy_raw = np.logical_xor(bits_raw[idx], helper).astype(np.uint8)
        margin_sel_raw = margin_raw[idx]
        llr_raw = emp_mod.modulate(
            noisy_raw, context={"margin": margin_sel_raw}
        ).flatten()
        ok_baseline = _try_decode(handler, llr_raw, key_hash)

        # --- TTA simulation ---
        # Tạo M bản sao embedding bằng cách thêm nhiễu nhẹ
        emb_verify_smoothed = np.zeros_like(feat_verify)
        for _ in range(M):
            noise = np.random.normal(0, sigma_tta, size=feat_verify.shape).astype(
                np.float32
            )
            emb_noisy = feat_verify + noise
            emb_verify_smoothed += emb_noisy
        emb_verify_smoothed /= M

        # Binarize embedding đã làm mượt
        proj_tta = np.dot(emb_verify_smoothed, handler.M_matrix)
        bits_tta, margin_tta = binarize_with_perbit_confidence(
            proj_tta, handler.intervals
        )
        noisy_tta = np.logical_xor(bits_tta[idx], helper).astype(np.uint8)
        margin_sel_tta = margin_tta[idx]
        llr_tta = emp_mod.modulate(
            noisy_tta, context={"margin": margin_sel_tta}
        ).flatten()
        ok_tta = _try_decode(handler, llr_tta, key_hash)

        if ok_baseline:
            pass_baseline += 1
        if ok_tta:
            pass_tta += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"Empirical LLR gốc (baseline) : {pass_baseline}/{n_total} ({100*pass_baseline/n_total:.2f}%)"
    )
    print(
        f"Empirical LLR + TTA (M={M}, σ={sigma_tta}) : {pass_tta}/{n_total} ({100*pass_tta/n_total:.2f}%)"
    )

    handler.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--M", type=int, default=10, help="Số lần cộng nhiễu (số mẫu TTA)")
    ap.add_argument(
        "--sigma-tta", type=float, default=0.01, help="Độ lệch chuẩn nhiễu Gaussian"
    )
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
        handler, pair_iter, args.max_pairs, args.lookup, args.M, args.sigma_tta
    )


if __name__ == "__main__":
    main()
