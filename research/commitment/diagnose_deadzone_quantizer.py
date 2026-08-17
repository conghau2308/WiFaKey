"""
diagnose_deadzone_quantizer.py

Đánh giá hiệu quả của Dead-zone Quantizer: mở rộng vùng chết quanh threshold 2
để giảm BER. Quét tham số delta (độ dịch ngưỡng) và đo GMR + FAR.
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


def run_diagnostic(
    handler, genuine_iter, impostor_iter, max_pairs, lookup_path, deltas
):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    original_intervals = handler.intervals.copy()

    # Lưu kết quả
    results = []
    # Đánh giá genuine
    genuine_records = []
    for feat_enroll, feat_verify, row in genuine_iter:
        if max_pairs and len(genuine_records) >= max_pairs:
            break
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)
        proj_verify = np.dot(feat_verify, handler.M_matrix)
        bits_raw, margin_raw = binarize_with_perbit_confidence(
            proj_verify, original_intervals
        )
        idx = np.where(sel_mask == 1)[0]
        noisy_raw = np.logical_xor(bits_raw[idx], helper).astype(np.uint8)
        margin_sel_raw = margin_raw[idx]
        llr_raw = emp_mod.modulate(
            noisy_raw, context={"margin": margin_sel_raw}
        ).flatten()
        ok = _try_decode(handler, llr_raw, key_hash)
        genuine_records.append(
            (feat_enroll, feat_verify, helper, sel_mask, key_hash, ok)
        )

    # Đánh giá impostor
    impostor_records = []
    for feat_enroll, feat_verify, row in impostor_iter:
        if max_pairs and len(impostor_records) >= max_pairs:
            break
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)
        impostor_records.append((feat_verify, helper, sel_mask, key_hash))

    for delta in deltas:
        # Tạo intervals mới
        new_intervals = original_intervals.copy()
        new_intervals[1] += delta  # dịch threshold giữa
        # Cập nhật handler (tạm thời)
        handler.intervals = new_intervals

        # Genuine
        gen_pass = 0
        for feat_enroll, feat_verify, helper, sel_mask, key_hash, _ in genuine_records:
            proj_verify = np.dot(feat_verify, handler.M_matrix)
            bits, margin = binarize_with_perbit_confidence(proj_verify, new_intervals)
            idx = np.where(sel_mask == 1)[0]
            noisy = np.logical_xor(bits[idx], helper).astype(np.uint8)
            margin_sel = margin[idx]
            llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
            if _try_decode(handler, llr, key_hash):
                gen_pass += 1

        # Impostor
        imp_pass = 0
        for feat_verify, helper, sel_mask, key_hash in impostor_records:
            proj_verify = np.dot(feat_verify, handler.M_matrix)
            bits, margin = binarize_with_perbit_confidence(proj_verify, new_intervals)
            idx = np.where(sel_mask == 1)[0]
            noisy = np.logical_xor(bits[idx], helper).astype(np.uint8)
            margin_sel = margin[idx]
            llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
            if _try_decode(handler, llr, key_hash):
                imp_pass += 1

        n_gen = len(genuine_records)
        n_imp = len(impostor_records)
        gmr = gen_pass / n_gen * 100 if n_gen else 0
        far = imp_pass / n_imp * 100 if n_imp else 0
        results.append((delta, gen_pass, n_gen, gmr, imp_pass, n_imp, far))
        print(
            f"delta={delta:+.4f}: GMR={gmr:.2f}% ({gen_pass}/{n_gen}), FAR={far:.4f}% ({imp_pass}/{n_imp})"
        )

    # Khôi phục intervals gốc
    handler.intervals = original_intervals
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument(
        "--deltas", type=float, nargs="+", default=[-0.05, -0.02, 0.0, 0.02, 0.05]
    )
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    pairs_dir = os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    cache_dir = os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )
    genuine_csv = os.path.join(pairs_dir, "tune_genuine.csv")
    impostor_csv = os.path.join(pairs_dir, "tune_impostor.csv")

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    genuine_iter = iterate_pairs(genuine_csv, cache_dir, args.max_pairs)
    impostor_iter = iterate_pairs(impostor_csv, cache_dir, args.max_pairs)
    run_diagnostic(
        handler, genuine_iter, impostor_iter, args.max_pairs, args.lookup, args.deltas
    )


if __name__ == "__main__":
    main()
