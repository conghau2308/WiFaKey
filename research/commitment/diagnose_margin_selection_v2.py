"""
diagnose_margin_selection_v2.py (HỖ TRỢ --mode)

Đánh giá Reliability‑mask (chọn bit theo margin) kết hợp với Empirical LLR.
Hỗ trợ chạy riêng lẻ từng chế độ để tránh OOM:
  --mode baseline : chỉ chạy baseline (selection ngẫu nhiên)
  --mode margin   : chỉ chạy margin selection
"""

import argparse, csv, hashlib, os, sys, json, numpy as np
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
        except:
            continue


class MarginSelectionHandler(SecureWiFaKeyHandler):
    def enroll(self, feature_vector_float):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        projected = np.dot(feature_vector_float, self.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, self.intervals)
        selection_indices = np.argpartition(-margin, self.feature_length)[
            : self.feature_length
        ]
        selection_indices.sort()
        selection_mask = np.zeros(FULL_BITS, dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected = b_full[selection_indices]
        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()
        return helper_data, selection_mask, key_hash


def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def run_baseline(handler, pairs_iter, max_pairs, lookup_path):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    results = []
    for feat_enroll, feat_verify, row in pairs_iter:
        if max_pairs and len(results) >= max_pairs:
            break
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)
        b_full_v = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel = b_full_v[idx]
        noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
        _, margin_v = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_v[idx]
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        ok = _try_decode(handler, llr, key_hash)
        results.append(
            {
                "pair_id": f"{row['name_enroll']}_{row['imagenum_enroll']}_{row['name_verify']}_{row['imagenum_verify']}",
                "success": bool(ok),
            }
        )
    return results


def run_margin(handler, pairs_iter, max_pairs, lookup_path):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    results = []
    for feat_enroll, feat_verify, row in pairs_iter:
        if max_pairs and len(results) >= max_pairs:
            break
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)
        b_full_v = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel = b_full_v[idx]
        noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
        _, margin_v = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_v[idx]
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        ok = _try_decode(handler, llr, key_hash)
        results.append(
            {
                "pair_id": f"{row['name_enroll']}_{row['imagenum_enroll']}_{row['name_verify']}_{row['imagenum_verify']}",
                "success": bool(ok),
            }
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline", "margin"])
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--output", required=True, help="File JSON để lưu kết quả")
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

    if args.mode == "baseline":
        handler = SecureWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
    else:
        handler = MarginSelectionHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )

    pairs_iter = iterate_pairs(pairs_csv, cache_dir, args.max_pairs)

    if args.mode == "baseline":
        results = run_baseline(handler, pairs_iter, args.max_pairs, args.lookup)
    else:
        results = run_margin(handler, pairs_iter, args.max_pairs, args.lookup)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    n_total = len(results)
    n_success = sum(r["success"] for r in results)
    print(f"GMR {args.mode}: {n_success}/{n_total} ({100*n_success/n_total:.2f}%)")
    handler.sess.close()


if __name__ == "__main__":
    main()
