"""
run_final_benchmark_extended.py (CẬP NHẬT: hỗ trợ margin‑selection)

Benchmark mở rộng: hỗ trợ Margin Selection Handler.

Cách dùng:
  # Baseline (random selection)
  python run_final_benchmark_extended.py --lookup <path> --output results/baseline.json

  # Margin selection
  python run_final_benchmark_extended.py --lookup <path> --margin-selection --output results/margin.json

  # Margin selection + Multi‑start
  python run_final_benchmark_extended.py --lookup <path> --margin-selection --K 5 --sigma 0.2 --output results/margin_multi.json
"""

import argparse, csv, hashlib, json, os, sys, numpy as np
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


# ----------------------------------------------------------------------
# Margin Selection Handler
# ----------------------------------------------------------------------
class MarginSelectionHandler(SecureWiFaKeyHandler):
    """Chọn 832 bit có margin lớn nhất từ ảnh enrollment."""

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


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
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


def pairs_iter(pairs_csv, cache_dir, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2, row
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


# ----------------------------------------------------------------------
# Multi‑start decode
# ----------------------------------------------------------------------
def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, handler.N, handler.Z)
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
    llr_clean = llr_flat.reshape(1, handler.N, handler.Z)
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


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
def run_benchmark(handler, pairs_csv, cache_dir, max_pairs, lookup_path, K, sigma):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    results = {"genuine": [], "impostor": []}
    genuine_success, genuine_total = 0, 0
    impostor_success, impostor_total = 0, 0

    # Genuine
    genuine_csv = (
        pairs_csv.replace("impostor", "genuine")
        if "impostor" in pairs_csv
        else pairs_csv
    )
    for feat_enroll, feat_verify, row in pairs_iter(genuine_csv, cache_dir, max_pairs):
        genuine_total += 1
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
        ok = multi_start_decode(handler, llr, key_hash, K, sigma)
        if ok:
            genuine_success += 1
        pair_id = f"{row['name_enroll']}_{row['imagenum_enroll']}_{row['name_verify']}_{row['imagenum_verify']}"
        results["genuine"].append({"pair_id": pair_id, "success": bool(ok)})

    # Impostor
    impostor_csv = (
        pairs_csv.replace("genuine", "impostor")
        if "genuine" in pairs_csv
        else pairs_csv
    )
    if os.path.exists(impostor_csv):
        for feat_enroll, feat_verify, row in pairs_iter(
            impostor_csv, cache_dir, max_pairs
        ):
            impostor_total += 1
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
            ok = multi_start_decode(handler, llr, key_hash, K, sigma)
            if ok:
                impostor_success += 1
            pair_id = f"{row['name_enroll']}_{row['imagenum_enroll']}_{row['name_verify']}_{row['imagenum_verify']}"
            results["impostor"].append({"pair_id": pair_id, "success": bool(ok)})

    return genuine_success, genuine_total, impostor_success, impostor_total, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--margin-selection", action="store_true")
    ap.add_argument("--K", type=int, default=0, help="Multi‑start K (0 = tắt)")
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--output", required=True)
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
    genuine_csv = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")

    label = "margin_selection" if args.margin_selection else "baseline_random"
    if args.K > 0:
        label += f"_multistart_K{args.K}_s{args.sigma}"

    if args.margin_selection:
        handler = MarginSelectionHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
    else:
        handler = SecureWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )

    gen_succ, gen_total, imp_succ, imp_total, results = run_benchmark(
        handler, genuine_csv, cache_dir, args.max_pairs, args.lookup, args.K, args.sigma
    )

    output_data = {
        "label": label,
        "tier": args.tier,
        "genuine_success": gen_succ,
        "genuine_total": gen_total,
        "impostor_success": imp_succ,
        "impostor_total": imp_total,
        "FRR": (gen_total - gen_succ) / gen_total if gen_total else 0.0,
        "FAR": imp_succ / imp_total if imp_total else 0.0,
        "results": {},
    }
    for entry in results["genuine"]:
        output_data["results"][entry["pair_id"]] = {
            "is_genuine": True,
            "success": entry["success"],
        }
    for entry in results["impostor"]:
        output_data["results"][entry["pair_id"]] = {
            "is_genuine": False,
            "success": entry["success"],
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"GMR: {gen_succ}/{gen_total} ({100*gen_succ/gen_total:.2f}%)")
    print(f"FAR: {imp_succ}/{imp_total} ({100*imp_succ/imp_total:.4f}%)")
    handler.sess.close()


if __name__ == "__main__":
    main()
