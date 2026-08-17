"""
diagnose_norm_filter.py

Đánh giá hiệu quả của việc lọc chất lượng mẫu bằng feature‑norm (AdaFace).
Quét qua các ngưỡng norm khác nhau, loại bỏ cặp có norm thấp hơn ngưỡng,
và đo GMR (với Empirical LLR) trên các cặp còn lại.
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


# --- Loaders ---
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


def run_diagnostic(handler, pairs_iter, max_pairs, lookup_path, percentile_thresholds):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)

    # Lưu trữ norm và kết quả cho từng cặp
    records = []  # mỗi phần tử: (norm_enroll, norm_verify, ok_baseline)

    for feat_enroll, feat_verify, row in pairs_iter:
        if max_pairs is not None and len(records) >= max_pairs:
            break

        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # --- Empirical LLR gốc ---
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
        ok = _try_decode(handler, llr_raw, key_hash)

        # Tính norm của embedding
        norm_enroll = np.linalg.norm(feat_enroll)
        norm_verify = np.linalg.norm(feat_verify)

        records.append((norm_enroll, norm_verify, ok))

    if not records:
        print("Không có dữ liệu.")
        return

    # Tính phân phối norm
    norms_enroll = np.array([r[0] for r in records])
    norms_verify = np.array([r[1] for r in records])

    print(
        f"Phân phối norm enroll: min={norms_enroll.min():.4f}, median={np.median(norms_enroll):.4f}, max={norms_enroll.max():.4f}"
    )
    print(
        f"Phân phối norm verify: min={norms_verify.min():.4f}, median={np.median(norms_verify):.4f}, max={norms_verify.max():.4f}"
    )

    # Tính GMR baseline (không lọc)
    total_pairs = len(records)
    baseline_gmr = sum(r[2] for r in records) / total_pairs * 100
    print(
        f"\nBaseline GMR (không lọc): {baseline_gmr:.2f}% ({sum(r[2] for r in records)}/{total_pairs})"
    )

    # Quét qua các ngưỡng percentile
    print(
        "\nQuét qua các ngưỡng norm (loại bỏ cặp có norm_enroll < threshold HOẶC norm_verify < threshold):"
    )
    print(
        f"{'Percentile':>12s} {'Threshold':>10s} {'#Passed':>10s} {'GMR':>10s} {'Từ chối':>10s}"
    )
    print("-" * 55)

    for pct in percentile_thresholds:
        threshold = np.percentile(np.concatenate([norms_enroll, norms_verify]), pct)
        passed = []
        for norm_e, norm_v, ok in records:
            if norm_e >= threshold and norm_v >= threshold:
                passed.append(ok)
        if passed:
            gmr = sum(passed) / len(passed) * 100
        else:
            gmr = 0.0
        rejected = total_pairs - len(passed)
        print(
            f"{pct:12.1f} {threshold:10.4f} {len(passed):10d} {gmr:10.2f}% {rejected:10d} ({rejected/total_pairs*100:.1f}%)"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20],
        help="Danh sách các percentile để quét ngưỡng",
    )
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
    run_diagnostic(handler, pair_iter, args.max_pairs, args.lookup, args.percentiles)


if __name__ == "__main__":
    main()
