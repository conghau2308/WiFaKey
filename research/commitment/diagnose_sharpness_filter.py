"""
diagnose_sharpness_filter.py

Đánh giá hiệu quả của việc lọc ảnh chất lượng kém bằng độ sắc nét (Variance of Laplacian).
Ảnh có sharpness thấp (mờ, nhòe) sẽ bị loại, chỉ giữ lại các cặp có cả enroll và verify đủ nét.
Đo GMR với Empirical LLR trên các cặp còn lại, quét qua các ngưỡng percentile của sharpness.
"""

import argparse, csv, hashlib, os, sys, cv2, numpy as np
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
RAW_IMG_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "labeled_faces_in_the_wild", "lfw-deepfunneled"
)


def load_embedding(cache_dir, name, imagenum):
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def compute_sharpness(image_path):
    """Tính Variance of Laplacian cho ảnh xám."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap.var()


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
    records = []

    for feat_enroll, feat_verify, row in pairs_iter:
        if max_pairs is not None and len(records) >= max_pairs:
            break

        helper, sel_mask, key_hash = handler.enroll(feat_enroll)

        # Empirical LLR
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

        # Tính sharpness cho ảnh enroll và verify từ ảnh gốc
        enroll_path = os.path.join(
            RAW_IMG_DIR,
            row["name_enroll"],
            f"{row['name_enroll']}_{int(row['imagenum_enroll']):04d}.jpg",
        )
        verify_path = os.path.join(
            RAW_IMG_DIR,
            row["name_verify"],
            f"{row['name_verify']}_{int(row['imagenum_verify']):04d}.jpg",
        )
        sharp_e = compute_sharpness(enroll_path)
        sharp_v = compute_sharpness(verify_path)

        if sharp_e is not None and sharp_v is not None:
            records.append((sharp_e, sharp_v, ok))
        else:
            # Nếu không đọc được ảnh, vẫn giữ lại nhưng không lọc
            records.append((float("inf"), float("inf"), ok))

    if not records:
        print("Không có dữ liệu.")
        return

    # Thống kê sharpness
    sharps_enroll = np.array([r[0] for r in records])
    sharps_verify = np.array([r[1] for r in records])
    all_sharps = np.concatenate([sharps_enroll, sharps_verify])
    finite_sharps = all_sharps[np.isfinite(all_sharps)]

    print(
        f"Phân phối sharpness: min={np.min(finite_sharps):.2f}, median={np.median(finite_sharps):.2f}, max={np.max(finite_sharps):.2f}"
    )

    total_pairs = len(records)
    baseline_gmr = sum(r[2] for r in records) / total_pairs * 100
    print(
        f"\nBaseline GMR (không lọc): {baseline_gmr:.2f}% ({sum(r[2] for r in records)}/{total_pairs})"
    )

    print(
        "\nQuét qua các ngưỡng sharpness (loại bỏ cặp có sharpness_enroll < threshold HOẶC sharpness_verify < threshold):"
    )
    print(
        f"{'Percentile':>12s} {'Threshold':>10s} {'#Passed':>10s} {'GMR':>10s} {'Từ chối':>10s}"
    )
    print("-" * 55)

    for pct in percentile_thresholds:
        if len(finite_sharps) == 0:
            threshold = 0
        else:
            threshold = np.percentile(finite_sharps, pct)
        passed = []
        for sharp_e, sharp_v, ok in records:
            if sharp_e >= threshold and sharp_v >= threshold:
                passed.append(ok)
        if passed:
            gmr = sum(passed) / len(passed) * 100
        else:
            gmr = 0.0
        rejected = total_pairs - len(passed)
        print(
            f"{pct:12.1f} {threshold:10.2f} {len(passed):10d} {gmr:10.2f}% {rejected:10d} ({rejected/total_pairs*100:.1f}%)"
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
