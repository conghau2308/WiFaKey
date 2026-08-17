"""
diagnostic_margin_erasure_v2.py

Đánh giá Margin‑based Erasure (Hướng K) một cách an toàn, không phá vỡ baseline:
- Dùng SecureWiFaKeyHandler.verify (hash‑based) để có GMR baseline đúng.
- Tạo erasure_mask từ margin của enrollment, rồi truyền cho hàm giải mã
  erasure riêng (có oracle key để so sánh trực tiếp).
- So sánh GMR BPSK (baseline) vs BPSK + Erasure.

Cách chạy:
  python research/commitment/diagnostic_margin_erasure_v2.py --erasure-count 100 --max-pairs 200
"""

import argparse
import csv
import hashlib
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from wifakey_module.wifakey_lib import Modulation


# --- Loaders (giữ nguyên) ---
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


def genuine_pairs_iter(pairs_csv, cache_dir, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


def enroll_with_oracle(handler, feature_vector_float, erasure_count):
    """
    Tạo enrollment giống hệt SecureWiFaKeyHandler.enroll nhưng trả thêm
    random_key và erasure_mask (dựa trên margin toàn cục).
    """
    b_full = handler._binarize_full(feature_vector_float).astype(np.uint8)

    # Chọn ngẫu nhiên feature_length vị trí (giống hệt SecureWiFaKeyHandler)
    rng = np.random.default_rng()
    selection_indices = rng.choice(
        len(b_full), size=handler.feature_length, replace=False
    )
    selection_indices.sort()
    selection_mask = np.zeros(len(b_full), dtype=np.uint8)
    selection_mask[selection_indices] = 1
    b_selected = b_full[selection_indices]

    # Tạo key, codeword, helper_data
    random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
    codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
    helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
    key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

    # Tính erasure_mask từ margin (1536 bit), chọn erasure_count bit margin nhỏ nhất
    _, margin_all = binarize_with_perbit_confidence(
        np.dot(feature_vector_float, handler.M_matrix), handler.intervals
    )
    if erasure_count > 0:
        erasure_pos = np.argpartition(margin_all, erasure_count)[:erasure_count]
        erasure_mask = np.zeros(len(b_full), dtype=bool)
        erasure_mask[erasure_pos] = True
    else:
        erasure_mask = np.zeros(len(b_full), dtype=bool)

    return helper_data, selection_mask, key_hash, random_key.flatten(), erasure_mask


def decode_with_erasure(
    handler, feature_vector_float, helper_data, selection_mask, erasure_mask
):
    """
    Giải mã BPSK cứng, nhưng gán LLR = 0 cho các vị trí có erasure_mask True.
    Trả về key 160-bit.
    """
    b_full = handler._binarize_full(feature_vector_float).astype(np.uint8)
    idx = np.where(selection_mask == 1)[0]
    b_sel = b_full[idx]
    noisy = np.logical_xor(b_sel, helper_data).astype(np.uint8)

    # BPSK cứng
    y_llr = (2 * noisy.astype(np.float32) - 1).reshape(1, handler.N, handler.Z)

    # Erasure: đặt LLR=0 tại các vị trí trong erasure_mask (sau khi đã chọn)
    erasure_selected = erasure_mask[idx].reshape(1, handler.N, handler.Z)
    y_llr = np.where(erasure_selected, 0.0, y_llr)

    y_pred_llr = handler.sess.run(handler.decoder_output, feed_dict={handler.xa: y_llr})
    y_pred_llr = y_pred_llr.flatten()
    decoded_codeword = (y_pred_llr > 0).astype(int)
    return decoded_codeword[: handler.key_length]


def run_diagnostic(handler, pairs_iter, erasure_count, debug=False):
    n_total = 0
    pass_bpsk = 0
    pass_erasure = 0

    for i, (feat_enroll, feat_verify) in enumerate(pairs_iter):
        n_total += 1

        # Enrollment với oracle
        helper, sel_mask, key_hash, true_key, erasure_mask = enroll_with_oracle(
            handler, feat_enroll, erasure_count
        )

        # Baseline: dùng verify gốc của SecureWiFaKeyHandler (trả về bool)
        ok_bpsk = handler.verify(feat_verify, helper, sel_mask, key_hash)

        # Erasure: tự giải mã rồi so sánh với true_key
        rec_key = decode_with_erasure(
            handler, feat_verify, helper, sel_mask, erasure_mask
        )
        ok_erasure = np.array_equal(rec_key, true_key)

        if debug and i < 3:
            print(
                f"Pair {i}: true_key[:10]={true_key[:10]}, baseline={ok_bpsk}, erasure={ok_erasure}"
            )

        if ok_bpsk:
            pass_bpsk += 1
        if ok_erasure:
            pass_erasure += 1

    print(f"Erasure count: {erasure_count} bits")
    print(f"Tổng số cặp genuine: {n_total}")
    print(
        f"GMR BPSK baseline (hash‑verify): {pass_bpsk}/{n_total} ({100*pass_bpsk/n_total:.2f}%)"
    )
    print(
        f"GMR BPSK + Erasure              : {pass_erasure}/{n_total} ({100*pass_erasure/n_total:.2f}%)"
    )
    return pass_bpsk, pass_erasure


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--erasure-count", type=int, default=100)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dataset-folder", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "embeddings_cache"
    )
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    pairs_csv = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")

    print(f"Đọc genuine pairs từ: {pairs_csv}")
    print(f"Cache embedding từ : {cache_dir}")

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(handler, pairs_iter, args.erasure_count, debug=args.debug)


if __name__ == "__main__":
    main()
