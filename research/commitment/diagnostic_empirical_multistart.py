"""
diagnostic_empirical_multistart.py

Kết hợp Empirical LLR (v2_empirical_llr) và Multi‑start Decoding (hướng J).
So sánh GMR giữa:
    - baseline: BPSK cứng, 1 lần giải mã
    - empirical: LLR hiệu chỉnh, 1 lần
    - empirical_multi: LLR hiệu chỉnh + K lần khởi động lại (có nhiễu)

Không sửa handler gốc, chỉ kế thừa SecureWiFaKeyHandler.
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
from research.modulation.v2_empirical_llr import EmpiricalLLR
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


# --- Handler tích hợp ---
class EmpiricalMultiStartHandler(SecureWiFaKeyHandler):
    def __init__(self, empirical_llr=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.empirical_llr = empirical_llr or EmpiricalLLR()

    def enroll(self, feature_vector_float):
        # Giống SecureWiFaKeyHandler.enroll nhưng trả thêm random_key
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full), size=self.feature_length, replace=False
        )
        selection_indices.sort()
        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected = b_full[selection_indices]

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()
        return helper_data, selection_mask, key_hash, random_key.flatten()

    def verify_bpsk(self, feature_vector_float, helper_data, selection_mask):
        """BPSK cứng, 1 lần"""
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper_data)
        llr = Modulation.BPSK(noisy).astype(np.float32).reshape(1, self.N, self.Z)
        out = self.sess.run(self.decoder_output, feed_dict={self.xa: llr}).flatten()
        return (out > 0).astype(int)[: self.key_length]

    def verify_empirical(self, feature_vector_float, helper_data, selection_mask):
        """Empirical LLR, 1 lần"""
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper_data).astype(np.uint8)

        # Lấy margin per-bit cho toàn bộ 1536 bit, rồi chọn theo mask
        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feature_vector_float, self.M_matrix), self.intervals
        )
        margin_sel = margin_all[idx]

        llr = self.empirical_llr.modulate(noisy, context={"margin": margin_sel})
        llr = llr.reshape(1, self.N, self.Z)
        out = self.sess.run(self.decoder_output, feed_dict={self.xa: llr}).flatten()
        return (out > 0).astype(int)[: self.key_length]

    def verify_empirical_multi(
        self,
        feature_vector_float,
        helper_data,
        selection_mask,
        key_hash,
        K=5,
        sigma=0.2,
    ):
        """Empirical LLR + K lần thử, trả về True nếu có lần khớp hash"""
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper_data).astype(np.uint8)

        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feature_vector_float, self.M_matrix), self.intervals
        )
        margin_sel = margin_all[idx]

        llr_clean = self.empirical_llr.modulate(
            noisy, context={"margin": margin_sel}
        ).reshape(1, self.N, self.Z)

        for _ in range(K):
            noise = np.random.normal(0, sigma, size=llr_clean.shape).astype(np.float32)
            llr_noisy = llr_clean + noise
            out = self.sess.run(
                self.decoder_output, feed_dict={self.xa: llr_noisy}
            ).flatten()
            key = (out > 0).astype(int)[: self.key_length]
            if hashlib.sha256(key.tobytes()).digest() == key_hash:
                return True
        return False


def run_diagnostic(handler, pairs_iter, K=5, sigma=0.2):
    n_total = 0
    pass_bpsk = 0
    pass_emp = 0
    pass_emp_multi = 0
    rescued_by_multi = 0  # empirical fail nhưng multi pass

    for feat_enroll, feat_verify in pairs_iter:
        n_total += 1
        helper, mask, key_hash, true_key = handler.enroll(feat_enroll)

        # BPSK baseline
        key_bpsk = handler.verify_bpsk(feat_verify, helper, mask)
        ok_bpsk = np.array_equal(key_bpsk, true_key)

        # Empirical single
        key_emp = handler.verify_empirical(feat_verify, helper, mask)
        ok_emp = np.array_equal(key_emp, true_key)

        # Empirical multi (chỉ chạy nếu single fail để tiết kiệm)
        ok_emp_multi = ok_emp
        if not ok_emp:
            ok_emp_multi = handler.verify_empirical_multi(
                feat_verify, helper, mask, key_hash, K=K, sigma=sigma
            )
            if ok_emp_multi:
                rescued_by_multi += 1

        if ok_bpsk:
            pass_bpsk += 1
        if ok_emp:
            pass_emp += 1
        if ok_emp_multi:
            pass_emp_multi += 1

    print(f"K={K}, sigma={sigma}")
    print(f"Tổng số cặp genuine: {n_total}")
    print(
        f"GMR BPSK (baseline)      : {pass_bpsk}/{n_total} ({100*pass_bpsk/n_total:.2f}%)"
    )
    print(
        f"GMR Empirical (single)   : {pass_emp}/{n_total} ({100*pass_emp/n_total:.2f}%)"
    )
    print(
        f"GMR Empirical + multi    : {pass_emp_multi}/{n_total} ({100*pass_emp_multi/n_total:.2f}%)"
    )
    print(f"Ca được multi cứu thêm   : {rescued_by_multi}")
    return locals()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.2)
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

    handler = EmpiricalMultiStartHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(handler, pairs_iter, K=args.K, sigma=args.sigma)


if __name__ == "__main__":
    main()
