"""
diagnostic_multistart_decoding.py

Đánh giá hướng J: giải mã lặp lại với khởi tạo ngẫu nhiên (multi‑start decoding).
Khi verify, thay vì chạy Neural‑MS decoder đúng 1 lần, ta chạy K lần độc lập,
mỗi lần thêm nhiễu Gaussian nhỏ (sigma) vào LLR đầu vào. Chỉ cần một lần khớp
hash là verify thành công. So sánh GMR với baseline 1 lần.

Không sửa handler gốc – kế thừa SecureWiFaKeyHandler, thêm oracle key.
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
from wifakey_module.wifakey_lib import Modulation


# --- Loaders (giữ nguyên convention) ---
def load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def load_pairs(pairs_csv: str, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def genuine_pairs_iter(pairs_csv: str, cache_dir: str, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


class MultiStartDiagnosticHandler(SecureWiFaKeyHandler):
    """
    Handler chẩn đoán: enroll trả về random_key (oracle).
    verify_multi_start() thực hiện K lần giải mã, mỗi lần thêm nhiễu Gaussian
    vào LLR đầu vào, hash‑check đến khi thành công hoặc hết K lần.
    """

    def enroll(self, feature_vector_float: np.ndarray):
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

    def verify_single(self, feature_vector_float, helper_data, selection_mask):
        """Giải mã 1 lần, trả về reconstructed_key (160 bit) và thành công hay không."""
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        selection_indices = np.where(selection_mask == 1)[0]
        b_selected = b_full[selection_indices]
        y_noisy_bits = np.logical_xor(b_selected, helper_data)

        y_llr = (
            Modulation.BPSK(y_noisy_bits)
            .astype(np.float32)
            .reshape((1, self.N, self.Z))
        )
        y_pred_llr = self.sess.run(self.decoder_output, feed_dict={self.xa: y_llr})
        y_pred_llr = y_pred_llr.flatten()
        decoded_codeword = (y_pred_llr > 0).astype(int)
        return decoded_codeword[: self.key_length]

    def verify_multi_start(
        self,
        feature_vector_float,
        helper_data,
        selection_mask,
        key_hash,
        K=5,
        sigma=0.1,
    ):
        """
        Thử giải mã tối đa K lần. Mỗi lần thêm nhiễu Gaussian với std=sigma
        vào LLR đầu vào. Trả về True nếu ít nhất một lần khớp hash.
        """
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        selection_indices = np.where(selection_mask == 1)[0]
        b_selected = b_full[selection_indices]
        y_noisy_bits = np.logical_xor(b_selected, helper_data)

        # LLR gốc
        y_llr_clean = (
            Modulation.BPSK(y_noisy_bits)
            .astype(np.float32)
            .reshape((1, self.N, self.Z))
        )

        for _ in range(K):
            # Thêm nhiễu Gaussian nhỏ
            noise = np.random.normal(0, sigma, size=y_llr_clean.shape).astype(
                np.float32
            )
            y_llr = y_llr_clean + noise

            y_pred_llr = self.sess.run(self.decoder_output, feed_dict={self.xa: y_llr})
            y_pred_llr = y_pred_llr.flatten()
            decoded_codeword = (y_pred_llr > 0).astype(int)
            reconstructed_key = decoded_codeword[: self.key_length]

            # Hash check
            recon_hash = hashlib.sha256(reconstructed_key.tobytes()).digest()
            if recon_hash == key_hash:
                return True

        return False


def run_diagnostic(handler, genuine_pairs_iter, K=5, sigma=0.1):
    n_total = 0
    n_pass_single = 0
    n_pass_multi = 0
    n_rescued = 0  # fail single nhưng pass multi

    for feature_enroll, feature_verify in genuine_pairs_iter:
        n_total += 1

        helper_data, selection_mask, key_hash, random_key = handler.enroll(
            feature_enroll
        )

        # Single
        rec_key_single = handler.verify_single(
            feature_verify, helper_data, selection_mask
        )
        pass_single = np.array_equal(rec_key_single, random_key)

        if pass_single:
            n_pass_single += 1
            n_pass_multi += 1  # multi cũng pass vì thử 1 lần đã đúng, nhưng ta không chạy multi khi đã pass
        else:
            # Chỉ chạy multi nếu single fail
            pass_multi = handler.verify_multi_start(
                feature_verify, helper_data, selection_mask, key_hash, K=K, sigma=sigma
            )
            if pass_multi:
                n_pass_multi += 1
                n_rescued += 1

    print(f"K={K}, sigma={sigma}")
    print(f"Tổng số cặp genuine: {n_total}")
    print(
        f"GMR single (1 lần): {n_pass_single}/{n_total} ({100*n_pass_single/n_total:.2f}%)"
    )
    print(
        f"GMR multi-start ({K} lần): {n_pass_multi}/{n_total} ({100*n_pass_multi/n_total:.2f}%)"
    )
    print(f"Số ca được cứu thêm: {n_rescued}")

    return {
        "n_total": n_total,
        "pass_single": n_pass_single,
        "pass_multi": n_pass_multi,
        "rescued": n_rescued,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--K", type=int, default=5, help="Số lần thử giải mã (mặc định 5)")
    ap.add_argument(
        "--sigma",
        type=float,
        default=0.2,
        help="Độ lệch chuẩn nhiễu Gaussian thêm vào LLR (mặc định 0.2)",
    )
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

    handler = MultiStartDiagnosticHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(handler, pairs_iter, K=args.K, sigma=args.sigma)


if __name__ == "__main__":
    main()
