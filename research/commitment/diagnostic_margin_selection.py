"""
diagnostic_margin_selection.py

Đánh giá hiệu quả của việc chọn vị trí bit theo margin (khoảng cách đến ngưỡng
nhị phân hoá) thay vì chọn ngẫu nhiên. Hướng I: cá nhân hoá ngay từ một ảnh
enrollment, không cần multi‑sample, không giảm entropy, không rò rỉ khóa.

Cách dùng:
    # Baseline (ngẫu nhiên)
    python research/commitment/diagnostic_margin_selection.py --mode random --max-pairs 200
    # Margin selection
    python research/commitment/diagnostic_margin_selection.py --mode margin --max-pairs 200

Kết quả in ra GMR và BER trên 1536/832 bit.
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


# --- Handler hỗ trợ cả hai chế độ chọn vị trí ---
class DiagnosticHandler(SecureWiFaKeyHandler):
    """
    Kế thừa SecureWiFaKeyHandler, bổ sung khả năng chọn selection_mask theo
    margin (mode='margin') hoặc ngẫu nhiên (mode='random'). Luôn trả về
    random_key (oracle) cho mục đích chẩn đoán.
    """

    def __init__(self, selection_mode="random", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if selection_mode not in ("random", "margin"):
            raise ValueError("selection_mode must be 'random' or 'margin'")
        self.selection_mode = selection_mode

    def enroll(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        if self.selection_mode == "random":
            # Giống hệt SecureWiFaKeyHandler.enroll nhưng trả thêm random_key
            rng = np.random.default_rng()
            selection_indices = rng.choice(
                len(b_full), size=self.feature_length, replace=False
            )
            selection_indices.sort()
        else:  # margin
            # Tính margin cho từng bit trong b_full
            projected = np.dot(feature_vector_float, self.M_matrix)  # (512,)
            intervals = self.intervals  # mảng 1D các ngưỡng, length = n_thr (3)
            n_dim = len(projected)  # 512
            margins = np.zeros(len(b_full), dtype=np.float32)
            # Thứ tự bit: ngưỡng 0 cho 512 chiều, ngưỡng 1 cho 512 chiều, ngưỡng 2 cho 512 chiều
            for t_idx, thr in enumerate(intervals):
                start = t_idx * n_dim
                end = start + n_dim
                margins[start:end] = np.abs(projected - thr)

            # Chọn top feature_length indices có margin lớn nhất
            # Dùng argpartition để lấy top mà không cần sắp xếp toàn bộ
            top_indices = np.argpartition(-margins, self.feature_length)[
                : self.feature_length
            ]
            selection_indices = np.sort(top_indices)

        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected = b_full[selection_indices]

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        return helper_data, selection_mask, key_hash, random_key.flatten()

    def verify_from_b_full(self, b_full, helper_data, selection_mask):
        """
        Giải mã LDPC từ b_full (đã sửa hoặc chưa), trả về reconstructed_key (160 bit).
        """
        selection_indices = np.where(selection_mask == 1)[0]
        if len(selection_indices) != self.feature_length:
            raise ValueError("selection_mask does not match feature_length")
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


def run_diagnostic(handler, genuine_pairs_iter):
    n_total = 0
    n_pass = 0
    ber_1536 = []
    ber_832 = []

    for feature_enroll, feature_verify in genuine_pairs_iter:
        n_total += 1

        helper_data, selection_mask, key_hash, random_key = handler.enroll(
            feature_enroll
        )

        b_full_verify = handler._binarize_full(feature_verify).astype(np.uint8)
        rec_key = handler.verify_from_b_full(b_full_verify, helper_data, selection_mask)
        if np.array_equal(rec_key, random_key):
            n_pass += 1

        # BER
        b_full_enroll = handler._binarize_full(feature_enroll).astype(np.uint8)
        ber_1536.append(np.mean(b_full_enroll != b_full_verify))
        sel_idx = np.where(selection_mask == 1)[0]
        ber_832.append(np.mean(b_full_enroll[sel_idx] != b_full_verify[sel_idx]))

    print(f"Chế độ chọn vị trí: {handler.selection_mode}")
    print(f"Tổng số cặp genuine: {n_total}")
    print(f"GMR: {n_pass}/{n_total} ({100*n_pass/n_total:.2f}%)")
    print(
        f"BER trung bình 1536 bit: {np.mean(ber_1536):.4f} (median {np.median(ber_1536):.4f})"
    )
    print(
        f"BER trung bình 832 bit : {np.mean(ber_832):.4f} (median {np.median(ber_832):.4f})"
    )

    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "ber_1536_mean": np.mean(ber_1536),
        "ber_832_mean": np.mean(ber_832),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["random", "margin"],
        default="margin",
        help="Chế độ chọn vị trí: random (baseline) hoặc margin (đề xuất)",
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

    handler = DiagnosticHandler(
        selection_mode=args.mode,
        data_path=data_dir,
        weights_path=weights_path,
        biases_path=biases_path,
    )

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(handler, pairs_iter)


if __name__ == "__main__":
    main()
