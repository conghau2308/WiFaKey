"""
diagnostic_thermometer_consistency.py

Mục đích: đánh giá hiệu quả của việc sửa lỗi cục bộ dựa trên tính nhất quán
của thermometer code (mỗi chiều embedding được biểu diễn bằng 3 bit đơn điệu).
Script so sánh GMR giữa luồng gốc (SecureWiFaKeyHandler) và luồng có sửa nhất
quán trước khi giải mã LDPC.

Không sửa handler gốc – kế thừa SecureWiFaKeyHandler, thêm oracle key và khả
năng can thiệp b_full trước bước chọn bit.

Yêu cầu: file này đặt trong research/commitment/, chạy từ thư mục gốc dự án.
"""

import argparse
import csv
import hashlib
import os
import sys
import numpy as np

# Đảm bảo import được wifakey_module và research.commitment
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation


# --- Các hàm tải dữ liệu (giữ nguyên convention) ---
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


# --- Sửa lỗi nhất quán thermometer code ---
def repair_thermometer_code(b_full: np.ndarray):
    """
    b_full: mảng 1536 bit (512 chiều * 3 ngưỡng).
    Giả định thứ tự: cụm 3 bit liên tiếp ứng với cùng một chiều.
    Ví dụ: bit[0],bit[1],bit[2] cho chiều 0; bit[3],bit[4],bit[5] cho chiều 1; ...

    Trả về:
        b_repaired: mảng đã sửa.
        n_repaired: số cụm (chiều) đã được sửa.
    """
    n_dims = len(b_full) // 3
    if len(b_full) % 3 != 0:
        raise ValueError("b_full length must be multiple of 3")
    b = b_full.copy()
    repaired_dim = 0
    valid_patterns = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    for i in range(n_dims):
        b0, b1, b2 = int(b[3 * i]), int(b[3 * i + 1]), int(b[3 * i + 2])
        if (b0, b1, b2) in valid_patterns:
            continue
        # Tìm pattern hợp lệ gần nhất theo khoảng cách Hamming
        best = min(
            valid_patterns, key=lambda p: (p[0] != b0) + (p[1] != b1) + (p[2] != b2)
        )
        b[3 * i] = best[0]
        b[3 * i + 1] = best[1]
        b[3 * i + 2] = best[2]
        repaired_dim += 1
    return b, repaired_dim


# --- Handler chẩn đoán ---
class ThermometerRepairHandler(SecureWiFaKeyHandler):
    """
    Handler kế thừa SecureWiFaKeyHandler (an toàn), bổ sung:
        - enroll_with_oracle(): trả về random_key (chỉ dùng trong môi trường test).
        - verify_from_b_full(): giải mã trực tiếp từ b_full (đã sửa hoặc chưa) và trả về
          reconstructed_key (không hash check).
    """

    def enroll_with_oracle(self, feature_vector_float: np.ndarray):
        """
        Gọi enroll() của lớp cha, đồng thời trả thêm random_key làm oracle.
        """
        # Lấy helper_data, selection_mask, key_hash từ phương thức enroll an toàn
        helper_data, selection_mask, key_hash = super().enroll(feature_vector_float)
        # Tạo lại random_key (cần phải lấy từ bên trong enroll, nhưng ta không thể.
        # Giải pháp: tạm thời gọi lại logic enroll để lấy key.
        # Ta sẽ viết lại một enroll riêng cho diagnostic để expose key.
        # Tuy nhiên, để tránh sửa code gốc, ta có thể thực hiện lại các bước với
        # cùng feature_vector. Vì enroll dùng random_key ngẫu nhiên, ta không thể
        # tái tạo chính xác cùng một random_key nếu gọi lại. Do đó, ta cần override
        # hoàn toàn enroll để kiểm soát được random_key. Cách này an toàn vì chỉ
        # dùng trong test.
        pass  # Sẽ không dùng super().enroll() nữa, mà tự viết lại

    # Thay vào đó, ta sẽ override enroll hoàn toàn cho mục đích chẩn đoán
    def enroll(self, feature_vector_float: np.ndarray):
        # Sử dụng logic từ SecureWiFaKeyHandler nhưng lưu random_key
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        if len(b_full) <= self.feature_length:
            raise ValueError("Not enough bits after binarization")

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

        # Trả thêm random_key cho mục đích chẩn đoán
        return helper_data, selection_mask, key_hash, random_key.flatten()

    def verify_from_b_full(self, b_full, helper_data, selection_mask):
        """
        Thực hiện giải mã LDPC từ b_full (đã được sửa hoặc chưa).
        Trả về reconstructed_key (160 bit).
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
    n_pass_original = 0
    n_pass_repaired = 0
    n_fail_both = 0

    ber_1536_raw = []
    ber_1536_repaired = []
    ber_832_raw = []
    ber_832_repaired = []
    dims_repaired_list = []

    for feature_enroll, feature_verify in genuine_pairs_iter:
        n_total += 1

        # Enrollment: lấy helper, mask, key_hash, random_key
        helper_data, selection_mask, key_hash, random_key = handler.enroll(
            feature_enroll
        )

        # Verify gốc (không sửa)
        b_full_verify_raw = handler._binarize_full(feature_verify).astype(np.uint8)
        rec_key_orig = handler.verify_from_b_full(
            b_full_verify_raw, helper_data, selection_mask
        )
        pass_orig = np.array_equal(rec_key_orig, random_key)

        # Verify có sửa nhất quán
        b_full_verify_repaired, n_rep = repair_thermometer_code(b_full_verify_raw)
        rec_key_rep = handler.verify_from_b_full(
            b_full_verify_repaired, helper_data, selection_mask
        )
        pass_rep = np.array_equal(rec_key_rep, random_key)

        if pass_orig:
            n_pass_original += 1
        if pass_rep:
            n_pass_repaired += 1
        if not pass_orig and not pass_rep:
            n_fail_both += 1

        # Tính BER
        b_full_enroll = handler._binarize_full(feature_enroll).astype(np.uint8)
        ber_1536_raw.append(np.mean(b_full_enroll != b_full_verify_raw))
        ber_1536_repaired.append(np.mean(b_full_enroll != b_full_verify_repaired))

        sel_idx = np.where(selection_mask == 1)[0]
        b_sel_enroll = b_full_enroll[sel_idx]
        b_sel_verify_raw = b_full_verify_raw[sel_idx]
        b_sel_verify_rep = b_full_verify_repaired[sel_idx]
        ber_832_raw.append(np.mean(b_sel_enroll != b_sel_verify_raw))
        ber_832_repaired.append(np.mean(b_sel_enroll != b_sel_verify_rep))

        dims_repaired_list.append(n_rep)

    print(f"Tổng số cặp genuine: {n_total}")
    print(
        f"GMR gốc (không sửa): {n_pass_original}/{n_total} ({100*n_pass_original/n_total:.2f}%)"
    )
    print(
        f"GMR có sửa nhất quán: {n_pass_repaired}/{n_total} ({100*n_pass_repaired/n_total:.2f}%)"
    )
    print(f"Số cặp cùng fail: {n_fail_both}")
    print(f"Số cặp sửa cứu thêm được: {n_pass_repaired - n_pass_original}")
    print()
    print("Thống kê BER:")
    print(
        f"  BER 1536 bit (raw)      : mean {np.mean(ber_1536_raw):.4f}, median {np.median(ber_1536_raw):.4f}"
    )
    print(
        f"  BER 1536 bit (repaired) : mean {np.mean(ber_1536_repaired):.4f}, median {np.median(ber_1536_repaired):.4f}"
    )
    print(
        f"  BER 832 bit (raw)       : mean {np.mean(ber_832_raw):.4f}, median {np.median(ber_832_raw):.4f}"
    )
    print(
        f"  BER 832 bit (repaired)  : mean {np.mean(ber_832_repaired):.4f}, median {np.median(ber_832_repaired):.4f}"
    )
    print(f"  Số cụm được sửa trung bình/cặp: {np.mean(dims_repaired_list):.2f} / 512")

    return {
        "n_total": n_total,
        "pass_orig": n_pass_original,
        "pass_rep": n_pass_repaired,
        "fail_both": n_fail_both,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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

    handler = ThermometerRepairHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    # Kiểm tra nhanh thứ tự bit của thermometer code
    print("Kiểm tra cấu trúc thermometer code (3 bit đầu tiên của vài embedding):")
    # Tạo một embedding giả hoặc dùng một mẫu thật để xem
    try:
        sample_iter = genuine_pairs_iter(pairs_csv, cache_dir, max_pairs=1)
        e1, e2 = next(sample_iter)
        b_full = handler._binarize_full(e1).astype(np.uint8)
        print(f"  full_binary_length = {len(b_full)}")
        print(f"  10 cụm 3-bit đầu tiên: {b_full[:30].reshape(-1, 3)}")
        # Kiểm tra tỉ lệ pattern không hợp lệ trên toàn bộ
        invalid_count = 0
        for i in range(len(b_full) // 3):
            b0, b1, b2 = b_full[3 * i], b_full[3 * i + 1], b_full[3 * i + 2]
            if (b0, b1, b2) not in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]:
                invalid_count += 1
        print(
            f"  Số cụm không hợp lệ trên embedding đầu: {invalid_count}/{len(b_full)//3}"
        )
        print(
            "  Nếu số này >0 và việc sửa diễn ra bình thường thì giả định thứ tự đúng."
        )
    except Exception as e:
        print(f"  Không thể kiểm tra mẫu: {e}")

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(handler, pairs_iter)


if __name__ == "__main__":
    main()
