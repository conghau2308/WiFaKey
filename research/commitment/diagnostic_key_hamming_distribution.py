"""
diagnostic_key_hamming_distribution.py

Đo khoảng cách Hamming giữa random_key GỐC (biết trước, vì ta tự sinh lúc
enroll trong benchmark) và reconstructed_key (decode được lúc verify), trên
TOÀN BỘ cặp genuine (không chỉ ca thất bại) -- để trả lời dứt điểm: decoder
thất bại kiểu "thác nước" (sai rất nhiều bit khi thất bại) hay "gần đúng"
(sai ít bit, chỉ hụt sát ngưỡng)?

Nếu phần lớn thất bại có Hamming distance lớn (gần 80/160 -- gần ngẫu
nhiên) => C.5 (giảm effective_key_length ở bất kỳ mức nào) không thể cứu
được => đóng C.5 dứt điểm.
Nếu có cụm đáng kể ở mức thấp (~20-40/160) => còn cửa thử effective_key_length
thấp hơn nữa.

Cách chạy:
    python research/commitment/diagnostic_key_hamming_distribution.py \\
        --max-pairs 200   # bỏ flag để chạy full tune_genuine.csv
"""

import argparse
import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from wifakey_module.wifakey_lib import Modulation
from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler


class DiagnosticKeyHandler(SecureWiFaKeyHandler):
    """Giống hệt v1, nhưng enroll() trả thêm random_key thật, và có thêm
    hàm decode_key() trả về reconstructed_key thô (không hash) để so sánh
    trực tiếp -- CHỈ dùng để chẩn đoán, không dùng để deploy (không nên để
    lộ random_key ra ngoài phạm vi benchmark nội bộ)."""

    def enroll_with_ground_truth(self, feature_vector_float: np.ndarray):
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

        return helper_data, selection_mask, random_key.flatten()

    def decode_key_with_llr(
        self, feature_vector_float: np.ndarray, helper_data, selection_mask
    ):
        """Giống decode_key() nhưng trả thêm biên độ LLR thô (trước threshold)
        cho self.key_length vị trí đầu -- dùng để kiểm tra tương quan giữa
        độ tin cậy decoder và vị trí lỗi thật."""
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
        decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
        llr_magnitude = np.abs(y_pred_llr.flatten())
        return decoded_codeword[: self.key_length], llr_magnitude[: self.key_length]

    def decode_key(
        self, feature_vector_float: np.ndarray, helper_data, selection_mask
    ) -> np.ndarray:
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
        decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
        return decoded_codeword[: self.key_length]


def load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")

    with open(os.path.join(pairs_dir, "tune_genuine.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if args.max_pairs:
        rows = rows[: args.max_pairs]

    print(f"Genuine pairs: {len(rows)}")
    handler = DiagnosticKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    hamming_distances = []
    # Với mỗi ca có lỗi (0 < HD < key_length), ghi lại: trong số 32 bit có
    # |LLR| THẤP NHẤT (ứng viên "bỏ động"), có bao nhiêu % bit lỗi thật nằm
    # trong đó -- so với kỳ vọng ngẫu nhiên (32/160=20%).
    capture_rates = []

    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            helper_data, mask, true_key = handler.enroll_with_ground_truth(e1)
            reconstructed_key, llr_mag = handler.decode_key_with_llr(
                e2, helper_data, mask
            )
            hd = int(np.sum(true_key != reconstructed_key))
            hamming_distances.append(hd)

            if 0 < hd < len(true_key):
                error_positions = set(
                    np.where(true_key != reconstructed_key)[0].tolist()
                )
                lowest_conf_32 = set(np.argsort(llr_mag)[:32].tolist())
                captured = len(error_positions & lowest_conf_32)
                capture_rates.append(captured / len(error_positions))
        except Exception as e:
            print(f"  [WARN] lỗi ({row}): {e}", file=sys.stderr)

    hd_arr = np.array(hamming_distances)
    key_length = handler.key_length

    print(
        f"\n=== Phân phối Hamming distance (random_key vs reconstructed_key, /{key_length} bit) ==="
    )
    print(f"Số mẫu: {len(hd_arr)}")
    print(
        f"Thành công hoàn toàn (HD=0): {(hd_arr == 0).sum()} ({100*(hd_arr==0).mean():.1f}%)"
    )
    print(f"Mean HD: {hd_arr.mean():.2f}, Median: {np.median(hd_arr):.1f}")
    print(
        f"Percentiles: p50={np.percentile(hd_arr,50):.1f}, p75={np.percentile(hd_arr,75):.1f}, "
        f"p90={np.percentile(hd_arr,90):.1f}, p95={np.percentile(hd_arr,95):.1f}"
    )

    print("\nPhân bố theo khoảng (chỉ tính các ca HD > 0, tức thất bại):")
    failed = hd_arr[hd_arr > 0]
    if len(failed) > 0:
        bins = [0, 8, 16, 24, 32, 48, 64, 80, key_length + 1]
        hist, edges = np.histogram(failed, bins=bins)
        for i in range(len(hist)):
            print(
                f"  HD trong [{edges[i]}, {edges[i+1]}): {hist[i]} ca ({100*hist[i]/len(failed):.1f}% trong số thất bại)"
            )
    else:
        print("  Không có ca thất bại nào trong mẫu này.")

    print(
        "\n=> Nếu phần lớn ca thất bại tập trung ở HD lớn (>=48-64/160, gần "
        "50% -- gần ngẫu nhiên): xác nhận decoder thất bại kiểu 'thác nước', "
        "C.5 (giảm effective_key_length ở BẤT KỲ mức nào) không cứu được -- "
        "đóng C.5.\n"
        "   Nếu có cụm đáng kể ở HD thấp (<=32-40/160): còn cửa thử "
        "effective_key_length thấp hơn 128 (ví dụ 96, 64)."
    )

    if capture_rates:
        cr = np.array(capture_rates)
        expected_random = 32 / key_length  # ~0.20 nếu chọn ngẫu nhiên 32/160
        print(f"\n=== Tương quan giữa |LLR| thấp và vị trí lỗi thật ===")
        print(f"Số ca có lỗi (0 < HD < {key_length}) dùng để đo: {len(cr)}")
        print(
            f"Tỷ lệ trung bình bit lỗi thật nằm trong 32 bit |LLR| thấp nhất: {cr.mean()*100:.1f}%"
        )
        print(
            f"(Kỳ vọng nếu KHÔNG có tương quan gì, chọn ngẫu nhiên: {expected_random*100:.1f}%)"
        )
        if cr.mean() > expected_random * 1.5:
            print(
                "=> CÓ tương quan rõ rệt: bit lỗi thật có xu hướng trùng với bit "
                "decoder 'không chắc' (|LLR| thấp). Đáng đầu tư cơ chế bỏ ĐỘNG "
                "theo độ tin cậy (erasure-based) thay vì bỏ cố định vị trí."
            )
        else:
            print(
                "=> KHÔNG thấy tương quan rõ rệt hơn ngẫu nhiên đáng kể -- "
                "|LLR| của decoder không dự đoán tốt vị trí lỗi thật. Hướng "
                "'bỏ động theo độ tin cậy' khó khả thi, cần cân nhắc dừng C.5."
            )


if __name__ == "__main__":
    main()
