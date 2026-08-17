"""
simulate_attack.py

Mô phỏng tấn công thống kê dựa trên rò rỉ entropy từ reliability mask.
- Ước tính P(bit=1|vị trí được chọn) từ tập train.
- Thực hiện weighted guessing attack để khôi phục khóa.
- Đo số lần thử trung bình cần thiết.
"""

import os, sys, csv, hashlib, numpy as np
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

N, m, Z = 52, 42, 16
FULL_BITS = 1536
FEATURE_LENGTH = 832
KEY_LENGTH = 160  # code_k * Z = 10 * 16


def load_embedding(name, imagenum):
    path = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "embeddings_cache",
        f"{name}_{int(imagenum):04d}.npy",
    )
    return np.load(path)


def load_genuine_pairs(max_pairs=None):
    pairs_csv = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "pairs",
        "tune_genuine.csv",
    )
    pairs = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                (
                    row["name_enroll"],
                    int(row["imagenum_enroll"]),
                    row["name_verify"],
                    int(row["imagenum_verify"]),
                )
            )
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def estimate_prior(handler, pairs):
    """
    Ước tính P(bit=1 | vị trí được chọn) cho từng vị trí trong 832 bit.
    Trả về mảng prior[832] chứa xác suất P(bit=1) tại vị trí đó.
    """
    # Đếm số lần mỗi vị trí được chọn và số lần bit=1 tại đó
    count_selected = np.zeros(FEATURE_LENGTH)
    count_bit1 = np.zeros(FEATURE_LENGTH)

    for name_e, img_e, name_v, img_v in pairs:
        emb = load_embedding(name_e, img_e)
        b_full = handler._binarize_full(emb).astype(np.uint8)
        projected = np.dot(emb, handler.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, handler.intervals)

        # Tìm 832 vị trí margin lớn nhất
        selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
        selection_indices.sort()

        # Cập nhật đếm
        for pos_832, pos_1536 in enumerate(selection_indices):
            count_selected[pos_832] += 1
            if b_full[pos_1536] == 1:
                count_bit1[pos_832] += 1

    # Tính xác suất, tránh chia cho 0
    prior = np.divide(
        count_bit1,
        count_selected,
        out=np.full(FEATURE_LENGTH, 0.5),
        where=count_selected > 0,
    )
    prior = np.clip(prior, 0.01, 0.99)  # Giới hạn để tránh p=0 hoặc p=1
    return prior


def weighted_guessing_attack(
    handler, helper_data, sel_mask, key_hash, prior, max_attempts=100000
):
    """
    Tấn công đoán khóa bằng cách thử các tổ hợp bit có khả năng cao nhất.
    prior: P(bit=1|vị trí được chọn) cho 832 bit.
    Cách đơn giản: tạo ứng viên bằng cách lấy bit có xác suất cao nhất (0 hoặc 1) tại mỗi vị trí,
    sau đó thử lật các bit có entropy thấp nhất (gần 0 hoặc 1 nhất).
    """
    # Tạo ứng viên cơ sở: bit = 1 nếu prior > 0.5, ngược lại bit = 0
    base_candidate = (prior > 0.5).astype(np.uint8)

    # Tính "độ không chắc chắn" của mỗi bit: uncertainty = 1 - max(p, 1-p)
    uncertainty = 1 - np.maximum(prior, 1 - prior)
    # Sắp xếp vị trí theo uncertainty tăng dần (bit chắc chắn nhất trước)
    sorted_indices = np.argsort(uncertainty)

    # Thử base candidate trước
    b_selected_guess = base_candidate.copy()
    noisy_guess = np.logical_xor(b_selected_guess, helper_data).astype(np.uint8)
    # Giải mã thử
    if try_decode(handler, noisy_guess, key_hash):
        return 1  # Thành công ngay lần đầu

    # Thử lật từng bit một (theo thứ tự uncertainty tăng dần)
    for attempt, idx in enumerate(sorted_indices, start=2):
        b_selected_guess[idx] ^= 1  # Lật bit
        noisy_guess = np.logical_xor(b_selected_guess, helper_data).astype(np.uint8)
        if try_decode(handler, noisy_guess, key_hash):
            return attempt
        if attempt >= max_attempts:
            break

    return None  # Không thành công trong giới hạn


def try_decode(handler, noisy_bits, key_hash):
    """Thử giải mã với noisy_bits (832,) và kiểm tra hash."""
    from wifakey_module.wifakey_lib import Modulation

    llr = Modulation.BPSK(noisy_bits).astype(np.float32).reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[:KEY_LENGTH]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    # ---- 1. Ước tính prior từ tập train (400 cặp) ----
    print("Ước tính prior từ tập train...")
    train_pairs = load_genuine_pairs(max_pairs=400)
    prior = estimate_prior(handler, train_pairs)
    entropy_per_bit = -(prior * np.log2(prior) + (1 - prior) * np.log2(1 - prior))
    print(
        f"Entropy trung bình mỗi bit (theo prior): {np.mean(entropy_per_bit):.4f} bit"
    )
    print(f"Entropy tổng 832 bit (theo prior): {np.sum(entropy_per_bit):.2f} bit")

    # ---- 2. Tấn công trên tập test (200 cặp) ----
    print("\nMô phỏng tấn công trên tập test...")
    test_pairs = load_genuine_pairs(max_pairs=200)
    attempts_list = []
    success_count = 0

    for name_e, img_e, name_v, img_v in test_pairs:
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)

        helper, sel_mask, key_hash = handler.enroll(emb_e)
        attempts = weighted_guessing_attack(handler, helper, sel_mask, key_hash, prior)
        if attempts is not None:
            attempts_list.append(attempts)
            success_count += 1

    # ---- 3. Kết quả ----
    print(f"\n=== KẾT QUẢ TẤN CÔNG ===")
    print(f"Số cặp test: {len(test_pairs)}")
    print(f"Số cặp tấn công thành công: {success_count}/{len(test_pairs)}")
    if attempts_list:
        print(f"Số lần thử trung bình (khi thành công): {np.mean(attempts_list):.2f}")
        print(f"Số lần thử nhỏ nhất: {np.min(attempts_list)}")
        print(f"Số lần thử lớn nhất: {np.max(attempts_list)}")
        # Ước tính entropy hiệu dụng: log2(số lần thử trung bình)
        effective_entropy = np.log2(np.mean(attempts_list))
        print(f"Entropy hiệu dụng ước tính: {effective_entropy:.2f} bit")
        print(f"(So với entropy lý tưởng của khóa: 160 bit)")
    else:
        print("Không có cặp nào bị tấn công thành công trong giới hạn thử.")

    handler.sess.close()


if __name__ == "__main__":
    main()
