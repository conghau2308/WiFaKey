"""
simulate_attack_v2.py (ĐÃ SỬA LỖI a, b, c)

Mô phỏng tấn công thống kê dựa trên rò rỉ entropy từ reliability mask.
- Ước tính prior P(bit=1|vị trí 1536) từ tập train.
- Tìm kiếm tổ hợp (combinatorial search) thay vì đường thẳng.
- Dùng đúng Empirical LLR và pipeline thật.
"""

import os, sys, csv, hashlib, numpy as np
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
FEATURE_LENGTH = 832
KEY_LENGTH = 160


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


def estimate_prior_global(handler, pairs):
    """
    Ước tính prior cho TỪNG vị trí trong 1536 bit:
    - p_selected[pos]: xác suất vị trí pos được chọn vào mask
    - p_bit1[pos]: xác suất bit=1 tại vị trí pos
    """
    count_selected = np.zeros(FULL_BITS)
    count_bit1 = np.zeros(FULL_BITS)
    count_total = 0

    for name_e, img_e, name_v, img_v in pairs:
        emb = load_embedding(name_e, img_e)
        b_full = handler._binarize_full(emb).astype(np.uint8)
        projected = np.dot(emb, handler.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, handler.intervals)

        # Tìm 832 vị trí margin lớn nhất
        selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
        for pos in selection_indices:
            count_selected[pos] += 1
            if b_full[pos] == 1:
                count_bit1[pos] += 1
        count_total += 1

    p_selected = count_selected / count_total
    p_bit1 = np.divide(
        count_bit1,
        count_selected,
        out=np.full(FULL_BITS, 0.5),
        where=count_selected > 0,
    )
    p_bit1 = np.clip(p_bit1, 0.01, 0.99)
    return p_selected, p_bit1


def combinatorial_attack(
    handler, helper_data, sel_mask, key_hash, p_bit1, emp_mod, max_attempts=50000
):
    """
    Tấn công tổ hợp: thử lật L bit có entropy thấp nhất (gần 0 hoặc 1 nhất).
    Bắt đầu với L=0 (không lật), rồi L=1, L=2,... cho đến khi hết budget.
    """
    # Lấy danh sách vị trí được chọn (trong không gian 1536)
    selected_positions = np.where(sel_mask == 1)[0]  # shape (832,)

    # Tính entropy cho từng vị trí được chọn
    p = p_bit1[selected_positions]
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    # Sắp xếp theo entropy tăng dần (bit dễ đoán nhất trước)
    sorted_local_indices = np.argsort(entropy)  # chỉ số trong mảng 832

    # Tạo ứng viên cơ sở: bit có p>0.5 thì đoán là 1, ngược lại đoán 0
    base_bits = (p > 0.5).astype(np.uint8)

    # Lấy noisy_bits từ helper_data (để dùng trong try_decode)
    b_full_v = handler._binarize_full(np.zeros(512))  # dummy, sẽ bị ghi đè
    # Thực tế cần b_full_v của ảnh verify để tính margin, nhưng attacker không có ảnh verify.
    # Attacker chỉ có helper_data và mask. Họ phải đoán b_selected, rồi tính noisy = b_selected XOR helper.
    # try_decode sẽ nhận noisy và margin. Vì attacker không có ảnh verify, họ không có margin thật.
    # => Họ dùng margin trung bình hoặc margin từ ảnh enrollment (cũng không có).
    # Để mô phỏng thực tế, ta cho attacker dùng margin từ tập train (giả định họ có dữ liệu công khai).
    # Nhưng đây là một giả định mạnh. Tạm thời ta dùng margin = 0 (LLR = 0) cho các bit không chắc chắn,
    # và margin lớn cho các bit chắc chắn. Đây là cách tiếp cận bảo thủ (có lợi cho attacker).

    # Ta sẽ tạo margin giả: bit có entropy thấp (p gần 0 hoặc 1) thì margin cao, ngược lại margin thấp.
    margin_fake = np.zeros(FEATURE_LENGTH)
    margin_fake[sorted_local_indices] = np.linspace(
        2.0, 0.1, FEATURE_LENGTH
    )  # giảm dần

    # Thử L=0 (base candidate)
    noisy_base = np.logical_xor(base_bits, helper_data).astype(np.uint8)
    if try_decode_with_llr(handler, noisy_base, margin_fake, key_hash, emp_mod):
        return 1, "base"

    # Duyệt qua các mức L=1,2,...
    attempt_count = 1
    # Sắp xếp các vị trí cần lật (ưu tiên entropy thấp)
    flip_candidates = sorted_local_indices[:30]  # Giới hạn 30 vị trí dễ đoán nhất

    from itertools import combinations

    for L in range(1, min(6, len(flip_candidates) + 1)):  # L=1 đến 5
        for combo in combinations(flip_candidates, L):
            attempt_count += 1
            if attempt_count > max_attempts:
                return None, "budget_exceeded"

            # Tạo ứng viên bằng cách lật các bit trong combo
            candidate = base_bits.copy()
            for idx in combo:
                candidate[idx] ^= 1

            noisy = np.logical_xor(candidate, helper_data).astype(np.uint8)
            if try_decode_with_llr(handler, noisy, margin_fake, key_hash, emp_mod):
                return attempt_count, f"L={L}"

    return None, "exhausted"


def try_decode_with_llr(handler, noisy_bits, margin, key_hash, emp_mod):
    """Giải mã với Empirical LLR (giống pipeline thật)."""
    llr = emp_mod.modulate(noisy_bits, context={"margin": margin}).flatten()
    llr = llr.reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[:KEY_LENGTH]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    emp_mod = EmpiricalLLR(
        lookup_path=os.path.join(
            _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
        )
    )

    # ---- 1. Ước tính prior toàn cục ----
    print("Ước tính prior từ tập train...")
    train_pairs = load_genuine_pairs(max_pairs=400)
    p_selected, p_bit1 = estimate_prior_global(handler, train_pairs)

    # Đo entropy tại các vị trí được chọn (trung bình)
    selected_mask = p_selected > 0
    p_sel = p_bit1[selected_mask]
    entropy_sel = -(p_sel * np.log2(p_sel) + (1 - p_sel) * np.log2(1 - p_sel))
    print(f"Entropy trung bình tại vị trí được chọn: {np.mean(entropy_sel):.4f} bit")

    # ---- 2. Tấn công ----
    print("\nMô phỏng tấn công tổ hợp...")
    test_pairs = load_genuine_pairs(max_pairs=100)  # Giới hạn 100 cặp để chạy nhanh
    attempts_list = []
    success_count = 0

    for name_e, img_e, name_v, img_v in test_pairs:
        emb_e = load_embedding(name_e, img_e)
        helper, sel_mask, key_hash = handler.enroll(emb_e)
        attempts, method = combinatorial_attack(
            handler, helper, sel_mask, key_hash, p_bit1, emp_mod, max_attempts=20000
        )
        if attempts is not None:
            attempts_list.append(attempts)
            success_count += 1

    print(f"\n=== KẾT QUẢ TẤN CÔNG V2 ===")
    print(f"Số cặp test: {len(test_pairs)}")
    print(f"Số cặp tấn công thành công: {success_count}/{len(test_pairs)}")
    if attempts_list:
        print(f"Số lần thử trung bình: {np.mean(attempts_list):.2f}")
        print(f"Min: {np.min(attempts_list)}, Max: {np.max(attempts_list)}")
    else:
        print("Không có cặp nào bị tấn công thành công.")

    handler.sess.close()


if __name__ == "__main__":
    main()
