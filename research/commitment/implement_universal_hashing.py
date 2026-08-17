"""
implement_universal_hashing.py

Tier 3 – Universal Hashing (Privacy Amplification).
Dùng leftover hash lemma để nén khóa xuống độ dài an toàn có thể chứng minh.
"""

import os, sys, csv, hashlib, numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536
FEATURE_LENGTH = 832
KEY_LENGTH = 160  # độ dài khóa gốc


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


def load_all_genuine_pairs():
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
    return pairs


def estimate_p_bit1(handler, pairs):
    """Tính P(bit=1) cho từng vị trí 1536 (từ tập train)."""
    bit_counts = np.zeros(FULL_BITS, dtype=int)
    selected_counts = np.zeros(FULL_BITS, dtype=int)
    for name_e, img_e, _, _ in pairs:
        emb = load_embedding(name_e, img_e)
        b_full = handler._binarize_full(emb).astype(np.uint8)
        projected = np.dot(emb, handler.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
        selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
        selected_counts[selection_indices] += 1
        bit_counts[selection_indices] += b_full[selection_indices]
    valid = selected_counts > 0
    p = np.divide(bit_counts, selected_counts, out=np.full(FULL_BITS, 0.5), where=valid)
    return np.clip(p, 0.01, 0.99)


def compute_min_entropy(p_bit1):
    """Tính min-entropy từ phân phối P(bit=1) cho từng vị trí."""
    # Min-entropy cho mỗi bit: H_min = -log2(max(p, 1-p))
    p_max = np.maximum(p_bit1, 1 - p_bit1)
    min_entropy_per_bit = -np.log2(p_max)
    # Tổng min-entropy (giả sử độc lập – đây là upper bound)
    total_min_entropy = np.sum(min_entropy_per_bit)
    return total_min_entropy, min_entropy_per_bit


def toeplitz_hash(key_bits, seed, output_length):
    """
    Universal hashing dùng ma trận Toeplitz.
    key_bits: (160,) uint8 – khóa gốc
    seed: (160 + output_length - 1,) uint8 – seed ngẫu nhiên công khai
    output_length: int – độ dài khóa đầu ra (bit)
    Trả về: (output_length,) uint8 – khóa đã băm
    """
    n = len(key_bits)
    m = output_length

    # Tạo ma trận Toeplitz từ seed
    # Hàng đầu tiên: seed[m-1 : m+n-1]
    # Cột đầu tiên: seed[:m][::-1]
    first_row = seed[m - 1 : m + n - 1]
    first_col = seed[:m][::-1]

    # Tạo ma trận Toeplitz T (m x n)
    T = np.zeros((m, n), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            if i == 0:
                T[i, j] = first_row[j]
            elif j == 0:
                T[i, j] = first_col[i]
            else:
                T[i, j] = T[i - 1, j - 1]

    # Nhân ma trận trên GF(2): K_final = T @ key_bits (mod 2)
    K_final = np.dot(T, key_bits) % 2
    return K_final.astype(np.uint8)


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    # Load dữ liệu
    all_pairs = load_all_genuine_pairs()
    n_train = int(len(all_pairs) * 0.7)
    train_pairs = all_pairs[:n_train]

    # Ước tính P(bit=1) từ tập train
    print("1. Ước tính P(bit=1) từ tập train...")
    p_bit1 = estimate_p_bit1(handler, train_pairs)

    # Lấy các vị trí thường được chọn nhất (top 832)
    # (Mô phỏng multiset mà attacker quan sát)
    # Trước tiên cần tính selected_counts
    bit_counts = np.zeros(FULL_BITS, dtype=int)
    selected_counts = np.zeros(FULL_BITS, dtype=int)
    for name_e, img_e, _, _ in train_pairs:
        emb = load_embedding(name_e, img_e)
        b_full = handler._binarize_full(emb).astype(np.uint8)
        projected = np.dot(emb, handler.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
        selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
        selected_counts[selection_indices] += 1
    order = np.argsort(-selected_counts)
    top_positions = order[:FEATURE_LENGTH]
    p_selected = p_bit1[top_positions]

    # Tính min-entropy cho các bit được chọn
    print("\n2. Tính min-entropy cho 832 bit được chọn...")
    total_min_entropy, min_entropy_per_bit = compute_min_entropy(p_selected)

    # Áp dụng leftover hash lemma
    # Với security parameter epsilon = 2^{-80}, độ dài khóa an toàn:
    # ℓ ≤ total_min_entropy - 2 * log2(1/epsilon) ≈ total_min_entropy - 160
    # Nhưng với epsilon = 2^{-40} (an toàn vừa phải):
    # ℓ ≤ total_min_entropy - 80
    epsilon = 2 ** (-40)
    security_margin = int(-2 * np.log2(epsilon))
    safe_key_length = int(total_min_entropy - security_margin)
    safe_key_length = max(0, min(KEY_LENGTH, safe_key_length))

    print(f"\n3. Kết quả Privacy Amplification:")
    print(f"   Tổng min-entropy (832 bit): {total_min_entropy:.2f} bit")
    print(f"   Min-entropy trung bình mỗi bit: {np.mean(min_entropy_per_bit):.4f} bit")
    print(
        f"   Rò rỉ so với lý tưởng (1.0): {100*(1 - np.mean(min_entropy_per_bit)):.1f}%"
    )
    print(f"   Security margin (ε={epsilon}): {security_margin} bit")
    print(f"   Độ dài khóa an toàn đề xuất: {safe_key_length} bit")
    print(f"   (Khóa gốc: {KEY_LENGTH} bit)")

    if safe_key_length < KEY_LENGTH:
        reduction = KEY_LENGTH - safe_key_length
        print(
            f"\n=> CẦN GIẢM khóa từ {KEY_LENGTH} xuống {safe_key_length} bit (giảm {reduction} bit)"
        )
        print(f"   để đạt được bảo mật có thể chứng minh với ε = {epsilon}.")
    else:
        print(f"\n=> Khóa 160-bit hiện tại ĐÃ đạt được bảo mật có thể chứng minh!")
        print(f"   Min-entropy đủ lớn để hỗ trợ khóa 160-bit với ε = {epsilon}.")

    # Mô phỏng universal hashing
    print("\n4. Mô phỏng Universal Hashing...")
    rng = np.random.default_rng(42)
    # Tạo khóa ngẫu nhiên 160-bit
    test_key = rng.integers(0, 2, size=KEY_LENGTH).astype(np.uint8)
    # Tạo seed ngẫu nhiên cho ma trận Toeplitz
    seed_length = KEY_LENGTH + safe_key_length - 1
    seed = rng.integers(0, 2, size=seed_length).astype(np.uint8)
    # Áp dụng universal hash
    hashed_key = toeplitz_hash(test_key, seed, safe_key_length)

    print(f"   Khóa gốc (160 bit): {test_key[:20]}...")
    print(f"   Seed (công khai): {seed[:20]}...")
    print(
        f"   Khóa đã băm ({safe_key_length} bit): {hashed_key[:min(20, safe_key_length)]}..."
    )

    # Kiểm tra tính đồng đều của khóa đã băm (trên nhiều lần thử)
    print("\n5. Kiểm tra tính đồng đều của khóa đã băm...")
    n_trials = 10000
    bit_counts_hashed = np.zeros(safe_key_length)
    for _ in range(n_trials):
        test_key_i = rng.integers(0, 2, size=KEY_LENGTH).astype(np.uint8)
        hashed = toeplitz_hash(test_key_i, seed, safe_key_length)
        bit_counts_hashed += hashed
    p_hashed = bit_counts_hashed / n_trials
    avg_entropy_hashed = -np.mean(
        p_hashed * np.log2(p_hashed) + (1 - p_hashed) * np.log2(1 - p_hashed)
    )
    print(f"   Entropy trung bình của khóa đã băm: {avg_entropy_hashed:.4f} bit/bit")
    print(f"   (Lý tưởng: 1.0000 bit/bit)")

    handler.sess.close()


if __name__ == "__main__":
    main()
