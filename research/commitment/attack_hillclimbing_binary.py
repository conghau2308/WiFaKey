"""
attack_hillclimbing_binary.py

Mô phỏng tấn công leo đồi với phản hồi NHỊ PHÂN (thành công/thất bại).
So sánh hiệu quả khi có và không có secret permutation.
"""

import os, sys, csv, math, numpy as np
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536
FEATURE_LENGTH = 832
ERROR_THRESHOLD = 80  # ngưỡng sửa lỗi của decoder (ước tính)
MAX_ATTEMPTS_PER_USER = 5000  # tổng số lần thử tối đa cho mỗi user
TRAIN_RATIO = 0.5
SEED = 12345


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


def get_selection_and_bits(handler, name, imagenum):
    emb = load_embedding(name, imagenum)
    b_full = handler._binarize_full(emb).astype(np.uint8)
    projected = np.dot(emb, handler.M_matrix)
    _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
    selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
    selection_indices.sort()
    b_selected = b_full[selection_indices]
    return selection_indices, b_selected


def estimate_population_multiset(handler, identities):
    bit_counts = np.zeros(FULL_BITS, dtype=np.float64)
    selected_counts = np.zeros(FULL_BITS, dtype=np.float64)
    for name, imagenum in identities:
        sel_idx, b_sel = get_selection_and_bits(handler, name, imagenum)
        selected_counts[sel_idx] += 1
        bit_counts[sel_idx] += b_sel
    valid = selected_counts > 0
    p_bit1 = np.divide(
        bit_counts, selected_counts, out=np.full(FULL_BITS, 0.5), where=valid
    )
    p_bit1 = np.clip(p_bit1, 1e-6, 1 - 1e-6)
    order = np.argsort(-selected_counts)
    top_positions = order[:FEATURE_LENGTH]
    return p_bit1[top_positions]


def sample_candidates(rng, prior, n_candidates):
    """Tạo n_candidates ứng viên từ prior (832,)."""
    rand_vals = rng.random((n_candidates, FEATURE_LENGTH))
    return (rand_vals < prior[None, :]).astype(np.uint8)


def binary_hillclimbing_attack(
    rng, b_true, initial_prior, max_attempts, error_threshold
):
    """
    Tấn công với phản hồi nhị phân: tạo ứng viên, kiểm tra xem có <= error_threshold không.
    Nếu không thành công, tạo ứng viên mới (không học từ phản hồi).
    Trả về số lần thử cần thiết (hoặc None nếu thất bại).
    """
    prior = initial_prior.copy()

    # Sinh tối đa max_attempts ứng viên
    batch_size = 100  # sinh từng batch để tiết kiệm bộ nhớ
    attempts = 0

    while attempts < max_attempts:
        # Sinh batch ứng viên
        n_batch = min(batch_size, max_attempts - attempts)
        candidates = sample_candidates(rng, prior, n_batch)
        distances = np.sum(candidates != b_true[None, :], axis=1)

        # Kiểm tra từng ứng viên (mô phỏng phản hồi nhị phân)
        for i in range(n_batch):
            attempts += 1
            if distances[i] <= error_threshold:
                return attempts  # Thành công

        # Nếu không thành công, điều chỉnh prior một chút để tăng đa dạng
        # (thêm nhiễu nhẹ vào prior để tránh bị kẹt)
        noise = rng.normal(0, 0.05, size=FEATURE_LENGTH)
        prior = np.clip(prior + noise, 0.05, 0.95)

    return None  # Thất bại sau max_attempts


def main():
    start_time = datetime.now()
    print(
        f"Bắt đầu thí nghiệm Hill‑Climbing Binary lúc: {start_time.strftime('%H:%M:%S')}"
    )
    print(
        f"Tham số: MAX_ATTEMPTS={MAX_ATTEMPTS_PER_USER}, ERROR_THRESHOLD={ERROR_THRESHOLD}"
    )

    rng = np.random.default_rng(SEED)
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    all_pairs = load_all_genuine_pairs()
    unique_names = []
    seen = set()
    for name_e, img_e, _, _ in all_pairs:
        if name_e not in seen:
            seen.add(name_e)
            unique_names.append((name_e, img_e))

    rng.shuffle(unique_names)
    n_train = int(len(unique_names) * TRAIN_RATIO)
    train_identities = unique_names[:n_train]
    test_identities = unique_names[n_train:]

    print(f"\nSố người dùng TRAIN: {len(train_identities)}")
    print(f"Số người dùng TEST: {len(test_identities)}")

    # Ước tính prior thật
    multiset_p = estimate_population_multiset(handler, train_identities)
    prior_random = np.full(FEATURE_LENGTH, 0.5)

    # Chạy tấn công trên tập test
    results_no_perm = []
    results_with_perm = []

    print(f"\nChạy Hill‑Climbing Binary Attack trên {len(test_identities)} user...")
    for i, (name, imagenum) in enumerate(test_identities):
        _, b_true = get_selection_and_bits(handler, name, imagenum)

        # Tấn công với prior thật (không permutation)
        rng_attack = np.random.default_rng(SEED + i)
        attempts_no_perm = binary_hillclimbing_attack(
            rng_attack, b_true, multiset_p, MAX_ATTEMPTS_PER_USER, ERROR_THRESHOLD
        )
        if attempts_no_perm is not None:
            results_no_perm.append(attempts_no_perm)

        # Tấn công với prior ngẫu nhiên (có permutation)
        rng_attack2 = np.random.default_rng(SEED + i + 100000)
        attempts_with_perm = binary_hillclimbing_attack(
            rng_attack2, b_true, prior_random, MAX_ATTEMPTS_PER_USER, ERROR_THRESHOLD
        )
        if attempts_with_perm is not None:
            results_with_perm.append(attempts_with_perm)

        if (i + 1) % 50 == 0 or (i + 1) == len(test_identities):
            elapsed = datetime.now() - start_time
            print(
                f"  [{i+1}/{len(test_identities)}] {name}: "
                f"no_perm success={len(results_no_perm)}, "
                f"with_perm success={len(results_with_perm)} "
                f"(elapsed: {elapsed})"
            )

    print("\n" + "=" * 60)
    print("KẾT QUẢ HILL‑CLIMBING BINARY ATTACK")
    print("=" * 60)
    print(f"Không permutation (prior thật):")
    print(f"  Thành công: {len(results_no_perm)}/{len(test_identities)}")
    if results_no_perm:
        print(f"  Số lần thử trung bình: {np.mean(results_no_perm):.2f}")
        print(f"  Số lần thử nhỏ nhất: {np.min(results_no_perm)}")

    print(f"\nCó permutation (prior ngẫu nhiên):")
    print(f"  Thành công: {len(results_with_perm)}/{len(test_identities)}")
    if results_with_perm:
        print(f"  Số lần thử trung bình: {np.mean(results_with_perm):.2f}")
        print(f"  Số lần thử nhỏ nhất: {np.min(results_with_perm)}")

    print("\n=== NHẬN ĐỊNH ===")
    if len(results_no_perm) == 0 and len(results_with_perm) == 0:
        print("Cả hai chiến lược đều thất bại hoàn toàn với ngân sách 5000 lần thử.")
        print("=> Hệ thống an toàn trong thực tế với ngân sách tấn công hợp lý.")
    elif len(results_no_perm) > len(results_with_perm):
        print("Tấn công hiệu quả hơn khi KHÔNG có permutation (đúng như dự đoán).")
        print("=> Secret permutation làm tăng đáng kể độ khó cho attacker.")
    else:
        print("Tấn công không hiệu quả hơn khi không có permutation.")
        print("=> Cần xem xét lại mô hình hoặc tham số thí nghiệm.")

    total_time = datetime.now() - start_time
    print(f"\nTổng thời gian chạy: {total_time}")
    handler.sess.close()


if __name__ == "__main__":
    main()
