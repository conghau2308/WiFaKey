"""
attack_hillclimbing_simulation.py

Mô phỏng tấn công leo đồi (hill‑climbing) với oracle giả lập.
So sánh hiệu quả tấn công khi có và không có secret permutation.
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
CANDIDATES_PER_ROUND = 2000  # số ứng viên mỗi vòng
TOP_K = 100  # số ứng viên tốt nhất dùng để cập nhật prior
MAX_ROUNDS = 50  # số vòng tối đa
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


def update_prior(candidates, distances, top_k):
    """Cập nhật prior dựa trên top_k ứng viên tốt nhất."""
    best_indices = np.argpartition(distances, top_k)[:top_k]
    best_candidates = candidates[best_indices]
    # Tính tần suất bit=1 trong top_k
    new_prior = np.mean(best_candidates, axis=0)
    # Làm mịn: trộn với prior cũ (tránh overfitting)
    return np.clip(new_prior, 0.01, 0.99)


def hillclimbing_attack(
    rng, b_true, initial_prior, max_rounds, candidates_per_round, top_k
):
    """Thực hiện tấn công leo đồi. Trả về số vòng cần thiết để thành công (hoặc None)."""
    prior = initial_prior.copy()
    best_dist = FEATURE_LENGTH

    for round_idx in range(max_rounds):
        candidates = sample_candidates(rng, prior, candidates_per_round)
        distances = np.sum(candidates != b_true[None, :], axis=1)
        min_dist = distances.min()

        if min_dist < best_dist:
            best_dist = min_dist

        if best_dist <= ERROR_THRESHOLD:
            return round_idx + 1, best_dist

        # Cập nhật prior từ top_k
        prior = update_prior(candidates, distances, top_k)

    return None, best_dist


def main():
    start_time = datetime.now()
    print(f"Bắt đầu thí nghiệm Hill‑Climbing lúc: {start_time.strftime('%H:%M:%S')}")
    print(
        f"Tham số: CANDIDATES_PER_ROUND={CANDIDATES_PER_ROUND}, TOP_K={TOP_K}, MAX_ROUNDS={MAX_ROUNDS}, ERROR_THRESHOLD={ERROR_THRESHOLD}"
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

    print(f"\nChạy Hill‑Climbing Attack trên {len(test_identities)} user...")
    for i, (name, imagenum) in enumerate(test_identities):
        _, b_true = get_selection_and_bits(handler, name, imagenum)

        # Tấn công với prior thật (không permutation)
        rng_attack = np.random.default_rng(SEED + i)
        rounds_no_perm, best_dist_no_perm = hillclimbing_attack(
            rng_attack, b_true, multiset_p, MAX_ROUNDS, CANDIDATES_PER_ROUND, TOP_K
        )
        if rounds_no_perm is not None:
            results_no_perm.append(rounds_no_perm)

        # Tấn công với prior ngẫu nhiên (có permutation)
        rng_attack2 = np.random.default_rng(SEED + i + 100000)
        rounds_with_perm, best_dist_with_perm = hillclimbing_attack(
            rng_attack2, b_true, prior_random, MAX_ROUNDS, CANDIDATES_PER_ROUND, TOP_K
        )
        if rounds_with_perm is not None:
            results_with_perm.append(rounds_with_perm)

        if (i + 1) % 50 == 0 or (i + 1) == len(test_identities):
            elapsed = datetime.now() - start_time
            print(
                f"  [{i+1}/{len(test_identities)}] {name}: "
                f"no_perm success={len(results_no_perm)}, "
                f"with_perm success={len(results_with_perm)} "
                f"(elapsed: {elapsed})"
            )

    print("\n" + "=" * 60)
    print("KẾT QUẢ HILL‑CLIMBING ATTACK")
    print("=" * 60)
    print(f"Không permutation (prior thật):")
    print(f"  Thành công: {len(results_no_perm)}/{len(test_identities)}")
    if results_no_perm:
        print(f"  Số vòng trung bình: {np.mean(results_no_perm):.2f}")
        print(
            f"  Tổng lần thử trung bình: {np.mean(results_no_perm) * CANDIDATES_PER_ROUND:.0f}"
        )

    print(f"\nCó permutation (prior ngẫu nhiên):")
    print(f"  Thành công: {len(results_with_perm)}/{len(test_identities)}")
    if results_with_perm:
        print(f"  Số vòng trung bình: {np.mean(results_with_perm):.2f}")
        print(
            f"  Tổng lần thử trung bình: {np.mean(results_with_perm) * CANDIDATES_PER_ROUND:.0f}"
        )

    print("\n=== NHẬN ĐỊNH ===")
    if len(results_no_perm) > len(results_with_perm):
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
