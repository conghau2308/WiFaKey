"""
attack_composition_vs_naive_full.py

Mở rộng thí nghiệm so sánh hai chiến lược tấn công với cỡ mẫu LỚN.
Dùng toàn bộ người dùng trong tập test (khoảng 200-250 người).
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

N_TRIALS_PER_USER = 50000  # tăng lên để best-of-N chính xác hơn
TRAIN_RATIO = 0.5  # dùng 50% người dùng để ước tính multiset
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


def batch_naive_guesses(rng, n_bits, n_trials):
    return rng.integers(0, 2, size=(n_trials, n_bits)).astype(np.uint8)


def batch_composition_guesses(rng, multiset_p, n_trials):
    n_bits = len(multiset_p)
    random_keys = rng.random((n_trials, n_bits))
    perm_idx = np.argsort(random_keys, axis=1)
    shuffled_p = multiset_p[perm_idx]
    rand_vals = rng.random((n_trials, n_bits))
    return (rand_vals < shuffled_p).astype(np.uint8)


def binomial_sign_test(k, n, p=0.5):
    if n == 0:
        return 1.0

    def pmf(x):
        return math.comb(n, x) * (p**x) * ((1 - p) ** (n - x))

    obs_p = pmf(k)
    total = sum(pmf(x) for x in range(n + 1) if pmf(x) <= obs_p * 1.0000001)
    return min(total, 1.0)


def main():
    start_time = datetime.now()
    print(f"Bắt đầu thí nghiệm lúc: {start_time.strftime('%H:%M:%S')}")
    print(
        f"Tham số: N_TRIALS={N_TRIALS_PER_USER}, TRAIN_RATIO={TRAIN_RATIO}, SEED={SEED}"
    )

    rng = np.random.default_rng(SEED)
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    all_pairs = load_all_genuine_pairs()

    # --- Chia TRAIN / TEST theo NGƯỜI DÙNG ---
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

    print(f"\nSố người dùng TRAIN (ước tính multiset): {len(train_identities)}")
    print(f"Số người dùng TEST (bị 'tấn công'): {len(test_identities)}")

    print("\nĐang ước tính multiset {p_i} từ tập TRAIN...")
    multiset_p = estimate_population_multiset(handler, train_identities)
    print(f"  mean(p) = {multiset_p.mean():.4f}, std(p) = {multiset_p.std():.4f}")
    print(
        f"  Số vị trí p>0.9: {np.sum(multiset_p > 0.9)}, p<0.1: {np.sum(multiset_p < 0.1)}"
    )

    theoretical_mean_mismatch = FEATURE_LENGTH * 0.5
    print(
        f"\n(Kỳ vọng lý thuyết mismatch trung bình: {theoretical_mean_mismatch:.0f}/{FEATURE_LENGTH})"
    )

    results_min_A, results_min_B = [], []
    results_mean_A, results_mean_B = [], []

    print(
        f"\nChạy tấn công trên {len(test_identities)} user, "
        f"{N_TRIALS_PER_USER} lần thử/strategy/user..."
    )
    for i, (name, imagenum) in enumerate(test_identities):
        _, b_true = get_selection_and_bits(handler, name, imagenum)

        guesses_A = batch_naive_guesses(rng, FEATURE_LENGTH, N_TRIALS_PER_USER)
        guesses_B = batch_composition_guesses(rng, multiset_p, N_TRIALS_PER_USER)

        dists_A = np.sum(guesses_A != b_true[None, :], axis=1)
        dists_B = np.sum(guesses_B != b_true[None, :], axis=1)

        results_min_A.append(dists_A.min())
        results_min_B.append(dists_B.min())
        results_mean_A.append(dists_A.mean())
        results_mean_B.append(dists_B.mean())

        if (i + 1) % 50 == 0 or (i + 1) == len(test_identities):
            elapsed = datetime.now() - start_time
            print(
                f"  [{i+1}/{len(test_identities)}] {name}: "
                f"min A={np.mean(results_min_A):.0f}, min B={np.mean(results_min_B):.0f} "
                f"(elapsed: {elapsed})"
            )

    results_min_A = np.array(results_min_A)
    results_min_B = np.array(results_min_B)
    mean_A = np.array(results_mean_A)
    mean_B = np.array(results_mean_B)

    print("\n" + "=" * 60)
    print("KẾT QUẢ TỔNG HỢP")
    print("=" * 60)
    print(
        f"Mismatch trung bình MỖI LẦN đoán (kỳ vọng {theoretical_mean_mismatch:.0f}):"
    )
    print(f"  Strategy A (naive):       {mean_A.mean():.2f}")
    print(f"  Strategy B (composition): {mean_B.mean():.2f}")

    print(
        f"\nMismatch NHỎ NHẤT tìm được (best-of-{N_TRIALS_PER_USER}), "
        f"trung bình qua {len(test_identities)} user:"
    )
    print(f"  Strategy A: {results_min_A.mean():.2f} (std {results_min_A.std():.2f})")
    print(f"  Strategy B: {results_min_B.mean():.2f} (std {results_min_B.std():.2f})")

    b_better = int(np.sum(results_min_B < results_min_A))
    a_better = int(np.sum(results_min_A < results_min_B))
    ties = len(test_identities) - b_better - a_better
    print(f"\nSo sánh ghép cặp theo từng user:")
    print(
        f"  Strategy B thắng (best-of-N mismatch thấp hơn): {b_better}/{len(test_identities)}"
    )
    print(f"  Strategy A thắng: {a_better}/{len(test_identities)}")
    print(f"  Hoà: {ties}/{len(test_identities)}")

    pvalue = None
    if b_better + a_better > 0:
        pvalue = binomial_sign_test(b_better, b_better + a_better, p=0.5)
        print(f"  Sign test hai phía (H0: B không tốt hơn A): p-value = {pvalue:.4f}")

    print("\n=== NHẬN ĐỊNH ===")
    diff = results_min_A.mean() - results_min_B.mean()
    if pvalue is not None and pvalue < 0.05 and diff > 0:
        print(
            f"CẢNH BÁO: Strategy B tốt hơn Strategy A có ý nghĩa thống kê "
            f"(chênh lệch {diff:.1f} bit, p={pvalue:.4f})."
        )
        print("=> Secret permutation KHÔNG đủ để trung hòa hoàn toàn multiset.")
    else:
        print("Không có bằng chứng thống kê cho thấy Strategy B tốt hơn Strategy A.")
        print(
            "=> Secret permutation dường như trung hòa hiệu quả lợi thế của multiset."
        )
        print("   (Trong mô hình tấn công không thích ứng, không oracle.)")

    total_time = datetime.now() - start_time
    print(f"\nTổng thời gian chạy: {total_time}")
    handler.sess.close()


if __name__ == "__main__":
    main()
