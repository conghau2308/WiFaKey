"""
attack_multisession_partial_leak.py

Mô phỏng hai mô hình tấn công nâng cao:
  1) Multi‑session Attack: Attacker quan sát nhiều phiên của CÙNG user.
  2) Partial Permutation Leak: Một phần permutation bị lộ.
"""

import os, sys, csv, math, numpy as np
from collections import defaultdict
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536
FEATURE_LENGTH = 832
ERROR_THRESHOLD = 80
MAX_ATTEMPTS = 10000
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


def binary_attack(rng, b_true, prior, max_attempts, error_threshold):
    """Tấn công với phản hồi nhị phân, trả về số lần thử hoặc None."""
    batch_size = 200
    attempts = 0
    current_prior = prior.copy()
    while attempts < max_attempts:
        n_batch = min(batch_size, max_attempts - attempts)
        candidates = (
            rng.random((n_batch, FEATURE_LENGTH)) < current_prior[None, :]
        ).astype(np.uint8)
        distances = np.sum(candidates != b_true[None, :], axis=1)
        for i in range(n_batch):
            attempts += 1
            if distances[i] <= error_threshold:
                return attempts
        # Điều chỉnh prior nhẹ để tăng đa dạng
        noise = rng.normal(0, 0.03, size=FEATURE_LENGTH)
        current_prior = np.clip(current_prior + noise, 0.05, 0.95)
    return None


# ========================
# MÔ HÌNH 2: MULTI‑SESSION
# ========================
def simulate_multisession_attack(handler, test_identities, rng):
    """Với mỗi user, dùng 3 ảnh khác để ước tính prior riêng, rồi tấn công."""
    # Tìm người dùng có >= 4 ảnh trong tập test
    user_images = defaultdict(list)
    for name, imagenum in test_identities:
        user_images[name].append(imagenum)

    multisession_users = {
        name: imgs for name, imgs in user_images.items() if len(imgs) >= 4
    }
    print(f"Số user đủ điều kiện multi‑session (>=4 ảnh): {len(multisession_users)}")

    results = {"population_prior": [], "personalized_prior": []}

    for name, imagelist in multisession_users.items():
        # Dùng 3 ảnh đầu để ước tính prior riêng
        train_imgs = imagelist[:3]
        test_img = imagelist[3]

        # Ước tính prior riêng
        bit_counts = np.zeros(FULL_BITS, dtype=int)
        selected_counts = np.zeros(FULL_BITS, dtype=int)
        for img_num in train_imgs:
            sel_idx, b_sel = get_selection_and_bits(handler, name, img_num)
            selected_counts[sel_idx] += 1
            bit_counts[sel_idx] += b_sel
        valid = selected_counts > 0
        personalized_prior_full = np.divide(
            bit_counts, selected_counts, out=np.full(FULL_BITS, 0.5), where=valid
        )
        personalized_prior_full = np.clip(personalized_prior_full, 1e-6, 1 - 1e-6)
        order = np.argsort(-selected_counts)
        top_positions = order[:FEATURE_LENGTH]
        personalized_prior = personalized_prior_full[top_positions]

        # Prior toàn cục (population) – đã có sẵn từ hàm khác, truyền vào sau
        # Tạm thời dùng prior=0.5 cho cả hai, sẽ thay bằng prior thật trong main

        # Lấy b_true cho ảnh test
        _, b_true = get_selection_and_bits(handler, name, test_img)

        # Tấn công với prior riêng
        rng_attack = np.random.default_rng(SEED + hash(name) % 10000)
        attempts_personalized = binary_attack(
            rng_attack, b_true, personalized_prior, MAX_ATTEMPTS, ERROR_THRESHOLD
        )
        results["personalized_prior"].append(attempts_personalized)

    return results


# ================================
# MÔ HÌNH 3: PARTIAL PERMUTATION LEAK
# ================================
def simulate_partial_leak(handler, test_identities, multiset_p, rng):
    """Mô phỏng tấn công khi một phần permutation bị lộ."""
    leak_fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    results = {f"leak_{int(f*100)}pct": [] for f in leak_fractions}

    for name, imagenum in test_identities:
        _, b_true = get_selection_and_bits(handler, name, imagenum)

        for leak_frac in leak_fractions:
            # Tạo prior: kết hợp giữa prior thật (cho phần bị lộ) và 0.5 (cho phần không lộ)
            n_leaked = int(FEATURE_LENGTH * leak_frac)
            # Chọn ngẫu nhiên vị trí bị lộ
            leaked_positions = rng.choice(FEATURE_LENGTH, size=n_leaked, replace=False)
            prior = np.full(FEATURE_LENGTH, 0.5)
            prior[leaked_positions] = multiset_p[leaked_positions]

            rng_attack = np.random.default_rng(
                SEED + hash(name) % 10000 + int(leak_frac * 1000)
            )
            attempts = binary_attack(
                rng_attack, b_true, prior, MAX_ATTEMPTS, ERROR_THRESHOLD
            )
            results[f"leak_{int(leak_frac*100)}pct"].append(attempts)

    return results


def main():
    start_time = datetime.now()
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

    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_names)
    n_train = int(len(unique_names) * 0.5)
    train_identities = unique_names[:n_train]
    test_identities = unique_names[n_train:]

    # Ước tính multiset toàn cục (dùng cho cả hai mô hình)
    # (Hàm estimate_population_multiset đã có trong các script trước, ta gọi lại)
    from research.quantizer.v1_lssc_with_perbit_confidence import (
        binarize_with_perbit_confidence,
    )

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

    multiset_p = estimate_population_multiset(handler, train_identities)

    # ---- Mô hình 2: Multi‑session ----
    print("=" * 60)
    print("MÔ HÌNH 2: MULTI‑SESSION ATTACK")
    print("=" * 60)
    ms_results = simulate_multisession_attack(handler, test_identities, rng)
    for key, attempts_list in ms_results.items():
        success = [a for a in attempts_list if a is not None]
        print(f"\n{key}:")
        print(f"  Thành công: {len(success)}/{len(attempts_list)}")
        if success:
            print(f"  Số lần thử TB: {np.mean(success):.2f}")

    # ---- Mô hình 3: Partial Leak ----
    print("\n" + "=" * 60)
    print("MÔ HÌNH 3: PARTIAL PERMUTATION LEAK")
    print("=" * 60)
    leak_results = simulate_partial_leak(handler, test_identities, multiset_p, rng)
    for key, attempts_list in leak_results.items():
        success = [a for a in attempts_list if a is not None]
        print(f"\n{key}:")
        print(f"  Thành công: {len(success)}/{len(attempts_list)}")
        if success:
            print(f"  Số lần thử TB: {np.mean(success):.2f}")

    total_time = datetime.now() - start_time
    print(f"\nTổng thời gian: {total_time}")
    handler.sess.close()


if __name__ == "__main__":
    main()
