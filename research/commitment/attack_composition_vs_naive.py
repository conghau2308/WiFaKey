"""
attack_composition_vs_naive.py

Kiểm định thực nghiệm cho tuyên bố "secret permutation loại bỏ hoàn toàn
rò rỉ entropy" của prove_permutation_effectiveness.py.

Ý tưởng: attacker biết PHÂN BỐ TỔNG THỂ (multiset) các xác suất thiên lệch
P(bit=1 | được chọn) — thứ hoàn toàn công khai vì suy ra được từ M_matrix +
intervals (theo nguyên lý Kerckhoffs) — nhưng KHÔNG biết ánh xạ vị trí công
bố nào ứng với xác suất nào (bị secret permutation che giấu).

So sánh hai chiến lược đoán trên CÙNG một tập test:
  A) Naive      : mỗi bit ~ Bernoulli(0.5), độc lập
  B) Composition: xáo trộn ngẫu nhiên multiset {p_i} rồi gán cho 832 vị trí,
                   mỗi bit ~ Bernoulli(p đã gán) — mô phỏng attacker "biết
                   hình dạng phân bố nhưng không biết nhãn"

Về mặt kỳ vọng từng bit, cả hai chiến lược có xác suất đúng = 0.5 như nhau
(có thể chứng minh bằng đại số). Điều script này thực sự đo là: liệu chiến
lược B có tạo ra PHÂN BỐ SỐ BIT SAI với đuôi (tail) thuận lợi hơn cho attacker
hay không, tức là trong N lần thử, chiến lược nào tìm được ứng viên GẦN đúng
hơn (ít bit sai hơn) — đây mới là đại lượng liên quan trực tiếp đến khả năng
lọt vào bán kính sửa lỗi của decoder.

Tách TRAIN (ước tính multiset) / TEST (bị "tấn công") theo NGƯỜI DÙNG, không
chỉ theo cặp, để tránh rò rỉ dữ liệu giữa hai tập.
"""

import os
import sys
import csv
import math
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536
FEATURE_LENGTH = 832

N_TRIALS_PER_USER = 20000  # số lần đoán mỗi strategy, mỗi user
N_TEST_USERS = 30  # số user dùng để đánh giá tấn công
SEED = 12345


# ---------- Data loading (giống các script trước, giữ nguyên convention) ----------


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
    """Trả về (selection_indices [832], b_selected [832]) — chỉ số TUYỆT ĐỐI
    trong không gian 1536 chiều (đúng cách, giống diagnose_mask_entropy_v3.py,
    KHÔNG dùng chỉ số tương đối như bug cũ trong simulate_attack.py)."""
    emb = load_embedding(name, imagenum)
    b_full = handler._binarize_full(emb).astype(np.uint8)
    projected = np.dot(emb, handler.M_matrix)
    _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
    selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
    selection_indices.sort()
    b_selected = b_full[selection_indices]
    return selection_indices, b_selected


def estimate_population_multiset(handler, identities):
    """Ước tính multiset {p_i} (832 giá trị) từ tập TRAIN, dùng chỉ số tuyệt đối."""
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

    # Multiset đại diện: p tại 832 vị trí được chọn thường xuyên nhất trong
    # tập train — mô phỏng đúng "hình dạng" mà một lần chọn 832/1536 thực tế
    # sẽ tạo ra.
    order = np.argsort(-selected_counts)
    top_positions = order[:FEATURE_LENGTH]
    return p_bit1[top_positions]


# ---------- Hai chiến lược đoán, vector hoá để chạy nhanh ----------


def batch_naive_guesses(rng, n_bits, n_trials):
    return rng.integers(0, 2, size=(n_trials, n_bits)).astype(np.uint8)


def batch_composition_guesses(rng, multiset_p, n_trials):
    """Với mỗi trial, xáo trộn multiset_p độc lập rồi lấy mẫu Bernoulli.
    Dùng thủ thuật argsort-of-random-keys để vector hoá việc xáo trộn theo hàng."""
    n_bits = len(multiset_p)
    random_keys = rng.random((n_trials, n_bits))
    perm_idx = np.argsort(random_keys, axis=1)
    shuffled_p = multiset_p[perm_idx]  # (n_trials, n_bits)
    rand_vals = rng.random((n_trials, n_bits))
    return (rand_vals < shuffled_p).astype(np.uint8)


# ---------- Sign test thủ công (không cần scipy) ----------


def binomial_sign_test(k, n, p=0.5):
    """P-value hai phía cho sign test: k thành công trong n phép thử, H0: p=0.5."""
    if n == 0:
        return 1.0

    def pmf(x):
        return math.comb(n, x) * (p**x) * ((1 - p) ** (n - x))

    obs_p = pmf(k)
    total = sum(pmf(x) for x in range(n + 1) if pmf(x) <= obs_p * 1.0000001)
    return min(total, 1.0)


def main():
    rng = np.random.default_rng(SEED)
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    all_pairs = load_all_genuine_pairs()

    # --- Chia TRAIN / TEST theo NGƯỜI DÙNG (không chồng lấn) ---
    unique_names = []
    seen = set()
    for name_e, img_e, _, _ in all_pairs:
        if name_e not in seen:
            seen.add(name_e)
            unique_names.append((name_e, img_e))

    rng.shuffle(unique_names)
    n_train = len(unique_names) - N_TEST_USERS
    train_identities = unique_names[:n_train]
    test_identities = unique_names[n_train : n_train + N_TEST_USERS]

    print(f"Số người dùng TRAIN (ước tính multiset): {len(train_identities)}")
    print(f"Số người dùng TEST (bị 'tấn công'): {len(test_identities)}")

    print("\nĐang ước tính multiset {p_i} từ tập TRAIN...")
    multiset_p = estimate_population_multiset(handler, train_identities)
    print(f"  mean(p) = {multiset_p.mean():.4f}, std(p) = {multiset_p.std():.4f}")
    print(
        f"  Số vị trí p>0.9: {np.sum(multiset_p > 0.9)}, p<0.1: {np.sum(multiset_p < 0.1)}"
    )

    theoretical_mean_mismatch = FEATURE_LENGTH * 0.5
    print(
        f"\n(Kỳ vọng lý thuyết nếu cả 2 strategy đều có P(đúng mỗi bit)=0.5: "
        f"mismatch trung bình mỗi lần đoán ≈ {theoretical_mean_mismatch:.0f}/{FEATURE_LENGTH})"
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

        print(
            f"  [{i+1}/{len(test_identities)}] {name}: "
            f"min A={dists_A.min():4d} (mean {dists_A.mean():.1f}) | "
            f"min B={dists_B.min():4d} (mean {dists_B.mean():.1f})"
        )

    results_min_A = np.array(results_min_A)
    results_min_B = np.array(results_min_B)
    mean_A = np.array(results_mean_A)
    mean_B = np.array(results_mean_B)

    print("\n=== KẾT QUẢ TỔNG HỢP ===")
    print(
        f"Mismatch trung bình MỖI LẦN đoán (kỳ vọng lý thuyết {theoretical_mean_mismatch:.0f}):"
    )
    print(f"  Strategy A (naive):       {mean_A.mean():.2f}")
    print(f"  Strategy B (composition): {mean_B.mean():.2f}")
    print(
        f"  => nếu hai số này gần bằng nhau và gần {theoretical_mean_mismatch:.0f}, "
        f"khớp với suy luận đại số ở trên (không có gì bất thường ở đây)."
    )

    print(
        f"\nMismatch NHỎ NHẤT tìm được (best-of-{N_TRIALS_PER_USER}), "
        f"trung bình qua {N_TEST_USERS} user — đây mới là số liên quan tới rủi ro thật:"
    )
    print(f"  Strategy A: {results_min_A.mean():.2f} (std {results_min_A.std():.2f})")
    print(f"  Strategy B: {results_min_B.mean():.2f} (std {results_min_B.std():.2f})")

    b_better = int(np.sum(results_min_B < results_min_A))
    a_better = int(np.sum(results_min_A < results_min_B))
    ties = N_TEST_USERS - b_better - a_better
    print(f"\nSo sánh ghép cặp theo từng user:")
    print(
        f"  Strategy B thắng (best-of-N mismatch thấp hơn): {b_better}/{N_TEST_USERS}"
    )
    print(f"  Strategy A thắng: {a_better}/{N_TEST_USERS}")
    print(f"  Hoà: {ties}/{N_TEST_USERS}")

    pvalue = None
    if b_better + a_better > 0:
        pvalue = binomial_sign_test(b_better, b_better + a_better, p=0.5)
        print(f"  Sign test hai phía (H0: B không tốt hơn A): p-value = {pvalue:.4f}")

    print("\n=== NHẬN ĐỊNH ===")
    diff = results_min_A.mean() - results_min_B.mean()
    if pvalue is not None and pvalue < 0.05 and diff > 0:
        print(
            f"CẢNH BÁO: Strategy B tìm được ứng viên gần đúng hơn Strategy A một cách"
            f" có ý nghĩa thống kê (chênh lệch best-of-N trung bình {diff:.1f} bit,"
            f" p={pvalue:.4f})."
        )
        print(
            "=> Multiset công khai VẪN cho attacker lợi thế dù không biết permutation"
        )
        print("   cụ thể. Tuyên bố '0% rò rỉ' của prove_permutation_effectiveness.py")
        print("   CHƯA chính xác — cần xử lý thêm, không chỉ dựa vào permutation.")
    else:
        print(
            "Không tìm thấy bằng chứng thống kê cho thấy Strategy B tốt hơn Strategy A"
        )
        print("trong best-of-N. Trong MÔ HÌNH TẤN CÔNG một-lần/nhiều-lần-độc-lập,")
        print(
            "KHÔNG thích ứng (không dùng oracle để tinh chỉnh dần), không có quan sát"
        )
        print("lặp lại cùng một permutation qua nhiều phiên — secret permutation dường")
        print("như thực sự trung hoà được lợi thế của multiset công khai.")
        print("QUAN TRỌNG — điều này KHÔNG chứng minh an toàn tuyệt đối. Các mô hình")
        print("tấn công mạnh hơn CHƯA được kiểm tra ở đây:")
        print("  1) Attacker có quyền truy cập soft-output của decoder (không chỉ")
        print("     True/False cuối) để làm hill-climbing/leo đồi.")
        print(
            "  2) Attacker quan sát NHIỀU phiên xác thực của CÙNG một user (permutation"
        )
        print("     cố định, tái sử dụng) — có thể học được thống kê riêng cho user đó")
        print("     qua thời gian, không cần multiset toàn cục.")
        print("  3) Permutation bị lộ/đoán được do cách lưu trữ/khôi phục yếu.")

    handler.sess.close()


if __name__ == "__main__":
    main()
