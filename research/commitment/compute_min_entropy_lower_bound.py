"""
compute_min_entropy_lower_bound.py

Uoc tinh CAN DUOI co the chung minh (khong phai diem uoc luong) cho
min-entropy cua reliability mask, dung Clopper-Pearson CI + hieu chinh
Bonferroni cho 1536 vi tri dong thoi.
"""

import os, sys, csv
import numpy as np
from scipy import stats

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

FULL_BITS = 1536
FEATURE_LENGTH = 832
ALPHA_GLOBAL = 0.05  # muon dung 95% dong thoi cho toan bo cac vi tri
EPS_LEFTOVER_HASH = 2**-40  # khop epsilon da dung o Tier 3 truoc do


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


def load_all_identities():
    pairs_csv = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "pairs",
        "tune_genuine.csv",
    )
    identities = {}
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name_enroll"]
            if name not in identities:
                identities[name] = int(row["imagenum_enroll"])
    return list(identities.items())


def get_selection_and_bits(handler, name, imagenum):
    """Chi so TUYET DOI trong khong gian 1536 chieu (dung cach, khop
    diagnose_mask_entropy_v3.py, KHONG dung chi so tuong doi)."""
    emb = load_embedding(name, imagenum)
    b_full = handler._binarize_full(emb).astype(np.uint8)
    projected = np.dot(emb, handler.M_matrix)
    _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
    selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
    selection_indices.sort()
    return selection_indices, b_full[selection_indices]


def clopper_pearson_upper_bound(successes, n, alpha):
    """Can tren mot phia (1-alpha) cho xac suat nhi thuc that, chinh xac
    (khong xap xi), dua tren quan he Beta-Binomial."""
    if n == 0:
        return 1.0
    if successes >= n:
        return 1.0
    return stats.beta.ppf(1 - alpha, successes + 1, n - successes)


def per_position_min_entropy_bound(bit_count, selected_count, alpha):
    """Can duoi cua min-entropy 1 vi tri = -log2(can tren cua max(p,1-p))."""
    if selected_count == 0:
        return 1.0  # khong co du lieu quan sat => khong the ket luan gi, mac dinh an toan nhat
    p1_upper = clopper_pearson_upper_bound(bit_count, selected_count, alpha)
    p0_upper = clopper_pearson_upper_bound(
        selected_count - bit_count, selected_count, alpha
    )
    max_prob_upper_bound = min(max(p1_upper, p0_upper), 1.0 - 1e-15)
    return -np.log2(max_prob_upper_bound)


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    identities = load_all_identities()
    n_users = len(identities)
    print(f"So nguoi dung: {n_users}")

    bit_counts = np.zeros(FULL_BITS, dtype=np.int64)
    selected_counts = np.zeros(FULL_BITS, dtype=np.int64)
    user_selections = {}

    for name, imagenum in identities:
        sel_idx, b_sel = get_selection_and_bits(handler, name, imagenum)
        selected_counts[sel_idx] += 1
        bit_counts[sel_idx] += b_sel
        user_selections[name] = (sel_idx, b_sel)

    # --- Uoc luong diem (kieu cu) de doi chieu ---
    valid = selected_counts > 0
    p_point = np.divide(
        bit_counts, selected_counts, out=np.full(FULL_BITS, 0.5), where=valid
    )
    p_point = np.clip(p_point, 1e-12, 1 - 1e-12)
    entropy_point = -(p_point * np.log2(p_point) + (1 - p_point) * np.log2(1 - p_point))

    # --- Can duoi Clopper-Pearson, hieu chinh Bonferroni ---
    n_positions_tested = int(np.sum(valid))
    alpha_per_position = ALPHA_GLOBAL / n_positions_tested
    print(f"\nSo vi tri co du lieu: {n_positions_tested}/{FULL_BITS}")
    print(f"Alpha toan cuc mong muon: {ALPHA_GLOBAL}")
    print(f"Alpha moi vi tri sau Bonferroni: {alpha_per_position:.3e}")

    min_entropy_bound = np.ones(FULL_BITS, dtype=np.float64)
    for i in range(FULL_BITS):
        if selected_counts[i] > 0:
            min_entropy_bound[i] = per_position_min_entropy_bound(
                int(bit_counts[i]), int(selected_counts[i]), alpha_per_position
            )

    sel_counts_valid = selected_counts[valid]
    print(
        f"\nSo lan duoc chon moi vi tri - min: {sel_counts_valid.min()}, "
        f"max: {sel_counts_valid.max()}, median: {np.median(sel_counts_valid):.0f}"
    )
    print(
        "(N nho => CI rong => can duoi bi keo ve gan 0 bit nhieu hon - day la hanh vi"
    )
    print(" DUNG va MONG MUON cua mot can duoi bao thu, khong phai loi.)")

    print("\n=== SO SANH DIEM vs CAN DUOI (trung binh moi vi tri) ===")
    print(
        f"Entropy trung binh (DIEM, khong dung de chung minh an toan): {np.mean(entropy_point[valid]):.4f} bit"
    )
    print(
        f"Entropy trung binh (CAN DUOI Clopper-Pearson+Bonferroni):    {np.mean(min_entropy_bound[valid]):.4f} bit"
    )

    print(f"\n=== MIN-ENTROPY TONG (832 bit) THEO TUNG NGUOI DUNG ===")
    per_user_point, per_user_bound = [], []
    for name, (sel_idx, b_sel) in user_selections.items():
        per_user_point.append(np.sum(entropy_point[sel_idx]))
        per_user_bound.append(np.sum(min_entropy_bound[sel_idx]))
    per_user_point = np.array(per_user_point)
    per_user_bound = np.array(per_user_bound)

    print(
        f"Tong entropy DIEM      - mean: {per_user_point.mean():.2f}, min: {per_user_point.min():.2f}, max: {per_user_point.max():.2f}"
    )
    print(
        f"Tong min-entropy CAN DUOI - mean: {per_user_bound.mean():.2f}, min: {per_user_bound.min():.2f}, max: {per_user_bound.max():.2f}"
    )

    log2_1_over_eps = -np.log2(EPS_LEFTOVER_HASH)
    worst_m = per_user_bound.min()
    mean_m = per_user_bound.mean()
    safe_len_worst = max(0, int(np.floor(worst_m - 2 * log2_1_over_eps)))
    safe_len_mean = max(0, int(np.floor(mean_m - 2 * log2_1_over_eps)))

    print(f"\n=== DO DAI KHOA AN TOAN (leftover hash lemma, epsilon=2^-40) ===")
    print(f"Hao phi 2*log2(1/eps) = {2*log2_1_over_eps:.0f} bit")
    print(
        f"Dung m = TRUNG BINH qua user ({mean_m:.2f} bit) => khoa an toan: {safe_len_mean} bit"
    )
    print(
        f"Dung m = XAU NHAT qua user   ({worst_m:.2f} bit) => khoa an toan: {safe_len_worst} bit"
    )
    print(
        "\nLUU Y: nen dung con so XAU NHAT cho mot do dai khoa CO DINH ap dung cho MOI"
    )
    print(
        "nguoi dung - dung trung binh se khien nhung user duoi mean co khoa yeu hon cong bo."
    )

    print("\n=== SO SANH VOI TIER 3 TRUOC DO (190.11 bit / khoa 110-bit) ===")
    print("Neu can duoi o day THAP HON dang ke 190.11 bit: con so truoc la uoc luong")
    print("diem lac quan hon thuc te co the chung minh - nen thu nho khoa theo so nay.")
    print("Neu hai con so gan nhau: uoc luong truoc do tinh co da kha bao thu.")

    handler.sess.close()


if __name__ == "__main__":
    main()
