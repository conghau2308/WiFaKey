"""
diagnose_cdf_normalization.py

Tier 1 – CDF Normalization: biến đổi giá trị chiếu qua CDF thực nghiệm
để cân bằng phân phối bit, giữ nguyên ngưỡng gốc.
"""

import os, sys, csv, hashlib, numpy as np
from scipy.interpolate import interp1d

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
N_DIMS = 512


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


def estimate_cdfs(handler, pairs):
    """Ước tính CDF thực nghiệm cho từng chiều từ tập train."""
    all_projected = []
    for name_e, img_e, _, _ in pairs:
        emb = load_embedding(name_e, img_e)
        all_projected.append(np.dot(emb, handler.M_matrix))
    all_projected = np.array(all_projected)  # (N, 512)

    cdfs = []
    for d in range(N_DIMS):
        vals = np.sort(all_projected[:, d])
        # Tạo hàm CDF nội suy tuyến tính
        cdf = interp1d(
            vals, np.linspace(0, 1, len(vals)), bounds_error=False, fill_value=(0, 1)
        )
        cdfs.append(cdf)
    return cdfs


def apply_cdf_transform(projected, cdfs):
    """Áp dụng CDF transform lên vector projected (512,)."""
    u = np.zeros(N_DIMS)
    for d in range(N_DIMS):
        u[d] = cdfs[d]([projected[d]])[0]
    return u


def binarize_with_cdf(projected, cdfs, intervals):
    """Binarize sau khi CDF transform, dùng ngưỡng gốc."""
    u = apply_cdf_transform(projected, cdfs)
    # Ngưỡng gốc (đã sort)
    thresholds = np.sort(intervals)
    bits = np.zeros(FULL_BITS, dtype=np.uint8)
    margin = np.zeros(FULL_BITS, dtype=np.float32)
    for d in range(N_DIMS):
        thr = thresholds[::-1]  # Giảm dần
        for t_idx in range(3):
            pos = d * 3 + t_idx
            bits[pos] = u[d] >= thr[t_idx]
            margin[pos] = abs(u[d] - thr[t_idx])
    return bits, margin


def run_test(handler, test_pairs, cdfs, emp_mod):
    """Đánh giá GMR với CDF normalization."""
    pass_count = 0
    for name_e, img_e, name_v, img_v in test_pairs:
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)

        # Enrollment với CDF
        proj_e = np.dot(emb_e, handler.M_matrix)
        bits_e, margin_e = binarize_with_cdf(proj_e, cdfs, handler.intervals)
        selection_indices = np.argpartition(-margin_e, FEATURE_LENGTH)[:FEATURE_LENGTH]
        selection_indices.sort()
        b_sel_e = bits_e[selection_indices]

        rng = np.random.default_rng()
        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper = np.logical_xor(b_sel_e, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        # Verify với CDF
        proj_v = np.dot(emb_v, handler.M_matrix)
        bits_v, margin_v = binarize_with_cdf(proj_v, cdfs, handler.intervals)
        b_sel_v = bits_v[selection_indices]
        noisy = np.logical_xor(b_sel_v, helper).astype(np.uint8)
        margin_sel = margin_v[selection_indices]
        llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        llr = llr.reshape(1, N, Z)
        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            pass_count += 1
    return pass_count


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    emp_mod = EmpiricalLLR(
        lookup_path=os.path.join(
            _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
        )
    )

    # Chia train/test
    all_pairs = load_genuine_pairs()
    n_train = int(len(all_pairs) * 0.7)
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train : n_train + 200]

    # Ước tính CDFs
    print("1. Ước tính CDF thực nghiệm cho từng chiều...")
    cdfs = estimate_cdfs(handler, train_pairs)

    # Đo entropy trước CDF
    print("2. Đo entropy bit trước CDF...")
    # (Dùng code từ diagnose_mask_entropy_v3.py)
    # ...

    # Đánh giá GMR
    print("3. Đánh giá GMR với CDF normalization...")
    pass_cdf = run_test(handler, test_pairs, cdfs, emp_mod)
    print(
        f"\nGMR với CDF normalization: {pass_cdf}/{len(test_pairs)} ({100*pass_cdf/len(test_pairs):.2f}%)"
    )

    # So sánh với baseline (không CDF)
    pass_baseline = run_test(
        handler, test_pairs, None, emp_mod
    )  # Sẽ cần sửa run_test để dùng pipeline gốc
    # Tạm thời in kết quả CDF trước

    handler.sess.close()


if __name__ == "__main__":
    main()
