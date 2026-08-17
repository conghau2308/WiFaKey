"""
diagnose_denylist.py

Thử nghiệm Denylist: loại bỏ vị trí có P(bit=1) lệch khỏi 0.5.
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


def margin_selection_with_denylist(margin, p_bit1, denylist_threshold):
    """
    Chọn 832 vị trí có margin cao nhất, nhưng loại bỏ các vị trí có |p - 0.5| > threshold.
    Nếu không đủ 832, lấy thêm từ các vị trí không bị cấm (kể cả margin thấp hơn).
    """
    # Sắp xếp tất cả vị trí theo margin giảm dần
    sorted_by_margin = np.argsort(-margin)
    selected = []
    for pos in sorted_by_margin:
        if abs(p_bit1[pos] - 0.5) <= denylist_threshold:
            selected.append(pos)
        if len(selected) == FEATURE_LENGTH:
            break
    if len(selected) < FEATURE_LENGTH:
        # Nếu vẫn thiếu, lấy thêm từ các vị trí bị cấm (ưu tiên margin cao)
        for pos in sorted_by_margin:
            if pos not in selected:
                selected.append(pos)
            if len(selected) == FEATURE_LENGTH:
                break
    return np.array(sorted(selected))


def run_test(handler, test_pairs, p_bit1, denylist_threshold, emp_mod):
    """Đánh giá GMR với denylist."""
    pass_count = 0
    for name_e, img_e, name_v, img_v in test_pairs:
        emb_e = load_embedding(name_e, img_e)
        emb_v = load_embedding(name_v, img_v)
        proj_e = np.dot(emb_e, handler.M_matrix)
        _, margin_e = binarize_with_perbit_confidence(proj_e, handler.intervals)
        proj_v = np.dot(emb_v, handler.M_matrix)
        bits_v, margin_v = binarize_with_perbit_confidence(proj_v, handler.intervals)

        # Chọn vị trí theo margin + denylist
        sel_idx = margin_selection_with_denylist(margin_e, p_bit1, denylist_threshold)
        b_sel_e = handler._binarize_full(emb_e).astype(np.uint8)[sel_idx]

        rng = np.random.default_rng()
        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper = np.logical_xor(b_sel_e, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        # Verify
        b_sel_v = bits_v[sel_idx]
        noisy = np.logical_xor(b_sel_v, helper).astype(np.uint8)
        margin_sel = margin_v[sel_idx]
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
    test_pairs = all_pairs[n_train : n_train + 200]  # 200 cặp test

    # Tính p_bit1 từ train
    p_bit1 = estimate_p_bit1(handler, train_pairs)
    entropy_before = -(p_bit1 * np.log2(p_bit1) + (1 - p_bit1) * np.log2(1 - p_bit1))
    print(f"Entropy bit trung bình (trước denylist): {np.mean(entropy_before):.4f}")

    # Baseline (không denylist)
    pass_baseline = run_test(
        handler, test_pairs, p_bit1, denylist_threshold=1.0, emp_mod=emp_mod
    )
    print(
        f"\nBaseline GMR (không denylist): {pass_baseline}/{len(test_pairs)} ({100*pass_baseline/len(test_pairs):.2f}%)"
    )

    # Thử các ngưỡng denylist
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    for thresh in thresholds:
        # Đo entropy sau denylist (trên tập vị trí được chọn thực tế)
        # Lấy mẫu một user để xem
        sample_emb = load_embedding(test_pairs[0][0], test_pairs[0][1])
        proj = np.dot(sample_emb, handler.M_matrix)
        _, margin = binarize_with_perbit_confidence(proj, handler.intervals)
        sel = margin_selection_with_denylist(margin, p_bit1, thresh)
        p_sel = p_bit1[sel]
        entropy_sel = -(p_sel * np.log2(p_sel) + (1 - p_sel) * np.log2(1 - p_sel))
        avg_entropy = np.mean(entropy_sel)

        pass_thresh = run_test(handler, test_pairs, p_bit1, thresh, emp_mod)
        gmr = 100 * pass_thresh / len(test_pairs)
        print(
            f"Denylist |p-0.5| > {thresh:.2f}: GMR={pass_thresh}/{len(test_pairs)} ({gmr:.2f}%), entropy bit TB={avg_entropy:.4f}"
        )

    handler.sess.close()


if __name__ == "__main__":
    main()
