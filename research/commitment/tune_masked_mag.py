"""
tune_masked_mag.py

Quét tham số masked_mag cho Empirical LLR trên tập CPLFW
để tìm giá trị tối ưu giữ GMR và FAR.
Có bảng tổng hợp kết quả cuối cùng.
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


# --- Loaders ---
def load_embedding(cache_dir, name, imagenum):
    path = os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy")
    return np.load(path)


def load_pairs(pairs_csv, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def pairs_iter(pairs_csv, cache_dir, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2, row
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


class MarginSelectionHandler(SecureWiFaKeyHandler):
    def enroll(self, feature_vector_float):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        projected = np.dot(feature_vector_float, self.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, self.intervals)
        selection_indices = np.argpartition(-margin, FEATURE_LENGTH)[:FEATURE_LENGTH]
        selection_indices.sort()
        selection_mask = np.zeros(FULL_BITS, dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected = b_full[selection_indices]
        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()
        return helper_data, selection_mask, key_hash


def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def main():
    dataset_folder = "cplfw"
    tier = "select"
    pairs_dir = os.path.join(
        _PROJECT_ROOT, "datasets", "processed", dataset_folder, "pairs"
    )
    cache_dir = os.path.join(
        _PROJECT_ROOT, "datasets", "processed", dataset_folder, "embeddings_cache"
    )
    genuine_csv = os.path.join(pairs_dir, f"{tier}_genuine.csv")
    impostor_csv = os.path.join(pairs_dir, f"{tier}_impostor.csv")

    handler = MarginSelectionHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    lookup_path = os.path.join(
        _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
    )

    masked_mag_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    results = []

    print("=" * 60)
    print("QUÉT THAM SỐ masked_mag TRÊN TẬP CPLFW")
    print("=" * 60)

    for masked_mag in masked_mag_values:
        # Genuine
        gen_iter = pairs_iter(genuine_csv, cache_dir, max_pairs=500)
        gen_succ, gen_total = 0, 0
        emp_mod = EmpiricalLLR(lookup_path=lookup_path, masked_mag=masked_mag)
        for feat_enroll, feat_verify, _ in gen_iter:
            helper, sel_mask, key_hash = handler.enroll(feat_enroll)
            b_full_v = handler._binarize_full(feat_verify).astype(np.uint8)
            idx = np.where(sel_mask == 1)[0]
            b_sel = b_full_v[idx]
            noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
            _, margin_v = binarize_with_perbit_confidence(
                np.dot(feat_verify, handler.M_matrix), handler.intervals
            )
            margin_sel = margin_v[idx]
            llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
            if _try_decode(handler, llr, key_hash):
                gen_succ += 1
            gen_total += 1
        gmr = 100 * gen_succ / gen_total if gen_total else 0

        # Impostor
        imp_iter = pairs_iter(impostor_csv, cache_dir, max_pairs=500)
        imp_succ, imp_total = 0, 0
        for feat_enroll, feat_verify, _ in imp_iter:
            helper, sel_mask, key_hash = handler.enroll(feat_enroll)
            b_full_v = handler._binarize_full(feat_verify).astype(np.uint8)
            idx = np.where(sel_mask == 1)[0]
            b_sel = b_full_v[idx]
            noisy = np.logical_xor(b_sel, helper).astype(np.uint8)
            _, margin_v = binarize_with_perbit_confidence(
                np.dot(feat_verify, handler.M_matrix), handler.intervals
            )
            margin_sel = margin_v[idx]
            llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
            if _try_decode(handler, llr, key_hash):
                imp_succ += 1
            imp_total += 1
        far = 100 * imp_succ / imp_total if imp_total else 0

        results.append((masked_mag, gmr, far, gen_succ, gen_total, imp_succ, imp_total))
        print(
            f"masked_mag={masked_mag:.2f}: GMR={gmr:.2f}% ({gen_succ}/{gen_total}), FAR={far:.4f}% ({imp_succ}/{imp_total})"
        )

    # ---- BẢNG TỔNG HỢP ----
    print("\n" + "=" * 70)
    print("TỔNG HỢP KẾT QUẢ QUÉT masked_mag")
    print("=" * 70)
    print(
        f"{'masked_mag':<12} {'GMR':<12} {'FAR':<12} {'Genuine':<15} {'Impostor':<15}"
    )
    print("-" * 70)
    for masked_mag, gmr, far, gen_succ, gen_total, imp_succ, imp_total in results:
        print(
            f"{masked_mag:<12.2f} {gmr:<12.2f}% {far:<12.4f}% {gen_succ}/{gen_total:<10} {imp_succ}/{imp_total:<10}"
        )

    # Tìm giá trị tốt nhất (FAR = 0, GMR cao nhất)
    best = None
    for r in results:
        if r[2] == 0.0:
            if best is None or r[1] > best[1]:
                best = r
    if best:
        print(
            f"\n✅ Giá trị tối ưu (FAR=0%, GMR cao nhất): masked_mag = {best[0]:.2f} (GMR = {best[1]:.2f}%)"
        )
    else:
        # Tìm giá trị FAR thấp nhất, GMR cao nhất
        best = min(results, key=lambda x: (x[2], -x[1]))
        print(
            f"\n⚠️ Không có giá trị nào cho FAR=0%. Giá trị tốt nhất: masked_mag = {best[0]:.2f} (GMR = {best[1]:.2f}%, FAR = {best[2]:.4f}%)"
        )

    handler.sess.close()


if __name__ == "__main__":
    main()
