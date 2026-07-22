"""
16_verify_quantizer_is_equal_probable.py

De xuat "sua quantizer sang COVQ" gia dinh quantizer hien tai la
equal-probable (nguong tai phan vi 25/50/75%, gay margin hep he thong o
2 bin trung tam). Script nay KIEM CHUNG gia dinh do bang du lieu that,
truoc khi quyet dinh co dang dau tu redesign quantizer hay khong:

  1. Ty le mau that roi vao moi bin (neu ~25% moi bin -> xac nhan
     equal-probable; neu lech nhieu -> quantizer KHONG phai equal-probable).
  2. Do rong moi bin (bin trung tam co thuc su hep hon bin duoi khong).
  3. Khoang cach trung binh toi nguong gan nhat, TACH RIENG theo tung bin.

Cach chay:
    python scripts/research/16_verify_quantizer_is_equal_probable.py
"""

import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)


def main():
    handler = WiFaKeyHandler()
    intervals = np.sort(np.asarray(handler.intervals).flatten())
    n_thr = len(intervals)

    print("=== Gia tri nguong thuc te (binarization_intervals.npy) ===")
    print(f"n_thr = {n_thr}, intervals = {intervals}")
    print(f"Khoang cach giua cac nguong lien tiep: {np.diff(intervals)}")
    print("(Neu equal-space/uniform: cac khoang cach nay BANG NHAU)")
    print("(Neu equal-probable tren Gaussian: khoang cach GIAM dan ra 2 dau)\n")

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    n_sample = min(len(files), 500)
    print(f"Lay mau {n_sample} embedding that de kiem tra phan phoi...")
    sample_files = files[:n_sample]

    all_projected_values = []
    for fname in sample_files:
        emb = np.load(os.path.join(CACHE_DIR, fname))
        projected = np.dot(emb, handler.M_matrix)
        all_projected_values.append(projected)
    all_projected_values = np.concatenate(all_projected_values)

    bin_idx = np.searchsorted(intervals, all_projected_values, side="left")

    print(
        f"\n=== Ty le mau that roi vao moi bin (ky vong equal-probable: ~{100/(n_thr+1):.1f}% moi bin) ==="
    )
    bin_fracs = []
    for b in range(n_thr + 1):
        frac = np.mean(bin_idx == b)
        bin_fracs.append(frac)
        print(f"  Bin {b}: {frac:.1%}")

    print("\n=== Khoang cach trung binh toi nguong GAN NHAT, tach theo bin ===")
    dist_to_nearest = np.array(
        [np.min(np.abs(intervals - v)) for v in all_projected_values]
    )
    for b in range(n_thr + 1):
        mask = bin_idx == b
        if mask.sum() > 0:
            print(
                f"  Bin {b}: mean_dist={dist_to_nearest[mask].mean():.5f}, "
                f"median_dist={np.median(dist_to_nearest[mask]):.5f}, n={mask.sum()}"
            )

    diffs = np.diff(intervals)
    is_uniform = np.allclose(diffs, diffs[0], rtol=0.05) if len(diffs) > 0 else True
    is_equal_prob = np.allclose(bin_fracs, 1 / (n_thr + 1), atol=0.03)

    print("\n=== Ket luan ===")
    if is_uniform:
        print("Nguong co ve UNIFORM (equal-space) - KHONG phai equal-probable.")
        print("-> Tien de cua de xuat COVQ (gia dinh equal-probable) SAI voi")
        print("   he thong thuc te nay - can xem lai toan bo chan doan margin hep.")
    elif is_equal_prob:
        print("Nguong khop equal-probable (moi bin ~deu xac suat).")
        print("-> Tien de cua de xuat COVQ DUNG voi he thong nay - dang can nhac")
        print("   huong redesign quantizer NEU sau khi sua loss van khong du.")
    else:
        print("Nguong khong khop ro rang voi ca 2 gia thuyet tren - can xem lai")
        print("bang mat cac con so in ra o tren de tu danh gia.")


if __name__ == "__main__":
    main()
