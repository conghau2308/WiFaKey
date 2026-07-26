"""
11_verify_with_empirical_llr.py

Tái tạo đúng logic WiFaKeyHandler.verify() gốc, chỉ thay bước
Modulation.BPSK() bằng research/modulation/v2_empirical_llr.EmpiricalLLR
— không sửa wifakey_handler.py gốc (giữ đúng nguyên tắc "chỉ 1 dòng đã
sửa" xuyên suốt dự án). Margin per-bit lấy từ
research/quantizer/v1_lssc_with_perbit_confidence.binarize_with_perbit_confidence
(đã self-test bit-for-bit trước khi dùng — xem self_check()).

Đo TAR/FAR trên cùng bộ pairs DemogPairs, cùng MAX_PAIRS_PER_CATEGORY=500,
để so sánh trực tiếp với 08_full_pipeline_tar_far_by_fold.py (hard-BPSK
gốc): TAR 71.8-85.9%, FAR 0/500 mọi fold.

Cách chạy:
    python scripts/11_verify_with_empirical_llr.py
"""

import os
import sys
import csv
import hashlib
import numpy as np
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import utils
from research.modulation.v2_empirical_llr import EmpiricalLLR
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
    _selftest_against_original,
)

DATASET_NAME = "demogpairs"
PAIRS_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs")
GENUINE_CSV = os.path.join(PAIRS_DIR, "audit_genuine.csv")
IMPOSTOR_SAMEFOLD_CSV = os.path.join(PAIRS_DIR, "audit_impostor_samefold.csv")
EMBEDDINGS_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)
OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "out_dim_separation_audit")
os.makedirs(OUT_DIR, exist_ok=True)

FOLDS = [
    "Asian_Females",
    "Asian_Males",
    "Black_Females",
    "Black_Males",
    "White_Females",
    "White_Males",
]
MAX_PAIRS_PER_CATEGORY = 500  # khớp 08/10, để so sánh trực tiếp
RNG_SEED = 42

llr_modulation = EmpiricalLLR()  # dùng masked_mag=1.25 mặc định (đã tune sẵn)

_embedding_cache = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    if cache_filename not in _embedding_cache:
        path = os.path.join(EMBEDDINGS_CACHE_DIR, cache_filename)
        _embedding_cache[cache_filename] = np.load(path)
    return _embedding_cache[cache_filename]


def compute_perbit_margin(embedding: np.ndarray) -> np.ndarray:
    """Dùng đúng binarize_with_perbit_confidence — margin RIÊNG cho từng
    bit (không phải khoảng cách tới ngưỡng gần nhất kiểu v0)."""
    projected = np.dot(np.asarray(embedding, dtype=np.float64), handler.M_matrix)
    _, margin = binarize_with_perbit_confidence(projected, handler.intervals)
    return margin


def real_binarize_full(embedding: np.ndarray) -> np.ndarray:
    projected = np.dot(np.asarray(embedding, dtype=np.float64), handler.M_matrix)
    return (
        utils.lssc_binary(projected[None, :], interval=handler.intervals)
        .flatten()
        .astype(np.uint8)
    )


def verify_with_llr(
    embedding: np.ndarray,
    helper_data: np.ndarray,
    mask_r: np.ndarray,
    stored_key_hash: bytes,
) -> bool:
    """Tái tạo handler.verify() nguyên bản, chỉ thay bước modulation."""
    b_full = real_binarize_full(embedding)
    b_masked = (b_full & mask_r).astype(np.uint8)
    b_selected = b_masked[: handler.feature_length]
    y_noisy_bits = np.logical_xor(b_selected, helper_data).astype(np.uint8)

    margin = compute_perbit_margin(embedding)[: handler.feature_length]
    mask_bool = mask_r[: handler.feature_length].astype(bool)

    y_llr = llr_modulation.modulate(
        y_noisy_bits, context={"margin": margin, "mask": mask_bool}
    ).reshape((1, handler.N, handler.Z))

    y_pred_llr = handler.sess.run(handler.decoder_output, feed_dict={handler.xa: y_llr})
    decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
    reconstructed_key = decoded_codeword[: handler.key_length]
    recon_hash = hashlib.sha256(reconstructed_key.tobytes()).digest()
    return recon_hash == stored_key_hash


# ---- Phần còn lại: self-check, load pairs, vòng lặp theo fold ----
def self_check():
    """(1) Self-test bit-for-bit BẮT BUỘC (theo đúng yêu cầu trong
    v1_lssc_with_perbit_confidence.py) — margin vô nghĩa nếu bit sinh ra
    không khớp lssc_binary gốc. (2) Self-match: enroll+verify cùng 1 ảnh
    qua verify_with_llr() phải luôn thành công."""
    print("[self-check 1/2] Đối chiếu bit-for-bit binarize_with_perbit_confidence...")
    _selftest_against_original(utils.lssc_binary, n_thr=handler.intervals.size)

    print("[self-check 2/2] Self-match qua verify_with_llr()...")
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f))
    emb = load_embedding(first_row["cache_filename_1"])

    n_trials, n_success = 5, 0
    for _ in range(n_trials):
        helper_data, mask_r, key_hash = handler.enroll(emb)
        ok = verify_with_llr(emb, helper_data, mask_r, key_hash)
        n_success += int(bool(ok))
    print(f"  self-match: {n_success}/{n_trials} thành công.")
    if n_success != n_trials:
        raise SystemExit(
            "[self-check] FAIL — self-match qua verify_with_llr() phải LUÔN "
            "thành công. DỪNG, không tin số liệu bên dưới."
        )


def load_pairs():
    genuine = defaultdict(list)
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            genuine[row["fold"]].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )
    impostor = defaultdict(list)
    with open(IMPOSTOR_SAMEFOLD_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            impostor[row["fold"]].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )
    return genuine, impostor


def subsample(pairs, rng, max_n):
    if len(pairs) <= max_n:
        return pairs
    idx = rng.choice(len(pairs), size=max_n, replace=False)
    return [pairs[i] for i in idx]


def measure_success_rate(pairs, is_genuine: bool, label: str) -> dict:
    if not pairs:
        return {"n": 0, "success_rate": np.nan}
    n_success = 0
    for f1, f2 in pairs:
        emb1, emb2 = load_embedding(f1), load_embedding(f2)
        helper_data, mask_r, key_hash = handler.enroll(emb1)
        ok = verify_with_llr(emb2, helper_data, mask_r, key_hash)
        n_success += int(bool(ok))
    rate = n_success / len(pairs)
    metric = "TAR" if is_genuine else "FAR"
    print(f"  [{label}] n={len(pairs)}  {metric}={rate:.4f} ({n_success}/{len(pairs)})")
    return {"n": len(pairs), "success_rate": rate}


def main():
    self_check()
    rng = np.random.default_rng(RNG_SEED)
    genuine, impostor = load_pairs()

    print("\n" + "=" * 70)
    print("TAR/FAR với EmpiricalLLR — theo fold (so trực tiếp với 08: hard-BPSK)")
    print("=" * 70)
    tar_results, far_results = {}, {}
    for fold in FOLDS:
        print(f"\n[{fold}]")
        g_pairs = subsample(genuine.get(fold, []), rng, MAX_PAIRS_PER_CATEGORY)
        i_pairs = subsample(impostor.get(fold, []), rng, MAX_PAIRS_PER_CATEGORY)
        tar_results[fold] = measure_success_rate(g_pairs, True, f"{fold} genuine")
        far_results[fold] = measure_success_rate(i_pairs, False, f"{fold} impostor")

    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    for fold in FOLDS:
        if fold in tar_results and far_results.get(fold, {}).get("n", 0) > 0:
            far = far_results[fold]["success_rate"]
            flag = " *** FAR>0 — LỖ HỔNG, cần điều tra ***" if far > 0 else ""
            print(
                f"  {fold:16s}: TAR={tar_results[fold]['success_rate']:.4f}  "
                f"FAR={far:.4f}{flag}"
            )

    np.savez(
        os.path.join(OUT_DIR, "empirical_llr_tar_far.npz"),
        tar=np.array(list(tar_results.items()), dtype=object),
        far=np.array(list(far_results.items()), dtype=object),
    )
    print(f"\n[out] -> {os.path.join(OUT_DIR, 'empirical_llr_tar_far.npz')}")


if __name__ == "__main__":
    handler = WiFaKeyHandler()
    main()
