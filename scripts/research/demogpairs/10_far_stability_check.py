"""
12_far_stability_check.py

Kiểm tra FAR=1/500 (0.20%) vừa phát hiện ở Asian_Males (script 11) có
LẶP LẠI hay chỉ là 1 sự kiện đơn lẻ do may rủi của mask_r — vì mask_r/
random_key sinh MỚI ngẫu nhiên mỗi lần enroll() (không seed theo
identity), nên không thể "log lại đúng cặp cũ" để điều tra tĩnh; thay
vào đó đo XÁC SUẤT LẶP LẠI bằng cách chạy lại nhiều lần.

THIẾT KẾ: lặp N_REPEATS lần enroll+verify_with_llr trên CÙNG 500 cặp
impostor của Asian_Males (mỗi lần mask mới, đúng thiết kế gốc), ghi lại
tỷ lệ thành công-sai mỗi lần lặp.
  - Nếu tỷ lệ dao động quanh ~0.2% qua nhiều lần lặp (không chỉ 1/2500
    lượt) -> đặc tính hệ thống thật, đáng sửa masked_mag.
  - Nếu chỉ ~1/2500 (~0.04%) -> nhiều khả năng nhiễu thống kê mẫu nhỏ.

Cách chạy:
    python scripts/12_far_stability_check.py
"""

import os
import sys
import csv
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
IMPOSTOR_SAMEFOLD_CSV = os.path.join(PAIRS_DIR, "audit_impostor_samefold.csv")
GENUINE_CSV = os.path.join(PAIRS_DIR, "audit_genuine.csv")
EMBEDDINGS_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)
OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "out_dim_separation_audit")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_FOLD = "Asian_Males"  # fold vừa phát hiện FAR>0 ở script 11
MAX_PAIRS = 500  # khớp đúng 11, để dùng lại đúng bộ pairs đã cho FAR=1/500
N_REPEATS = 5  # 5 x 500 = 2500 lượt thử, mỗi lượt mask_r mới
RNG_SEED = 42  # cùng seed với 08/11 -> subsample ra đúng cùng 500 cặp

llr_modulation = EmpiricalLLR()

_embedding_cache = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    if cache_filename not in _embedding_cache:
        path = os.path.join(EMBEDDINGS_CACHE_DIR, cache_filename)
        _embedding_cache[cache_filename] = np.load(path)
    return _embedding_cache[cache_filename]


def compute_perbit_margin(embedding: np.ndarray) -> np.ndarray:
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


def verify_with_llr(embedding, helper_data, mask_r, stored_key_hash) -> bool:
    import hashlib

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


def self_check():
    print("[self-check 1/2] Đối chiếu bit-for-bit binarize_with_perbit_confidence...")
    _selftest_against_original(utils.lssc_binary, n_thr=handler.intervals.size)

    print("[self-check 2/2] Self-match qua verify_with_llr()...")
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f))
    emb = load_embedding(first_row["cache_filename_1"])
    n_trials, n_success = 5, 0
    for _ in range(n_trials):
        helper_data, mask_r, key_hash = handler.enroll(emb)
        n_success += int(bool(verify_with_llr(emb, helper_data, mask_r, key_hash)))
    print(f"  self-match: {n_success}/{n_trials} thành công.")
    if n_success != n_trials:
        raise SystemExit("[self-check] FAIL — dừng, không tin số liệu bên dưới.")


def load_target_pairs(rng):
    impostor_by_fold = defaultdict(list)
    with open(IMPOSTOR_SAMEFOLD_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            impostor_by_fold[row["fold"]].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )
    pairs = impostor_by_fold[TARGET_FOLD]
    if len(pairs) > MAX_PAIRS:
        idx = rng.choice(len(pairs), size=MAX_PAIRS, replace=False)
        pairs = [pairs[i] for i in idx]
    return pairs


def main():
    self_check()
    rng = np.random.default_rng(RNG_SEED)
    pairs = load_target_pairs(rng)
    print(
        f"\n[{TARGET_FOLD}] {len(pairs)} cặp impostor — lặp {N_REPEATS} lần "
        f"({len(pairs) * N_REPEATS} lượt thử tổng, mỗi lượt mask_r mới)\n"
    )

    per_repeat_results = []
    total_success = 0
    total_trials = 0

    for r in range(1, N_REPEATS + 1):
        n_success = 0
        for f1, f2 in pairs:
            emb1, emb2 = load_embedding(f1), load_embedding(f2)
            helper_data, mask_r, key_hash = handler.enroll(emb1)
            ok = verify_with_llr(emb2, helper_data, mask_r, key_hash)
            n_success += int(bool(ok))
        rate = n_success / len(pairs)
        per_repeat_results.append(n_success)
        total_success += n_success
        total_trials += len(pairs)
        print(
            f"  Lần {r}/{N_REPEATS}: {n_success}/{len(pairs)} thành công sai "
            f"(FAR={rate:.4f})"
        )

    print("\n" + "=" * 70)
    print("TỔNG KẾT ĐỘ ỔN ĐỊNH")
    print("=" * 70)
    overall_rate = total_success / total_trials
    print(
        f"Tổng: {total_success}/{total_trials} lượt thành công sai "
        f"(FAR gộp={overall_rate:.4f})"
    )
    print(
        f"Số lần lặp có ít nhất 1 thành công sai: "
        f"{sum(1 for n in per_repeat_results if n > 0)}/{N_REPEATS}"
    )

    if sum(1 for n in per_repeat_results if n > 0) <= 1:
        print(
            "\nDIỄN GIẢI: chỉ 1/{} lần lặp có sự kiện FAR>0 -> nghiêng về NHIỄU "
            "THỐNG KÊ ở mẫu nhỏ, không phải đặc tính hệ thống lặp lại ổn định. "
            "Chưa cần sửa masked_mag ngay, nhưng đáng theo dõi.".format(N_REPEATS)
        )
    else:
        print(
            "\nDIỄN GIẢI: FAR>0 xuất hiện LẶP LẠI qua nhiều lần độc lập -> đặc "
            "tính hệ thống thật (một số cặp người có khoảng cách đủ gần để "
            "occasionally decode thành công), không phải nhiễu đơn lẻ. Đáng "
            "cân nhắc retune masked_mag riêng cho DemogPairs (hiện=1.25, tune "
            "trên LFW/CPLFW) trước khi mở rộng benchmark sang cross-fold."
        )

    np.savez(
        os.path.join(OUT_DIR, "far_stability_result.npz"),
        target_fold=TARGET_FOLD,
        per_repeat_success=np.array(per_repeat_results),
        n_pairs=len(pairs),
        n_repeats=N_REPEATS,
        overall_rate=overall_rate,
    )
    print(f"\n[out] -> {os.path.join(OUT_DIR, 'far_stability_result.npz')}")


if __name__ == "__main__":
    handler = WiFaKeyHandler()
    main()
