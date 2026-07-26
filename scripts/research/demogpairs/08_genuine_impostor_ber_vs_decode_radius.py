"""
10_genuine_impostor_ber_vs_decode_radius.py

Chẩn đoán: TAR thấp (71-86%) là do (a) điểm vận hành hiện tại quá bảo
thủ — đuôi genuine vượt khả năng sửa lỗi thật của decoder nhưng vẫn
dưới impostor min rất xa (có "dư FAR" để đổi lấy TAR), hay (b) genuine/
impostor chồng lấn thật (không có gì để đổi). Đo THỰC NGHIỆM, không
suy diễn công thức lý thuyết cho "bán kính sửa lỗi".

CÁCH ĐO BER ĐẦU VÀO DECODER — KHÔNG CẦN BIẾT CODEWORD GỐC:
Từ mã nguồn wifakey_handler.py gốc:
  helper_data = b_selected_enroll XOR codeword
  y_noisy_bits (lúc verify) = b_selected_verify XOR helper_data
                             = b_selected_verify XOR b_selected_enroll XOR codeword
  => y_noisy_bits XOR codeword = b_selected_verify XOR b_selected_enroll
Nghĩa là "nhiễu" thật sự đưa vào decoder CHÍNH LÀ Hamming distance giữa
b_selected của 2 lần binarize (enroll vs verify) — đúng đại lượng
Genuine_BER/Impostor_BER đã đo xuyên suốt, chỉ khác: lần này đo trên
TOÀN BỘ 832 bit SAU MASK (không phải trung bình per-dimension trước
mask như 05/06/07/09). Không cần biết random_key/codeword thật để tính
— chỉ cần binarize lại 2 ảnh với ĐÚNG mask_r mà enroll() đã sinh ra
(enroll() trả về mask_r, tận dụng trực tiếp).

Cách chạy:
    python scripts/10_genuine_impostor_ber_vs_decode_radius.py
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

MAX_PAIRS_PER_CATEGORY = 500  # khớp 08, tăng dần sau khi xác nhận ổn
RNG_SEED = 42
N_BER_BINS = 20  # số bin cho đường cong empirical success-rate vs BER

_embedding_cache = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    if cache_filename not in _embedding_cache:
        path = os.path.join(EMBEDDINGS_CACHE_DIR, cache_filename)
        _embedding_cache[cache_filename] = np.load(path)
    return _embedding_cache[cache_filename]


def real_binarize_full(embedding: np.ndarray) -> np.ndarray:
    """Tái hiện đúng handler._binarize_full() — self-check bên dưới xác
    nhận khớp trước khi tin bất kỳ số liệu nào."""
    projected = np.dot(np.asarray(embedding, dtype=np.float64), handler.M_matrix)
    return (
        utils.lssc_binary(projected[None, :], interval=handler.intervals)
        .flatten()
        .astype(np.uint8)
    )


def compute_selected_bits(embedding: np.ndarray, mask_r: np.ndarray) -> np.ndarray:
    """Tái hiện đúng bước b_masked = b_full & mask_r; b_selected =
    b_masked[:feature_length] trong enroll()/verify() gốc."""
    b_full = real_binarize_full(embedding).astype(np.uint8)
    b_masked = (b_full & mask_r).astype(np.uint8)
    return b_masked[: handler.feature_length]


def self_check():
    """(1) self-match: enroll+verify cùng 1 ảnh phải luôn thành công VÀ
    BER tính lại phải =0. (2) BER tính lại (qua compute_selected_bits)
    phải khớp NHIỄU THẬT decoder nhận — kiểm gián tiếp bằng cách BER=0
    khi 2 ảnh giống hệt nhau (điều kiện cần, không đủ, nhưng là self-
    check khả thi không cần lộ codeword nội bộ của handler)."""
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f))
    emb = load_embedding(first_row["cache_filename_1"])

    n_trials, n_success, n_ber_zero = 5, 0, 0
    for _ in range(n_trials):
        helper_data, mask_r, key_hash = handler.enroll(emb)
        ok = handler.verify(emb, helper_data, mask_r, key_hash)
        n_success += int(bool(ok))

        b1 = compute_selected_bits(emb, mask_r)
        b2 = compute_selected_bits(emb, mask_r)  # cùng ảnh, cùng mask -> phải =0
        ber = np.mean(b1 != b2)
        n_ber_zero += int(ber == 0.0)

    print(f"[self-check] self-match verify: {n_success}/{n_trials} thành công.")
    print(f"[self-check] self-match BER=0: {n_ber_zero}/{n_trials}.")
    if n_success != n_trials or n_ber_zero != n_trials:
        raise SystemExit(
            "[self-check] FAIL — self-match phải LUÔN thành công VÀ BER phải "
            "luôn =0. DỪNG, kiểm tra lại compute_selected_bits()/mask_r trước "
            "khi tin số liệu bên dưới."
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


def measure_ber_and_success(pairs, rng, label):
    """Với mỗi pair: enroll ảnh 1 (sinh mask_r MỚI, đúng thiết kế gốc),
    tính BER thật (b_selected ảnh 1 vs ảnh 2, CÙNG mask_r), và verify()
    thật để lấy success/fail. Trả list[(ber, success)]."""
    records = []
    for f1, f2 in pairs:
        emb1, emb2 = load_embedding(f1), load_embedding(f2)
        helper_data, mask_r, key_hash = handler.enroll(emb1)

        b1 = compute_selected_bits(emb1, mask_r)
        b2 = compute_selected_bits(emb2, mask_r)
        ber = float(np.mean(b1 != b2))

        success = bool(handler.verify(emb2, helper_data, mask_r, key_hash))
        records.append((ber, success))
    return records


def summarize_ber(records, percentiles=(5, 50, 95)):
    bers = np.array([b for b, _ in records])
    out = {
        "mean": float(bers.mean()),
        "min": float(bers.min()),
        "max": float(bers.max()),
    }
    for p in percentiles:
        out[f"P{p}"] = float(np.percentile(bers, p))
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    self_check()

    genuine, impostor = load_pairs()

    all_records = []  # (ber, success, is_genuine, fold) — dùng cho đường cong chung
    genuine_summaries, impostor_summaries = {}, {}

    for fold in FOLDS:
        print(f"\n[{fold}]")
        g_pairs = subsample(genuine.get(fold, []), rng, MAX_PAIRS_PER_CATEGORY)
        i_pairs = subsample(impostor.get(fold, []), rng, MAX_PAIRS_PER_CATEGORY)

        g_records = measure_ber_and_success(g_pairs, rng, f"{fold} genuine")
        i_records = measure_ber_and_success(i_pairs, rng, f"{fold} impostor")

        g_sum = summarize_ber(g_records)
        i_sum = summarize_ber(i_records)
        genuine_summaries[fold] = g_sum
        impostor_summaries[fold] = i_sum

        g_tar = np.mean([s for _, s in g_records]) if g_records else np.nan
        i_far = np.mean([s for _, s in i_records]) if i_records else np.nan
        print(
            f"  genuine : n={len(g_records):4d}  BER mean={g_sum['mean']:.4f}  "
            f"P95={g_sum['P95']:.4f}  max={g_sum['max']:.4f}  TAR={g_tar:.4f}"
        )
        print(
            f"  impostor: n={len(i_records):4d}  BER mean={i_sum['mean']:.4f}  "
            f"min={i_sum['min']:.4f}  P5={i_sum['P5']:.4f}  FAR={i_far:.4f}"
        )

        all_records += [(b, s, True, fold) for b, s in g_records]
        all_records += [(b, s, False, fold) for b, s in i_records]

    # ---- Đường cong empirical: decode success-rate theo BER (gộp genuine+impostor) ----
    print("\n" + "=" * 70)
    print("ĐƯỜNG CONG THỰC NGHIỆM: decode success-rate theo BER đầu vào")
    print("(đây là 'bán kính sửa lỗi' THẬT của Neural-MS đã train, đo trực tiếp)")
    print("=" * 70)
    bers = np.array([b for b, _, _, _ in all_records])
    successes = np.array([s for _, s, _, _ in all_records])
    bin_edges = np.linspace(0, max(bers.max(), 0.5), N_BER_BINS + 1)
    radius_50 = radius_90 = None
    for i in range(N_BER_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (bers >= lo) & (bers < hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        rate = successes[mask].mean()
        print(f"  BER∈[{lo:.3f},{hi:.3f})  n={n_in_bin:5d}  success_rate={rate:.3f}")
        if radius_90 is None and rate < 0.90:
            radius_90 = lo
        if radius_50 is None and rate < 0.50:
            radius_50 = lo

    print(
        f"\nBán kính sửa lỗi thực nghiệm (BER nơi success rate tụt dưới 90%): "
        f"~{radius_90}"
    )
    print(
        f"Bán kính sửa lỗi thực nghiệm (BER nơi success rate tụt dưới 50%): "
        f"~{radius_50}"
    )

    print("\n" + "=" * 70)
    print("SO SÁNH ĐUÔI GENUINE vs BÁN KÍNH, THEO FOLD — chẩn đoán (a) vs (b)")
    print("=" * 70)
    for fold in FOLDS:
        if fold not in genuine_summaries or fold not in impostor_summaries:
            continue
        g_p95 = genuine_summaries[fold]["P95"]
        i_min = impostor_summaries[fold]["min"]
        print(
            f"  {fold:16s}: genuine P95={g_p95:.4f}  impostor min={i_min:.4f}  "
            f"gap(impostor_min - genuine_P95)={i_min - g_p95:+.4f}"
        )
    print(
        "\nDIỄN GIẢI:\n"
        "  - Nếu genuine P95 xấp xỉ/vượt bán kính (radius_90 ở trên) NHƯNG\n"
        "    impostor min vẫn cách xa bán kính đó -> (a) điểm vận hành quá\n"
        "    bảo thủ, CÓ dư FAR thật để đổi lấy TAR (retrain decoder/tăng\n"
        "    iters_max là hướng đúng, kỳ vọng cải thiện TAR mà FAR vẫn ~0).\n"
        "  - Nếu genuine P95 và impostor min RẤT GẦN NHAU (khoảng gap nhỏ)\n"
        "    -> (b) genuine/impostor chồng lấn thật ở đúng vùng biên, nới\n"
        "    bán kính sẽ kéo theo FAR tăng thật — không còn 'dư' để đổi."
    )

    np.savez(
        os.path.join(OUT_DIR, "ber_vs_radius_result.npz"),
        genuine_summaries=np.array(list(genuine_summaries.items()), dtype=object),
        impostor_summaries=np.array(list(impostor_summaries.items()), dtype=object),
        all_bers=bers,
        all_successes=successes,
        all_is_genuine=np.array([g for _, _, g, _ in all_records]),
        all_folds=np.array([f for _, _, _, f in all_records]),
    )
    print(f"\n[out] -> {os.path.join(OUT_DIR, 'ber_vs_radius_result.npz')}")


if __name__ == "__main__":
    handler = WiFaKeyHandler()
    main()
