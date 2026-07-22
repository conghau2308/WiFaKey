"""
run_ab_paired.py

So sánh CÔNG BẰNG THẬT SỰ (paired) giữa 3 variant trên CÙNG tập 'select':
  - hard_bpsk     : quantizer gốc + v0_hard_bpsk
  - soft_llr_v1   : v0_lssc_with_confidence + v1_soft_distance_llr (scale=60)
  - empirical_llr : v1_lssc_with_perbit_confidence + v2_empirical_llr (bảng
                    hiệu chỉnh thực nghiệm, decoder GỐC, KHÔNG fine-tune -
                    xem lịch sử hội thoại: fine-tune 20 epoch không cải
                    thiện, kết luận plateau thật).

VẤN ĐỀ ĐÃ SỬA so với run_ab_soft_llr.py / run_ab_empirical_llr.py cũ:
`WiFaKeyHandler.enroll()` dùng np.random.uniform/np.random.randint (RNG
TOÀN CỤC, không seed theo từng cặp). Chạy 2 script riêng biệt ở 2 lần gọi
khác nhau (như trước) khiến mỗi variant enroll với random_key/mask_r KHÁC
NHAU cho cùng 1 cặp embedding -> so sánh 401/413 vs 400/413 không phải
paired, chênh lệch 1 mẫu có thể chỉ do khác random_key, không phải do chất
lượng modulation.

CÁCH SỬA (không đụng wifakey_handler.py gốc): trước MỖI cặp, reseed
np.random bằng 1 seed suy ra ỔN ĐỊNH từ hash của (name_enroll,
imagenum_enroll, name_verify, imagenum_verify), rồi enroll() MỘT LẦN
DUY NHẤT và dùng chung helper_data/mask_r/key_hash cho cả 3 variant.
Seed giống hệt nhau + enroll giống hệt nhau -> đảm bảo random_key/mask_r
giống hệt nhau giữa các variant, chỉ có modulation/quantizer khác nhau.
Đây mới là so sánh công bằng đúng nghĩa (và tránh gọi enroll() 3 lần dư
thừa cho cùng 1 kết quả).

HỖ TRỢ NHIỀU DATASET (--dataset):
  - labeled_faces_in_the_wild            : có cả genuine + impostor pairs
  - face-detection-and-re-identification : CHỈ có genuine pairs (mỗi person
    1 cặp 0.jpg vs 1.jpg) -> chỉ đo được FRR, không đo FAR. Script tự động
    phát hiện thiếu file impostor và bỏ qua phần đo FAR/McNemar tương ứng.

Yêu cầu chuẩn bị trước (tuỳ dataset):
    python scripts/01_extract_embeddings.py
    python scripts/02_build_pairs_dataset.py
  hoặc (dataset face-detection):
    python scripts/02b_extract_embeddings_face_detection.py

Cách chạy:
    python experiments/run_ab_paired.py --dataset labeled_faces_in_the_wild
    python experiments/run_ab_paired.py --dataset face-detection-and-re-identification
"""

import sys
import os
import csv
import json
import hashlib
import argparse
import numpy as np
from scipy.stats import binomtest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib.utils import lssc_binary

from research.quantizer.v0_lssc_with_confidence import (
    binarize_with_confidence,
    _selftest_against_original as _selftest_v0,
)
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
    _selftest_against_original as _selftest_v1,
)
from research.modulation.v0_hard_bpsk import HardBPSK
from research.modulation.v1_soft_distance_llr import SoftDistanceLLR
from research.modulation.v2_empirical_llr import EmpiricalLLR
from research.decoder.v0_neural_ms_original import NeuralMSOriginal
from research.pipeline.verify_variant import verify_with_variant

_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "experiments", "results")
_LOOKUP_PATH = os.path.join(
    _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
)
MASKED_MAG = 1.0

# Danh sách dataset hợp lệ - thêm dataset mới vào đây khi cần mở rộng
KNOWN_DATASETS = [
    "labeled_faces_in_the_wild",
    "face-detection-and-re-identification",
    "cplfw",
]


def get_dataset_paths(dataset_name: str):
    """Trả về (cache_dir, pairs_dir) cho 1 dataset, dựa trên cấu trúc thư mục
    chuẩn datasets/processed/<dataset_name>/{embeddings_cache,pairs}."""
    cache_dir = os.path.join(
        _PROJECT_ROOT, "datasets", "processed", dataset_name, "embeddings_cache"
    )
    pairs_dir = os.path.join(
        _PROJECT_ROOT, "datasets", "processed", dataset_name, "pairs"
    )
    return cache_dir, pairs_dir


def _load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    path = os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy embedding {path}. Chạy script trích xuất embedding "
            f"cho dataset này trước."
        )
    return np.load(path)


def stable_seed(name_enroll, imagenum_enroll, name_verify, imagenum_verify) -> int:
    """Seed uint32 ổn định, suy từ hash SHA-256 của định danh cặp - KHÔNG phụ
    thuộc thứ tự dòng trong CSV, tái lập được 100% giữa các lần chạy."""
    key_str = f"{name_enroll}_{imagenum_enroll}_{name_verify}_{imagenum_verify}"
    digest = hashlib.sha256(key_str.encode("utf-8")).digest()
    return int.from_bytes(
        digest[:4], byteorder="big"
    )  # 0 .. 2**32-1, hợp lệ cho np.random.seed


def load_test_pairs_with_ids(cache_dir: str, pairs_dir: str, tier: str = "select"):
    """Đọc {tier}_genuine.csv (bắt buộc) và {tier}_impostor.csv (TUỲ CHỌN -
    một số dataset như face-detection-and-re-identification chỉ có genuine).
    Giữ lại (name, imagenum) của cả 2 phía - cần để tính stable_seed() cho
    từng cặp."""
    genuine_path = os.path.join(pairs_dir, f"{tier}_genuine.csv")
    impostor_path = os.path.join(pairs_dir, f"{tier}_impostor.csv")

    if not os.path.exists(genuine_path):
        raise FileNotFoundError(
            f"Không tìm thấy {genuine_path}. Chạy script build pairs cho "
            f"dataset này trước."
        )

    has_impostor = os.path.exists(impostor_path)
    sources = [(genuine_path, True)]
    if has_impostor:
        sources.append((impostor_path, False))
    else:
        print(
            f"[{tier}] Không tìm thấy {impostor_path} - dataset này chỉ có "
            f"genuine pairs, sẽ CHỈ đo được FRR (không đo FAR)."
        )

    pairs = []
    for path, is_genuine in sources:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emb_enroll = _load_embedding(
                    cache_dir, row["name_enroll"], row["imagenum_enroll"]
                )
                emb_verify = _load_embedding(
                    cache_dir, row["name_verify"], row["imagenum_verify"]
                )
                seed = stable_seed(
                    row["name_enroll"],
                    row["imagenum_enroll"],
                    row["name_verify"],
                    row["imagenum_verify"],
                )
                pairs.append(
                    dict(
                        emb_enroll=emb_enroll,
                        emb_verify=emb_verify,
                        is_genuine=is_genuine,
                        seed=seed,
                    )
                )

    print(
        f"[{tier}] Đã load {len(pairs)} cặp "
        f"({sum(1 for p in pairs if p['is_genuine'])} genuine, "
        f"{sum(1 for p in pairs if not p['is_genuine'])} impostor)"
    )
    return pairs, has_impostor


def run_all_variants_paired(handler, decoder, variants, test_pairs):
    """variants: list of (name, modulation_instance, quantizer_fn).
    Với MỖI cặp: reseed 1 lần rồi enroll() 1 LẦN DUY NHẤT (không phụ thuộc
    variant), sau đó chạy verify với cả 3 variant trên CÙNG helper_data/
    mask_r/key_hash đó -> đảm bảo random_key giống hệt nhau giữa các
    variant mà không tốn 3 lần enroll() cho cùng 1 kết quả.

    Trả về:
        metrics[name]  -> {"FRR", "FAR", "genuine_success", ...}
        per_pair[name] -> list[int] 0/1, THEO ĐÚNG THỨ TỰ test_pairs
                          (dùng để chạy McNemar's test giữa các variant)
    """
    metrics = {
        name: {
            "genuine_success": 0,
            "genuine_total": 0,
            "impostor_success": 0,
            "impostor_total": 0,
        }
        for name, _, _ in variants
    }
    per_pair = {name: [] for name, _, _ in variants}

    for pair in test_pairs:
        # Reseed 1 lần / cặp rồi enroll() 1 lần duy nhất - dùng chung cho
        # cả 3 variant bên dưới (enroll() không phụ thuộc modulation/
        # quantizer nên gọi lặp lại 3 lần chỉ cho cùng 1 kết quả là dư thừa).
        np.random.seed(pair["seed"])
        helper_data, mask_r, key_hash = handler.enroll(pair["emb_enroll"])

        for name, modulation, quantizer_fn in variants:
            success, _ = verify_with_variant(
                handler,
                quantizer_fn,
                modulation,
                decoder,
                pair["emb_verify"],
                helper_data,
                mask_r,
                key_hash,
            )
            per_pair[name].append(int(success))

            m = metrics[name]
            if pair["is_genuine"]:
                m["genuine_total"] += 1
                m["genuine_success"] += int(success)
            else:
                m["impostor_total"] += 1
                m["impostor_success"] += int(success)  # false accept nếu =1

    for name in metrics:
        m = metrics[name]
        m["FRR"] = 1 - m["genuine_success"] / max(m["genuine_total"], 1)
        m["FAR"] = m["impostor_success"] / max(m["impostor_total"], 1)

    return metrics, per_pair


def mcnemar_exact(success_a, success_b, label_a, label_b, subset_mask=None):
    """McNemar's exact test (binomial 2 phía trên các cặp DISCORDANT) - dùng
    khi b+c nhỏ, không cần statsmodels. subset_mask: list bool để chỉ tính
    trên 1 tập con (vd chỉ genuine pairs)."""
    a = np.array(success_a)
    b = np.array(success_b)
    if subset_mask is not None:
        m = np.array(subset_mask)
        a, b = a[m], b[m]

    if len(a) == 0:
        print(f"  [{label_a} vs {label_b}] Không có mẫu nào trong subset - bỏ qua.")
        return 0, 0, None

    # b_count: A đúng, B sai. c_count: A sai, B đúng. (quy ước McNemar)
    b_count = int(np.sum((a == 1) & (b == 0)))
    c_count = int(np.sum((a == 0) & (b == 1)))
    n_discordant = b_count + c_count

    if n_discordant == 0:
        print(
            f"  [{label_a} vs {label_b}] Không có cặp discordant nào - 2 variant "
            f"THẬT SỰ giống hệt nhau trên tập test này (mọi mẫu cùng đúng/sai)."
        )
        return b_count, c_count, 1.0

    p_value = binomtest(
        min(b_count, c_count), n_discordant, 0.5, alternative="two-sided"
    ).pvalue
    sig = (
        "CÓ ý nghĩa (p<0.05)"
        if p_value < 0.05
        else "KHÔNG có ý nghĩa thống kê (p>=0.05)"
    )
    print(
        f"  [{label_a} vs {label_b}] {label_a} đúng/{label_b} sai: {b_count}   "
        f"{label_a} sai/{label_b} đúng: {c_count}   "
        f"McNemar exact p-value = {p_value:.4f}  -> {sig}"
    )
    return b_count, c_count, p_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="So sánh paired giữa hard_bpsk / soft_llr_v1 / empirical_llr."
    )
    parser.add_argument(
        "--dataset",
        default="labeled_faces_in_the_wild",
        choices=KNOWN_DATASETS,
        help="Dataset để chạy A/B test (quyết định thư mục cache/pairs).",
    )
    parser.add_argument(
        "--tier",
        default="select",
        help="Prefix của file pairs, vd 'select' -> select_genuine.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== Dataset: {args.dataset} | tier: {args.tier} ===")
    cache_dir, pairs_dir = get_dataset_paths(args.dataset)

    print("Self-check quantizer (v0 block-shared + v1 per-bit) trước khi so sánh...")
    _selftest_v0(lssc_binary)
    _selftest_v1(lssc_binary)

    handler = WiFaKeyHandler()
    decoder = NeuralMSOriginal(handler)  # decoder GỐC dùng chung cho cả 3 variant

    test_pairs, has_impostor = load_test_pairs_with_ids(
        cache_dir, pairs_dir, tier=args.tier
    )

    baseline_quantizer = lambda projected, intervals: (
        lssc_binary(projected.reshape(1, -1), interval=intervals)
        .flatten()
        .astype(np.uint8),
        np.zeros_like(projected.repeat(len(intervals))),
    )

    variants = [
        ("hard_bpsk", HardBPSK(), baseline_quantizer),
        (
            "soft_llr_v1",
            SoftDistanceLLR(scale=60.0, min_mag=0.1, max_mag=5.0),
            binarize_with_confidence,
        ),
        (
            "empirical_llr",
            EmpiricalLLR(lookup_path=_LOOKUP_PATH, masked_mag=MASKED_MAG),
            binarize_with_perbit_confidence,
        ),
    ]

    metrics, per_pair = run_all_variants_paired(handler, decoder, variants, test_pairs)

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        _RESULTS_DIR, f"paired_comparison_{args.dataset}_{args.tier}.json"
    )
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nĐã lưu kết quả vào: {out_path}")

    print("\n" + "=" * 78)
    print(
        f"SO SÁNH PAIRED TRÊN '{args.dataset}' / tier='{args.tier}' "
        f"({len(test_pairs)} cặp, cùng random_key/mask_r cho mọi variant)"
    )
    print("=" * 78)
    for name, _, _ in variants:
        m = metrics[name]
        exact_rate = m["genuine_success"] / max(m["genuine_total"], 1)
        far_str = (
            f"FAR={m['FAR']:.4f}" if has_impostor else "FAR=N/A (không có impostor)"
        )
        print(
            f"{name:16s} FRR={m['FRR']:.4f}  {far_str}  "
            f"genuine_success={m['genuine_success']}/{m['genuine_total']} "
            f"({exact_rate:.4f})"
        )

    genuine_mask = [p["is_genuine"] for p in test_pairs]

    print("\n--- McNemar's exact test (chỉ trên genuine pairs, đo FRR) ---")
    mcnemar_exact(
        per_pair["hard_bpsk"],
        per_pair["empirical_llr"],
        "hard_bpsk",
        "empirical_llr",
        subset_mask=genuine_mask,
    )
    mcnemar_exact(
        per_pair["soft_llr_v1"],
        per_pair["empirical_llr"],
        "soft_llr_v1",
        "empirical_llr",
        subset_mask=genuine_mask,
    )
    mcnemar_exact(
        per_pair["hard_bpsk"],
        per_pair["soft_llr_v1"],
        "hard_bpsk",
        "soft_llr_v1",
        subset_mask=genuine_mask,
    )

    if not has_impostor:
        print("\n(Dataset này chỉ có genuine pairs - bỏ qua đo FAR/McNemar impostor.)")
    elif any(metrics[n]["FAR"] > 0 for n, _, _ in variants):
        impostor_mask = [not g for g in genuine_mask]
        print("\n--- McNemar's exact test (impostor pairs, đo FAR) ---")
        mcnemar_exact(
            per_pair["hard_bpsk"],
            per_pair["empirical_llr"],
            "hard_bpsk",
            "empirical_llr",
            subset_mask=impostor_mask,
        )
        mcnemar_exact(
            per_pair["soft_llr_v1"],
            per_pair["empirical_llr"],
            "soft_llr_v1",
            "empirical_llr",
            subset_mask=impostor_mask,
        )
        mcnemar_exact(
            per_pair["hard_bpsk"],
            per_pair["soft_llr_v1"],
            "hard_bpsk",
            "soft_llr_v1",
            subset_mask=impostor_mask,
        )
    else:
        print("\n(FAR = 0 cho mọi variant - không cần McNemar's test cho impostor.)")


if __name__ == "__main__":
    main()
