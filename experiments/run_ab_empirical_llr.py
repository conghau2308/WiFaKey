"""
run_ab_empirical_llr.py

So sánh trên tập `select` (Tầng 2 - dùng để SO SÁNH version, khác `tune` đã
dùng để hiệu chỉnh/fit empirical-LLR ở bước 2/3):

  - exp001_baseline           : quantizer gốc + v0_hard_bpsk           (đọc CACHE)
  - exp002_soft_llr_only      : v0_lssc_with_confidence + scale=60 LLR  (đọc CACHE)
  - exp004_empirical_llr      : v1_lssc_with_perbit_confidence (margin
                                 per-bit) + v2_empirical_llr (bảng hiệu
                                 chỉnh thực nghiệm) + decoder GỐC (CHƯA
                                 fine-tune - buoc 4 cho thay khong can thiet,
                                 xem lich su hoi thoai)                (CHẠY MỚI)

TIẾT KIỆM MÁY YẾU: exp001/exp002 đã được `run_ab_soft_llr.py` chạy và lưu
JSON trước đó (experiments/results/exp00X_*.json). Script này ĐỌC LẠI cache
đó thay vì chạy lại toàn bộ tập select qua decoder TF lần nữa - chỉ chạy
MỚI đúng 1 variant (exp004). Nếu không tìm thấy cache, in cảnh báo rõ ràng
và bỏ qua variant đó trong bảng so sánh (không tự ý chạy lại để tránh tốn
tài nguyên ngoài ý muốn - chạy `run_ab_soft_llr.py` trước nếu cần).

Cách chạy:
    python experiments/run_ab_empirical_llr.py
"""

import sys
import os
import json
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib.utils import lssc_binary
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
    _selftest_against_original,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR
from research.decoder.v0_neural_ms_original import NeuralMSOriginal
from research.pipeline.verify_variant import verify_with_variant

_CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)
_PAIRS_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
)
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "experiments", "results")
_LOOKUP_PATH = os.path.join(
    _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
)

MASKED_MAG = 1.0  # khop dung gia tri da dung khi fine-tune (train_empirical_llr.py)


def _load_embedding(name: str, imagenum) -> np.ndarray:
    path = os.path.join(_CACHE_DIR, f"{name}_{int(imagenum):04d}.npy")
    return np.load(path)


def load_test_pairs(tier: str = "select"):
    import csv as _csv

    genuine_path = os.path.join(_PAIRS_DIR, f"{tier}_genuine.csv")
    impostor_path = os.path.join(_PAIRS_DIR, f"{tier}_impostor.csv")
    for p in (genuine_path, impostor_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Không tìm thấy {p}. Chạy scripts/01_extract_embeddings.py rồi "
                f"scripts/02_build_pairs_dataset.py trước."
            )

    pairs = []
    for path, is_genuine in [(genuine_path, True), (impostor_path, False)]:
        with open(path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                emb_enroll = _load_embedding(row["name_enroll"], row["imagenum_enroll"])
                emb_verify = _load_embedding(row["name_verify"], row["imagenum_verify"])
                pairs.append((emb_enroll, emb_verify, is_genuine))

    print(
        f"[{tier}] Đã load {len(pairs)} cặp "
        f"({sum(1 for p in pairs if p[2])} genuine, "
        f"{sum(1 for p in pairs if not p[2])} impostor)"
    )
    return pairs


def run_variant(handler, decoder, modulation, quantizer_fn, test_pairs):
    results = {
        "genuine_success": 0,
        "genuine_total": 0,
        "impostor_success": 0,
        "impostor_total": 0,
    }  # impostor "success" = FALSE ACCEPT

    for emb_enroll, emb_verify, is_genuine in test_pairs:
        helper_data, mask_r, key_hash = handler.enroll(emb_enroll)

        success, _ = verify_with_variant(
            handler,
            quantizer_fn,
            modulation,
            decoder,
            emb_verify,
            helper_data,
            mask_r,
            key_hash,
        )

        if is_genuine:
            results["genuine_total"] += 1
            results["genuine_success"] += int(success)
        else:
            results["impostor_total"] += 1
            results["impostor_success"] += int(success)  # false accept nếu =1

    frr = 1 - results["genuine_success"] / max(results["genuine_total"], 1)
    far = results["impostor_success"] / max(results["impostor_total"], 1)
    return {"FRR": frr, "FAR": far, **results}


def _load_cached(name):
    path = os.path.join(_RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(
            f"[cảnh báo] Không tìm thấy cache {path} -> BỎ QUA '{name}' trong "
            f"bảng so sánh. Chạy `python experiments/run_ab_soft_llr.py` trước "
            f"nếu muốn có đủ baseline."
        )
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("Self-check quantizer per-bit trước khi chạy A/B (bắt buộc)...")
    _selftest_against_original(lssc_binary)

    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # --- Các variant ĐÃ CHẠY TRƯỚC (đọc cache, không tốn máy chạy lại) ---
    metrics_v0 = _load_cached("exp001_baseline")
    metrics_v1 = _load_cached("exp002_soft_llr_only")

    # --- Variant MỚI: empirical-LLR, decoder GỐC (chưa fine-tune) ---
    handler = WiFaKeyHandler()
    decoder = NeuralMSOriginal(
        handler
    )  # decoder GỐC - khớp quyết định "chưa cần fine-tune"
    test_pairs = load_test_pairs(tier="select")

    modulation = EmpiricalLLR(lookup_path=_LOOKUP_PATH, masked_mag=MASKED_MAG)
    metrics_v4 = run_variant(
        handler, decoder, modulation, binarize_with_perbit_confidence, test_pairs
    )
    with open(os.path.join(_RESULTS_DIR, "exp004_empirical_llr.json"), "w") as f:
        json.dump(metrics_v4, f, indent=2)
    print("exp004_empirical_llr:", metrics_v4)

    # --- Bảng so sánh ---
    print("\n" + "=" * 66)
    print(f"SO SÁNH TRÊN TẬP 'select' ({test_pairs and len(test_pairs)} cặp)")
    print("=" * 66)
    rows = [
        ("exp001_baseline (hard-BPSK)", metrics_v0),
        ("exp002_soft_llr_only (scale=60)", metrics_v1),
        ("exp004_empirical_llr (chưa fine-tune)", metrics_v4),
    ]
    for label, m in rows:
        if m is None:
            print(f"{label:42s} [không có dữ liệu]")
            continue
        exact_equiv = m["genuine_success"] / max(m["genuine_total"], 1)
        print(
            f"{label:42s} FRR={m['FRR']:.4f}  FAR={m['FAR']:.4f}  "
            f"genuine_success_rate={exact_equiv:.4f}"
        )

    if metrics_v0 is not None:
        print("\n--- Nhận định nhanh ---")
        exact_v4 = metrics_v4["genuine_success"] / max(metrics_v4["genuine_total"], 1)
        exact_v0 = metrics_v0["genuine_success"] / max(metrics_v0["genuine_total"], 1)
        if exact_v4 >= exact_v0 and metrics_v4["FAR"] <= metrics_v0["FAR"] * 1.5:
            print(
                "✅ empirical-LLR (chưa fine-tune) >= hard-BPSK trên select, FAR "
                "không tăng vọt -> có thể chốt version này, KHÔNG cần bước 4 "
                "fine-tune. Tiếp theo: đánh giá 1 lần trên tập 'final'."
            )
        elif exact_v4 >= exact_v0:
            print(
                "🟡 exact_match >= hard-BPSK nhưng FAR tăng đáng kể -> kiểm tra "
                "lại masked_mag / xem chi tiết impostor trước khi chốt."
            )
        else:
            print(
                "❌ empirical-LLR (chưa fine-tune) vẫn thua hard-BPSK trên select "
                "-> quay lại fine-tune (bước 4, train_empirical_llr.py) trước khi "
                "kết luận, hoặc kiểm tra xem select có phân phối khác tune không."
            )


if __name__ == "__main__":
    main()
