"""
run_ab_soft_llr.py

So sánh CÔNG BẰNG giữa:
  - exp001_baseline        : quantizer gốc + v0_hard_bpsk
  - exp002_soft_llr_only   : quantizer_with_confidence + v1_soft_distance_llr

Chỉ đổi ĐÚNG 1 biến số (modulation) giữa 2 exp, mọi thứ khác giữ nguyên
(cùng handler, cùng decoder gốc, cùng test set genuine/impostor).

Cách chạy:
    python experiments/run_ab_soft_llr.py

Yêu cầu chuẩn bị trước (không đi kèm trong bộ này):
  - test_pairs: list các tuple (embedding_enroll, embedding_verify, is_genuine)
    lấy từ tập dữ liệu test thật (không dùng dữ liệu train/production thật
    khi commit code — chỉ dùng cho môi trường thử nghiệm nội bộ).
"""

import sys
import os
import json
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from research.quantizer.v0_lssc_with_confidence import (
    binarize_with_confidence,
    _selftest_against_original,
)
from research.modulation.v0_hard_bpsk import HardBPSK
from research.modulation.v1_soft_distance_llr import SoftDistanceLLR
from research.decoder.v0_neural_ms_original import NeuralMSOriginal
from research.pipeline.verify_variant import verify_with_variant

from research.decoder.v1_neural_ms_finetuned.decoder import NeuralMSFinetuned

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


def _load_embedding(name: str, imagenum) -> np.ndarray:
    path = os.path.join(_CACHE_DIR, f"{name}_{int(imagenum):04d}.npy")
    return np.load(path)


def load_test_pairs(tier: str = "select"):
    """
    tier="tune"   -> Tầng 1: dùng khi HIỆU CHỈNH tham số (κ, scale LLR...).
                     KHÔNG dùng tier này để kết luận version nào tốt hơn.
    tier="select" -> Tầng 2: dùng khi SO SÁNH giữa các version (exp001 vs exp002...).
                     Đây là tier mặc định cho A/B test này.
    tier="final"  -> Tầng 3: chỉ dùng đúng 1 lần, sau khi đã chốt version cuối cùng
                     (cần tự build data/processed/pairs/final_*.csv từ pairs.csv trước,
                     KHÔNG build sẵn trong scripts/02 để tránh vô tình đụng vào quá sớm).

    Yêu cầu đã chạy trước:
        python scripts/01_extract_embeddings.py
        python scripts/02_build_pairs_dataset.py

    Trả về list[(embedding_enroll, embedding_verify, is_genuine)]
    """
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


def main():
    from wifakey_module.wifakey_lib.utils import lssc_binary

    _selftest_against_original(lssc_binary)  # bắt buộc trước khi tin kết quả A/B

    handler = WiFaKeyHandler()  # baseline, KHÔNG sửa
    decoder = NeuralMSOriginal(handler)  # dùng chung 1 decoder gốc cho cả 2 exp
    # Dùng tier="select" (Tầng 2: matchpairsDevTest/mismatchpairsDevTest) để SO SÁNH
    # v0 vs v1. Nếu bạn cần tune scale trước, dùng tier="tune" ở một script riêng,
    # KHÔNG lẫn với script so sánh cuối này.
    test_pairs = load_test_pairs(tier="select")

    out_dir = os.path.join(_PROJECT_ROOT, "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)

    # exp001: baseline hoàn toàn
    baseline_quantizer = lambda projected, intervals: (
        lssc_binary(projected.reshape(1, -1), interval=intervals)
        .flatten()
        .astype(np.uint8),
        np.zeros_like(projected.repeat(len(intervals))),  # v0 không cần confidence thật
    )
    metrics_v0 = run_variant(
        handler, decoder, HardBPSK(), baseline_quantizer, test_pairs
    )
    with open(os.path.join(out_dir, "exp001_baseline.json"), "w") as f:
        json.dump(metrics_v0, f, indent=2)
    print("exp001_baseline:", metrics_v0)

    # exp002: CHỈ đổi modulation, giữ nguyên mọi thứ khác
    metrics_v1 = run_variant(
        handler,
        decoder,
        SoftDistanceLLR(scale=60.0, min_mag=0.1, max_mag=5.0),
        binarize_with_confidence,
        test_pairs,
    )
    with open(os.path.join(out_dir, "exp002_soft_llr_only.json"), "w") as f:
        json.dump(metrics_v1, f, indent=2)
    print("exp002_soft_llr_only:", metrics_v1)

    decoder_finetuned = NeuralMSFinetuned(handler)
    metrics_v2 = run_variant(
        handler,
        decoder_finetuned,
        SoftDistanceLLR(scale=60.0, min_mag=0.1, max_mag=5.0, masked_mag=1.0),
        binarize_with_confidence,
        test_pairs,
    )
    with open(os.path.join(out_dir, "exp003_soft_llr_finetuned.json"), "w") as f:
        json.dump(metrics_v2, f, indent=2)
    print("exp003_soft_llr_finetuned:", metrics_v2)

    print("\n=== So sánh ===")
    print(f"FRR: baseline={metrics_v0['FRR']:.4f}  soft_llr={metrics_v1['FRR']:.4f}")
    print(f"FAR: baseline={metrics_v0['FAR']:.4f}  soft_llr={metrics_v1['FAR']:.4f}")


if __name__ == "__main__":
    main()
