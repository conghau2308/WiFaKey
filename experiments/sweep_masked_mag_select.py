"""
sweep_masked_mag_select.py

QUÉT LẠI masked_mag TRỰC TIẾP TRÊN TẬP 'select' (mẫu lớn hơn nhiều so với
'tune' - 1694-1742 impostor pairs trên CPLFW so với 835 trên tune), vì kết
quả thực nghiệm cho thấy masked_mag=1.5 (chọn trên tune, FAR=0.0000/835)
LỘ RA rò rỉ FAR thật (0.46%, 8/1742) khi test trên mẫu impostor lớn hơn -
tune không đủ mẫu để phát hiện FAR nhỏ cỡ đó (0 sự kiện/835 mẫu vẫn khớp
với FAR thật lên tới ~0.36% ở mức tin cậy 95%, theo rule-of-three).

TIÊU CHÍ CHỌN (ưu tiên bảo mật - theo yêu cầu):
  KHÔNG dùng "FAR = 0 tuyệt đối" làm mục tiêu - ngay cả hard_bpsk (baseline
  đang dùng) cũng không đạt FAR=0 tuyệt đối trên CPLFW (0.11%, 2/1742) vì
  đây là dataset khó (pose chéo, một số cặp impostor tự nhiên rất giống
  nhau). Mục tiêu hợp lý: chọn masked_mag có FAR <= FAR của hard_bpsk (an
  toàn ít nhất bằng baseline đang dùng), rồi trong số đó chọn FRR thấp nhất.

Cách chạy (mỗi dataset chạy riêng, giống run_ab_paired.py):
    python experiments/sweep_masked_mag_select.py --dataset labeled_faces_in_the_wild
    python experiments/sweep_masked_mag_select.py --dataset cplfw

Sau khi có kết quả cả 2 dataset, gộp thủ công (hoặc dùng script riêng) để
ra quyết định cuối cùng - script này chỉ in/lưu kết quả từng dataset.
"""

import sys
import os
import csv
import json
import hashlib
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib.utils import lssc_binary
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
    _selftest_against_original as _selftest_v1,
)
from research.modulation.v0_hard_bpsk import HardBPSK
from research.modulation.v2_empirical_llr import EmpiricalLLR
from research.decoder.v0_neural_ms_original import NeuralMSOriginal
from research.pipeline.verify_variant import verify_with_variant

_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "experiments", "results")
_LOOKUP_PATH = os.path.join(
    _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
)

# Bao gom lai 1.5 de doi chieu (da biet no leak tren select), them cac gia
# tri thap hon de tim diem an toan. Co the chinh sua list nay.
MASKED_MAG_CANDIDATES = [0.75, 1.0, 1.1, 1.25, 1.5]

KNOWN_DATASETS = [
    "labeled_faces_in_the_wild",
    "face-detection-and-re-identification",
    "cplfw",
]


def get_dataset_paths(dataset_name: str):
    cache_dir = os.path.join(
        _PROJECT_ROOT, "datasets", "processed", dataset_name, "embeddings_cache"
    )
    pairs_dir = os.path.join(
        _PROJECT_ROOT, "datasets", "processed", dataset_name, "pairs"
    )
    return cache_dir, pairs_dir


def _load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    path = os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy")
    return np.load(path)


def stable_seed(name_enroll, imagenum_enroll, name_verify, imagenum_verify) -> int:
    key_str = f"{name_enroll}_{imagenum_enroll}_{name_verify}_{imagenum_verify}"
    digest = hashlib.sha256(key_str.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def load_test_pairs_with_ids(cache_dir, pairs_dir, tier="select"):
    genuine_path = os.path.join(pairs_dir, f"{tier}_genuine.csv")
    impostor_path = os.path.join(pairs_dir, f"{tier}_impostor.csv")
    if not os.path.exists(genuine_path):
        raise FileNotFoundError(f"Không tìm thấy {genuine_path}.")

    has_impostor = os.path.exists(impostor_path)
    sources = [(genuine_path, True)]
    if has_impostor:
        sources.append((impostor_path, False))
    else:
        print(f"[{tier}] Không có {impostor_path} - chỉ đo được FRR.")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="labeled_faces_in_the_wild", choices=KNOWN_DATASETS
    )
    parser.add_argument("--tier", default="select")
    args = parser.parse_args()

    print(f"=== Dataset: {args.dataset} | tier: {args.tier} ===")
    cache_dir, pairs_dir = get_dataset_paths(args.dataset)

    print("Self-check quantizer per-bit trước khi sweep...")
    _selftest_v1(lssc_binary)

    handler = WiFaKeyHandler()
    decoder = NeuralMSOriginal(handler)

    test_pairs, has_impostor = load_test_pairs_with_ids(
        cache_dir, pairs_dir, tier=args.tier
    )

    baseline_quantizer = lambda projected, intervals: (
        lssc_binary(projected.reshape(1, -1), interval=intervals)
        .flatten()
        .astype(np.uint8),
        np.zeros_like(projected.repeat(len(intervals))),
    )
    hard_bpsk = HardBPSK()

    empirical_variants = {
        mag: EmpiricalLLR(lookup_path=_LOOKUP_PATH, masked_mag=mag)
        for mag in MASKED_MAG_CANDIDATES
    }

    metrics = {
        "hard_bpsk": {
            "genuine_success": 0,
            "genuine_total": 0,
            "impostor_success": 0,
            "impostor_total": 0,
        },
    }
    for mag in MASKED_MAG_CANDIDATES:
        metrics[f"empirical_llr_mag{mag}"] = {
            "genuine_success": 0,
            "genuine_total": 0,
            "impostor_success": 0,
            "impostor_total": 0,
        }

    print(
        f"\nQuét {len(MASKED_MAG_CANDIDATES)} giá trị masked_mag {MASKED_MAG_CANDIDATES} "
        f"+ hard_bpsk, trên {len(test_pairs)} cặp select.\n"
    )

    for pair in test_pairs:
        # hard_bpsk: dung quantizer + seed rieng (nhu run_ab_paired.py), tai
        # su dung CUNG mot lan enroll() cho ca hard_bpsk lan cac ung vien
        # empirical (giong het seed -> giong het random_key/mask_r).
        np.random.seed(pair["seed"])
        helper_data, mask_r, key_hash = handler.enroll(pair["emb_enroll"])
        success_hard, _ = verify_with_variant(
            handler,
            baseline_quantizer,
            hard_bpsk,
            decoder,
            pair["emb_verify"],
            helper_data,
            mask_r,
            key_hash,
        )
        m = metrics["hard_bpsk"]
        if pair["is_genuine"]:
            m["genuine_total"] += 1
            m["genuine_success"] += int(success_hard)
        else:
            m["impostor_total"] += 1
            m["impostor_success"] += int(success_hard)

        # empirical_llr: tinh margin/quantizer 1 LAN, tai dung qua moi mag
        # (DUNG CUNG helper_data/mask_r/key_hash da tinh o tren - khong goi
        # enroll() lai, dam bao random_key giong het hard_bpsk).
        projected_v = np.dot(pair["emb_verify"], handler.M_matrix)
        bits_v, margin_v = binarize_with_perbit_confidence(
            projected_v, handler.intervals
        )
        b_selected_v = (bits_v.astype(np.uint8) & mask_r)[: handler.feature_length]
        margin_selected = margin_v[: handler.feature_length]
        mask_selected = mask_r[: handler.feature_length]
        y_noisy_bits = np.logical_xor(b_selected_v, helper_data)

        for mag in MASKED_MAG_CANDIDATES:
            llr = empirical_variants[mag](
                y_noisy_bits, context={"margin": margin_selected, "mask": mask_selected}
            )
            reconstructed_key = decoder.decode(llr)
            import hashlib as _h

            recon_hash = _h.sha256(reconstructed_key.tobytes()).digest()
            success = int(recon_hash == key_hash)

            m = metrics[f"empirical_llr_mag{mag}"]
            if pair["is_genuine"]:
                m["genuine_total"] += 1
                m["genuine_success"] += success
            else:
                m["impostor_total"] += 1
                m["impostor_success"] += success

    for name in metrics:
        m = metrics[name]
        m["FRR"] = 1 - m["genuine_success"] / max(m["genuine_total"], 1)
        m["FAR"] = m["impostor_success"] / max(m["impostor_total"], 1)

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        _RESULTS_DIR, f"sweep_masked_mag_select_{args.dataset}.json"
    )
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Đã lưu: {out_path}\n")

    print("=" * 86)
    print(f"KẾT QUẢ TRÊN 'select' ({args.dataset})")
    print("=" * 86)
    hard_far = metrics["hard_bpsk"]["FAR"]
    print(
        f"{'variant':>22s}  {'FRR':>8s}  {'FAR':>8s}  {'genuine':>14s}  {'impostor':>14s}  an_toan?"
    )
    for name in metrics:
        m = metrics[name]
        safe = "✅" if m["FAR"] <= hard_far else "❌ FAR CAO HƠN hard_bpsk"
        print(
            f"{name:>22s}  {m['FRR']:8.4f}  {m['FAR']:8.4f}  "
            f"{m['genuine_success']:6d}/{m['genuine_total']:<6d}  "
            f"{m['impostor_success']:6d}/{m['impostor_total']:<6d}  {safe}"
        )

    safe_candidates = [
        mag
        for mag in MASKED_MAG_CANDIDATES
        if metrics[f"empirical_llr_mag{mag}"]["FAR"] <= hard_far
    ]
    print(f"\nFAR của hard_bpsk (baseline, mốc an toàn tối thiểu) = {hard_far:.4f}")
    if safe_candidates:
        best_mag = min(
            safe_candidates, key=lambda mag: metrics[f"empirical_llr_mag{mag}"]["FRR"]
        )
        print(
            f"-> Các masked_mag ĐẠT mốc an toàn (FAR <= hard_bpsk): {safe_candidates}"
        )
        print(
            f"-> Trong số đó, masked_mag={best_mag} cho FRR thấp nhất "
            f"({metrics[f'empirical_llr_mag{best_mag}']['FRR']:.4f})."
        )
    else:
        print(
            "-> KHÔNG có masked_mag nào trong danh sách đạt FAR <= hard_bpsk trên dataset này "
            "- cần thử giá trị thấp hơn nữa (sửa MASKED_MAG_CANDIDATES)."
        )


if __name__ == "__main__":
    main()
