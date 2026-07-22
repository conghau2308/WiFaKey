"""
sweep_masked_mag.py

TỐI ƯU masked_mag cho v2_empirical_llr TRÊN TẬP 'tune' (Tầng 1 - hiệu chỉnh
tham số, ĐÚNG NHƯ κ/scale trước đây - KHÔNG dùng select/final cho việc này).

Bối cảnh: masked_mag là biên độ LLR gán cho các bit bị mask (mask_r=0) - vị
trí mà y_noisy = helper_data BIẾT TRƯỚC TẤT ĐỊNH (không phụ thuộc embedding
người dùng), khác hẳn ý nghĩa "độ tin cậy" của margin thật. Giá trị này
CHƯA TỪNG được tối ưu cho empirical_llr - v1_soft_distance_llr cũ để lại
cảnh báo rõ: dùng max_mag (quá tự tin) cho các bit này từng gây FAR=39.2%
(decoder hội tụ đúng "key nhìn như đúng" cho CẢ kẻ mạo danh, vì các bit đó
không phân biệt được genuine/impostor). Sweep này tìm điểm cân bằng
FRR thấp nhất MÀ KHÔNG làm FAR tăng so với mức hiện có (0% trên tune).

PHƯƠNG PHÁP - paired sweep (công bằng, đỡ tốn máy):
  - Với MỖI cặp (enroll, verify), phần "chung" (random_key, mask_r, margin,
    y_noisy_bits) CHỈ tính 1 LẦN (không phụ thuộc masked_mag) - cache lại.
  - Chỉ phần modulation (numpy, rẻ) + decode (TF, đắt - không tránh được vì
    LLR tại vị trí mask đổi theo masked_mag) chạy LẶP LẠI cho từng ứng viên
    masked_mag, dùng CHUNG random_key/mask_r/margin đã cache -> so sánh công
    bằng giữa các ứng viên, giảm redundant computation so với chạy từng
    ứng viên độc lập từ đầu.
  - Seed mỗi cặp suy từ hash ổn định (giống run_ab_paired.py), để kết quả
    tái lập được và không phụ thuộc thứ tự dòng CSV.

Decoder dùng NeuralMSOriginal (GỐC, CHƯA fine-tune) - khớp quyết định đã
chốt trước đó (fine-tune 20 epoch không cải thiện, xem lịch sử hội thoại).

Cách chạy:
    python experiments/sweep_masked_mag.py

Muốn sweep nhanh hơn (máy yếu): giảm MASKED_MAG_CANDIDATES hoặc đặt
MAX_PAIRS xuống 1 số nhỏ để test thử trước khi chạy full.
"""

import sys
import os
import csv
import json
import hashlib
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

import hashlib as _hashlib

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

# Bien do can thu: bao trum ca duoi 1.0 (kem tu tin hon "trung tinh" cu) lan
# tren 1.0 (tu tin hon), nhung KHONG cham gan max cua thang do (~5.3, xem
# v2_empirical_llr eps=0.005) de tranh lap lai loi overconfidence da gay
# FAR=39.2% o v1. Chinh sua list nay neu muon quet mien khac / it gia tri
# hon (do nhanh hon tren may yeu).
MASKED_MAG_CANDIDATES = [0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5]

MAX_PAIRS = None  # None = dung het tap tune; dat so nho de test nhanh truoc


def _load_embedding(name: str, imagenum) -> np.ndarray:
    path = os.path.join(_CACHE_DIR, f"{name}_{int(imagenum):04d}.npy")
    return np.load(path)


def stable_seed(name_enroll, imagenum_enroll, name_verify, imagenum_verify) -> int:
    """Giong het run_ab_paired.py - seed on dinh tu hash, khong phu thuoc
    thu tu dong CSV, tai lap 100% giua cac lan chay."""
    key_str = f"{name_enroll}_{imagenum_enroll}_{name_verify}_{imagenum_verify}"
    digest = _hashlib.sha256(key_str.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def load_tune_pairs_with_ids():
    """tune_genuine.csv + tune_impostor.csv - dung TOAN BO (khong tach train/
    val noi bo nhu train_empirical_llr.py, vi o day KHONG fit tham so tren
    margin, chi so sanh FRR/FAR truc tiep - khong co nguy co overfit theo
    nghia do)."""
    genuine_path = os.path.join(_PAIRS_DIR, "tune_genuine.csv")
    impostor_path = os.path.join(_PAIRS_DIR, "tune_impostor.csv")
    for p in (genuine_path, impostor_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy {p}.")

    pairs = []
    for path, is_genuine in [(genuine_path, True), (impostor_path, False)]:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emb_enroll = _load_embedding(row["name_enroll"], row["imagenum_enroll"])
                emb_verify = _load_embedding(row["name_verify"], row["imagenum_verify"])
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

    if MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]

    print(
        f"[tune] Đã load {len(pairs)} cặp "
        f"({sum(1 for p in pairs if p['is_genuine'])} genuine, "
        f"{sum(1 for p in pairs if not p['is_genuine'])} impostor)"
    )
    return pairs


def compute_common_parts(handler, quantizer_fn, pair):
    """Phan KHONG phu thuoc masked_mag - tinh 1 LAN cho moi cap, tai su dung
    qua tat ca ung vien masked_mag (tiet kiem may yeu)."""
    np.random.seed(pair["seed"])
    helper_data, mask_r, key_hash = handler.enroll(pair["emb_enroll"])

    projected_v = np.dot(pair["emb_verify"], handler.M_matrix)
    bits_v, margin_v = quantizer_fn(projected_v, handler.intervals)
    b_selected_v = (bits_v.astype(np.uint8) & mask_r)[: handler.feature_length]
    margin_selected = margin_v[: handler.feature_length]
    mask_selected = mask_r[: handler.feature_length]

    y_noisy_bits = np.logical_xor(b_selected_v, helper_data)
    return y_noisy_bits, margin_selected, mask_selected, key_hash


def run_sweep(handler, decoder, quantizer_fn, candidates, pairs):
    modulations = {
        mag: EmpiricalLLR(lookup_path=_LOOKUP_PATH, masked_mag=mag)
        for mag in candidates
    }

    metrics = {
        mag: {
            "genuine_success": 0,
            "genuine_total": 0,
            "impostor_success": 0,
            "impostor_total": 0,
        }
        for mag in candidates
    }
    import hashlib as _h

    for pair in pairs:
        y_noisy_bits, margin_selected, mask_selected, key_hash = compute_common_parts(
            handler, quantizer_fn, pair
        )
        for mag in candidates:
            llr = modulations[mag](
                y_noisy_bits, context={"margin": margin_selected, "mask": mask_selected}
            )
            reconstructed_key = decoder.decode(llr)
            recon_hash = _h.sha256(reconstructed_key.tobytes()).digest()
            success = int(recon_hash == key_hash)

            m = metrics[mag]
            if pair["is_genuine"]:
                m["genuine_total"] += 1
                m["genuine_success"] += success
            else:
                m["impostor_total"] += 1
                m["impostor_success"] += success  # false accept neu =1

    for mag in candidates:
        m = metrics[mag]
        m["FRR"] = 1 - m["genuine_success"] / max(m["genuine_total"], 1)
        m["FAR"] = m["impostor_success"] / max(m["impostor_total"], 1)

    return metrics


def main():
    print("Self-check quantizer per-bit trước khi sweep (bắt buộc)...")
    _selftest_against_original(lssc_binary)

    handler = WiFaKeyHandler()
    decoder = NeuralMSOriginal(handler)  # decoder GỐC, chưa fine-tune

    pairs = load_tune_pairs_with_ids()

    print(
        f"\nSweep masked_mag qua {len(MASKED_MAG_CANDIDATES)} giá trị: {MASKED_MAG_CANDIDATES}"
    )
    print(
        f"Tổng số lần decode: {len(pairs)} cặp × {len(MASKED_MAG_CANDIDATES)} giá trị "
        f"= {len(pairs) * len(MASKED_MAG_CANDIDATES)} lần (phần margin/enroll chỉ tính 1 lần/cặp).\n"
    )

    metrics = run_sweep(
        handler, decoder, binarize_with_perbit_confidence, MASKED_MAG_CANDIDATES, pairs
    )

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    with open(os.path.join(_RESULTS_DIR, "sweep_masked_mag_tune.json"), "w") as f:
        json.dump({str(k): v for k, v in metrics.items()}, f, indent=2)

    print("=" * 78)
    print("KẾT QUẢ SWEEP masked_mag TRÊN TẬP 'tune'")
    print("=" * 78)
    print(
        f"{'masked_mag':>10s}  {'FRR':>8s}  {'FAR':>8s}  {'genuine':>14s}  {'impostor':>14s}"
    )
    for mag in MASKED_MAG_CANDIDATES:
        m = metrics[mag]
        print(
            f"{mag:10.2f}  {m['FRR']:8.4f}  {m['FAR']:8.4f}  "
            f"{m['genuine_success']:6d}/{m['genuine_total']:<6d}  "
            f"{m['impostor_success']:6d}/{m['impostor_total']:<6d}"
        )

    min_far = min(metrics[mag]["FAR"] for mag in MASKED_MAG_CANDIDATES)
    safe_candidates = [
        mag for mag in MASKED_MAG_CANDIDATES if metrics[mag]["FAR"] <= min_far
    ]
    best_mag = min(safe_candidates, key=lambda mag: metrics[mag]["FRR"])

    print(f"\n-> FAR thấp nhất đạt được trong sweep: {min_far:.4f}")
    print(
        f"-> Trong số các masked_mag đạt FAR thấp nhất đó, masked_mag={best_mag} "
        f"cho FRR thấp nhất ({metrics[best_mag]['FRR']:.4f})."
    )
    print(
        f"   Giá trị mặc định hiện tại là masked_mag=1.0 "
        f"(FRR={metrics.get(1.0, {}).get('FRR', float('nan')):.4f} nếu có trong danh sách)."
    )
    print(
        "\nBước tiếp theo: cập nhật MASKED_MAG mặc định trong "
        "research/modulation/v2_empirical_llr.py và train_empirical_llr.py "
        "thành giá trị tốt nhất tìm được, RỒI đánh giá lại 1 lần trên tập "
        "'select' (run_ab_paired.py) để xác nhận cải thiện có giữ được "
        "trên dữ liệu chưa từng dùng để tune masked_mag."
    )


if __name__ == "__main__":
    main()
