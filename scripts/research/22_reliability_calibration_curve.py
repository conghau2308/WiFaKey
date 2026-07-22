#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_reliability_calibration_curve.py
===================================================================
BƯỚC 2 — Thí nghiệm QUYẾT ĐỊNH go/no-go cho hướng soft-LLR.

Câu hỏi: "Độ tin cậy" mà soft-LLR gán cho mỗi bit (|khoảng cách tới ngưỡng
lượng tử| của mẫu VERIFY) có THỰC SỰ dự báo được xác suất bit đó bị lật?

  - Đường empirical-BER giảm đơn điệu MẠNH theo độ tin cậy  -> GO (bước 3).
  - Đường PHẲNG                                             -> NO-GO.

Nguyên tắc (mục 3 tài liệu nghiên cứu):
  - CHỈ dùng tập `tune`. Không chạm `select`/`final`.
  - Lấy bit bằng CHÍNH `utils.lssc_binary` GỐC (không tự chế quantizer).
  - self_check xác nhận công thức margin/bit suy ra từ lssc_binary KHỚP
    BIT-FOR-BIT với gốc trước khi tin reliability (nguyên tắc 3.4).

Công thức bit (suy trực tiếp từ utils.lssc_binary, không dò bằng thử-sai):
  lkut là mã nhiệt kế (thermometer code) nhưng bit CUỐI trong mỗi block
  bật lên trước:
      lkut[0] = [0,0,0]   ( v <  t0 )
      lkut[1] = [0,0,1]   ( t0 <= v < t1 )
      lkut[2] = [0,1,1]   ( t1 <= v < t2 )
      lkut[3] = [1,1,1]   ( v >= t2 )
  => với block-local index j (0-based, dim-major), bit_j = 1 iff
     v >= thr[N_THR-1-j]   (ngưỡng ĐẢO NGƯỢC trong mỗi block, so sánh >=).
===================================================================
"""

import os
import sys
import csv
import numpy as np

# ------------------------------------------------------------------
# 0. CẤU HÌNH ĐƯỜNG DẪN  (ADAPT nếu repo khác)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)  # để import wifakey_module

LFW_DIR = os.path.join(
    PROJECT_ROOT, "datasets", "processed", "labeled_faces_in_the_wild"
)
EMB_CACHE = os.path.join(LFW_DIR, "embeddings_cache")
PAIRS_DIR = os.path.join(LFW_DIR, "pairs")

# M_matrix / intervals nằm trong data_path của handler gốc:
DATA_PATH = os.path.join(PROJECT_ROOT, "wifakey_module", "data")
M_MATRIX_NPY = os.path.join(DATA_PATH, "M_matrix.npy")
INTERVALS_NPY = os.path.join(DATA_PATH, "binarization_intervals.npy")

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "out_step2")
os.makedirs(OUT_DIR, exist_ok=True)

N_BINS = 10
MAX_PAIRS = None  # None = dùng hết; đặt số nhỏ để test nhanh

# ------------------------------------------------------------------
# 1. LẤY BIT BẰNG lssc_binary GỐC + chuẩn bị margin để dò reliability
# ------------------------------------------------------------------
from wifakey_module.wifakey_lib import utils  # dùng ĐÚNG hàm gốc

M_matrix = np.load(M_MATRIX_NPY)
intervals = np.load(INTERVALS_NPY)  # truyền NGUYÊN cho lssc_binary
thr = np.sort(np.asarray(intervals).reshape(-1))  # ngưỡng tăng dần (để tính margin)
rev_thr = thr[::-1]  # ngưỡng đảo ngược — dùng trực tiếp trong _candidate
N_THR = thr.size
print(
    f"[load] M_matrix={M_matrix.shape}  intervals={np.asarray(intervals).shape}  N_THR={N_THR}"
)


def real_binarize(emb):
    """Bit THẬT — sao chép đúng WiFaKeyHandler._binarize_full()."""
    projected = np.dot(np.asarray(emb, dtype=np.float64), M_matrix)  # emb @ M
    return (
        utils.lssc_binary(projected[None, :], interval=intervals)
        .flatten()
        .astype(np.uint8)
    )


def _candidate(v):
    """Sinh (bit, margin) đúng theo công thức suy từ lssc_binary:
    bit_j (dim-major, block-local j) = 1 iff v >= thr[N_THR-1-j]
    (thermometer code với thứ tự ngưỡng đảo ngược trong mỗi block, so sánh >=).
    margin(d,j) = |v_d - thr_reversed_j|."""
    cmp = v[:, None] >= rev_thr[None, :]  # (D, N_THR)
    margin = np.abs(v[:, None] - rev_thr[None, :])
    # dim-major: [d0_j0,d0_j1,d0_j2, d1_j0,...] — khớp new_data[i*block:(i+1)*block]
    return cmp.astype(np.uint8).reshape(-1), margin.reshape(-1)


# ------------------------------------------------------------------
# 2. SELF-CHECK: xác nhận công thức khớp BIT-FOR-BIT với bản gốc
# ------------------------------------------------------------------
def self_check_against_original(pairs, n_probe=8):
    """Kiểm tra công thức thermometer-reversed trên vài mẫu.
    Nếu không khớp -> DỪNG, cần xem lại utils.lssc_binary (có thể N_THR/lkut đổi)."""
    embs = [p[0] for p in pairs[:n_probe]] + [p[1] for p in pairs[:n_probe]]
    for emb in embs:
        v = np.dot(np.asarray(emb, np.float64), M_matrix)
        cb, _ = _candidate(v)
        rb = real_binarize(emb)
        if cb.shape != rb.shape or not np.array_equal(cb, rb):
            print(
                "[self-check] ❌ FAIL — công thức thermometer-reversed KHÔNG khớp bit gốc."
            )
            print(f"    len(candidate)={cb.size}  len(real)={rb.size}")
            print(f"    real[:12]      ={rb[:12].tolist()}")
            print(f"    candidate[:12] ={cb[:12].tolist()}")
            print("    -> Kiểm tra lại utils.lssc_binary (lkut/N_THR có thể đã đổi).")
            raise SystemExit(1)
    print(
        f"[self-check] ✅ PASS — bit tái dựng KHỚP lssc_binary gốc trên {len(embs)} mẫu."
    )


def bits_and_reliability(emb):
    """Trả (bits_gốc, reliability) theo công thức thermometer-reversed đã xác nhận."""
    v = np.dot(np.asarray(emb, np.float64), M_matrix)
    _, rel = _candidate(v)
    return real_binarize(emb), rel  # bit = bit GỐC (đảm bảo đúng tuyệt đối)


# ------------------------------------------------------------------
# 3. LOADER CẶP GENUINE  (HÀM CỦA BẠN)
# ------------------------------------------------------------------
def load_embedding(name, imagenum):
    """HÀM CỦA BẠN — ADAPT format tên file .npy cho khớp cache của bạn."""
    p = os.path.join(EMB_CACHE, f"{name}_{int(imagenum):04d}.npy")  # ADAPT nếu khác
    return np.load(p)


def load_tune_genuine_pairs(val_fraction=0.15, seed=123):
    path = os.path.join(PAIRS_DIR, "tune_genuine.csv")
    pairs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emb1 = load_embedding(row["name_enroll"], row["imagenum_enroll"])
            emb2 = load_embedding(row["name_verify"], row["imagenum_verify"])
            pairs.append((emb1, emb2))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(int(len(pairs) * val_fraction), 1)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


# ------------------------------------------------------------------
# 4. THU THẬP (reliability_verify , flip_indicator) TRÊN MỌI BIT
# ------------------------------------------------------------------
def collect(pairs):
    if MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
    rel_all, flip_all, n_ok = [], [], 0
    for e_enroll, e_verify in pairs:
        b_en, _ = bits_and_reliability(e_enroll)
        b_ve, rel_ve = bits_and_reliability(e_verify)
        flip = (b_en != b_ve).astype(np.uint8)  # e_i = b_enroll XOR b_verify
        rel_all.append(rel_ve)
        flip_all.append(flip)  # reliability của mẫu VERIFY
        # đối xứng: đổi vai để nhân đôi dữ liệu
        _, rel_en = bits_and_reliability(e_enroll)
        rel_all.append(rel_en)
        flip_all.append(flip)
        n_ok += 1
    if n_ok == 0:
        raise RuntimeError("Không nạp được cặp nào — kiểm tra load_embedding().")
    print(
        f"[collect] dùng {n_ok} cặp genuine, "
        f"tổng {sum(len(x) for x in rel_all):,} quan sát bit."
    )
    return np.concatenate(rel_all), np.concatenate(flip_all)


# ------------------------------------------------------------------
# 5. PHÂN TÍCH + QUYẾT ĐỊNH
# ------------------------------------------------------------------
def analyse(rel, flip):
    import pandas as pd

    edges = np.quantile(rel, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(rel, edges[1:-1]), 0, N_BINS - 1)

    rows = []
    for b in range(N_BINS):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            dict(
                bin=b,
                rel_mean=float(rel[m].mean()),
                rel_lo=float(rel[m].min()),
                rel_hi=float(rel[m].max()),
                n=int(m.sum()),
                emp_BER=float(flip[m].mean()),
            )
        )
    stats = pd.DataFrame(rows)

    from scipy.stats import spearmanr

    rho, p = spearmanr(rel, flip)
    ber_low = stats.iloc[0]["emp_BER"]  # bin ÍT tin cậy
    ber_high = stats.iloc[-1]["emp_BER"]  # bin TIN CẬY cao
    ratio = ber_low / max(ber_high, 1e-9)

    print("\n================  BẢNG HIỆU CHỈNH ĐỘ TIN CẬY  ================")
    print(
        stats.to_string(
            index=False,
            formatters={
                "rel_mean": "{:.4f}".format,
                "rel_lo": "{:.4f}".format,
                "rel_hi": "{:.4f}".format,
                "emp_BER": "{:.4f}".format,
            },
        )
    )
    print("==============================================================")
    print(f"Spearman(rel, flip) = {rho:+.4f}  (p={p:.1e})   [kỳ vọng: ÂM mạnh]")
    print(f"BER bin ÍT-tin-cậy  = {ber_low:.4f}")
    print(f"BER bin TIN-CẬY-cao = {ber_high:.4f}")
    print(f"Tỷ lệ phân tách     = {ratio:.2f}×   [so mốc 1.31× ở mục 6.2]")

    print("\n---------------------  KẾT LUẬN GO / NO-GO  --------------------")
    if rho <= -0.15 and ratio >= 2.0:
        verdict = (
            "✅ GO — độ tin cậy per-bit DỰ BÁO TỐT xác suất lỗi. Sang BƯỚC 3: "
            "bỏ công thức tay, dùng EMPIRICAL-LLR  LLR_i=log((1-p_i)/p_i)  "
            "theo bin, rồi fine-tune decoder trên đúng phân phối input đó."
        )
    elif ratio >= 1.5:
        verdict = (
            "🟡 YẾU/BIÊN GIỚI — có tín hiệu nhưng nhỏ. Thử empirical-LLR được "
            "nhưng kỳ vọng cải thiện hạn chế; nhiều khả năng phải nâng cấp "
            "UPSTREAM (quantizer margin-equalizing)."
        )
    else:
        verdict = (
            f"❌ NO-GO — độ tin cậy per-bit KHÔNG predictive (~{ratio:.2f}×, xấp xỉ "
            "mốc 1.31×). Đổi công thức/scale/sigma hay retrain decoder đều vô ích. "
            "DỪNG soft-LLR, ghi nhận kết quả âm tính, pivot lên UPSTREAM (redesign "
            "quantizer) hoặc sang trục cải tiến khác."
        )
    print(verdict)
    print("----------------------------------------------------------------")

    stats.to_csv(os.path.join(OUT_DIR, "reliability_bins.csv"), index=False)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        plt.plot(stats["rel_mean"], stats["emp_BER"], "o-", lw=2)
        plt.axhline(
            flip.mean(), ls="--", c="gray", label=f"BER tổng = {flip.mean():.3f}"
        )
        plt.xlabel("Độ tin cậy trung bình mỗi bin (|khoảng cách tới ngưỡng|)")
        plt.ylabel("Xác suất lật bit thực nghiệm (empirical BER)")
        plt.title(f"Reliability calibration | Spearman={rho:+.3f} ratio={ratio:.2f}×")
        plt.grid(alpha=0.3)
        plt.legend()
        png = os.path.join(OUT_DIR, "reliability_calibration_curve.png")
        plt.savefig(png, dpi=130, bbox_inches="tight")
        print(f"[out] biểu đồ -> {png}")
    except Exception as ex:
        print(f"[warn] không vẽ được biểu đồ: {ex}")
    return stats, rho, ratio


# ------------------------------------------------------------------
def main():
    print("=" * 66)
    print("BƯỚC 2 — Reliability calibration curve (go/no-go cho soft-LLR)")
    print("=" * 66)
    train_pairs, val_pairs = load_tune_genuine_pairs()  # HÀM CỦA BẠN
    pairs = train_pairs + val_pairs  # diagnostic -> dùng hết
    print(
        f"[pairs] train={len(train_pairs)}  val={len(val_pairs)}  "
        f"-> dùng {len(pairs)} cặp cho calibration"
    )
    self_check_against_original(pairs)  # BẮT BUỘC pass trước khi tin kết quả
    rel, flip = collect(pairs)
    analyse(rel, flip)


if __name__ == "__main__":
    main()
