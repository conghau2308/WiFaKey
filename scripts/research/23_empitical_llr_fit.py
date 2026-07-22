#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_empirical_llr_fit.py
===================================================================
BƯỚC 3 — Empirical-LLR calibration (thay thế công thức tay scale=60).

Tiền đề: BƯỚC 2 đã cho kết quả GO
  (Spearman = -0.35, ratio = 22.47x trên toàn bộ tune).

Việc ở đây:
  1. FIT một hàm hiệu chỉnh  margin -> p(lật bit)  CHỈ trên `tune_train`
     (không đụng `tune_val`, `select`, `final`).
  2. Hàm hiệu chỉnh là ISOTONIC REGRESSION (ép đơn điệu không-tăng: margin
     càng lớn thì p càng nhỏ) — khớp đúng bản chất vật lý của bài toán,
     tổng quát hơn nhiều so với việc chỉ dùng 10 bin thô như BƯỚC 2.
     Cài đặt bằng thuật toán PAVA (Pool Adjacent Violators) thuần numpy,
     KHÔNG cần sklearn, chạy trên bin theo quantile (mặc định 200 bin)
     để vừa mịn vừa nhanh trên hàng triệu quan sát bit.
  3. VALIDATE trên `tune_val` (tách riêng, không dùng để fit) — so đường
     dự đoán margin_to_p() với BER thực nghiệm trên val, để chắc chắn
     đường cong không overfit vào 749 cặp train.
  4. Xuất LLR:  LLR(margin) = log((1 - p) / p),  p đã clip vào [eps, 0.5-eps]
     để tránh log(0) / log(vô cực). Biên độ LLR sẽ tự nhiên nhỏ
     (đúng như chẩn đoán: spread thật ~1.1-1.5, không phải 60).
  5. GHI RA:
       - experiments/out_step3/reliability_lookup.npz   (bảng breakpoint)
       - experiments/out_step3/empirical_llr.py          (module cắm thẳng
         vào pipeline: from empirical_llr import margin_to_llr)
       - experiments/out_step3/calibration_train_vs_val.png

  Dùng lại NGUYÊN VẸN công thức bit + self-check của BƯỚC 2
  (thermometer-reversed, đã xác nhận khớp bit-for-bit với lssc_binary gốc).
===================================================================
"""

import os
import sys
import csv
import numpy as np

# ------------------------------------------------------------------
# 0. CẤU HÌNH ĐƯỜNG DẪN  (ADAPT nếu repo khác — giống hệt bước 2)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

LFW_DIR = os.path.join(
    PROJECT_ROOT, "datasets", "processed", "labeled_faces_in_the_wild"
)
EMB_CACHE = os.path.join(LFW_DIR, "embeddings_cache")
PAIRS_DIR = os.path.join(LFW_DIR, "pairs")

DATA_PATH = os.path.join(PROJECT_ROOT, "wifakey_module", "data")
M_MATRIX_NPY = os.path.join(DATA_PATH, "M_matrix.npy")
INTERVALS_NPY = os.path.join(DATA_PATH, "binarization_intervals.npy")

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "out_step3")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_PAIRS = None  # None = dùng hết; đặt số nhỏ để test nhanh
N_FIT_BINS = 200  # số bin quantile dùng để FIT isotonic (càng lớn càng mịn)
N_REPORT_BINS = 10  # số bin dùng để IN BẢNG so sánh train/val (dễ đọc)
EPS_P = 0.005  # clip p vào [EPS_P, 0.5 - EPS_P] trước khi tính LLR

# ------------------------------------------------------------------
# 1. CÔNG THỨC BIT (giống hệt bước 2 — đã self-check PASS)
# ------------------------------------------------------------------
from wifakey_module.wifakey_lib import utils  # dùng ĐÚNG hàm gốc

M_matrix = np.load(M_MATRIX_NPY)
intervals = np.load(INTERVALS_NPY)
thr = np.sort(np.asarray(intervals).reshape(-1))
rev_thr = thr[::-1]
N_THR = thr.size
print(
    f"[load] M_matrix={M_matrix.shape}  intervals={np.asarray(intervals).shape}  N_THR={N_THR}"
)


def real_binarize(emb):
    """Bit THẬT — sao chép đúng WiFaKeyHandler._binarize_full()."""
    projected = np.dot(np.asarray(emb, dtype=np.float64), M_matrix)
    return (
        utils.lssc_binary(projected[None, :], interval=intervals)
        .flatten()
        .astype(np.uint8)
    )


def _candidate(v):
    """bit_j (dim-major) = 1 iff v >= thr[N_THR-1-j]  (thermometer đảo ngược, >=)."""
    cmp = v[:, None] >= rev_thr[None, :]
    margin = np.abs(v[:, None] - rev_thr[None, :])
    return cmp.astype(np.uint8).reshape(-1), margin.reshape(-1)


def self_check_against_original(pairs, n_probe=8):
    embs = [p[0] for p in pairs[:n_probe]] + [p[1] for p in pairs[:n_probe]]
    for emb in embs:
        v = np.dot(np.asarray(emb, np.float64), M_matrix)
        cb, _ = _candidate(v)
        rb = real_binarize(emb)
        if cb.shape != rb.shape or not np.array_equal(cb, rb):
            print(
                "[self-check] ❌ FAIL — công thức thermometer-reversed KHÔNG khớp bit gốc."
            )
            print(f"    real[:12]      ={rb[:12].tolist()}")
            print(f"    candidate[:12] ={cb[:12].tolist()}")
            raise SystemExit(1)
    print(
        f"[self-check] ✅ PASS — bit tái dựng KHỚP lssc_binary gốc trên {len(embs)} mẫu."
    )


def bits_and_reliability(emb):
    v = np.dot(np.asarray(emb, np.float64), M_matrix)
    _, rel = _candidate(v)
    return real_binarize(emb), rel


# ------------------------------------------------------------------
# 2. LOADER CẶP GENUINE  (giống bước 2 — cùng seed=123 để tái lập đúng
#    train/val split đã dùng ở bước 2)
# ------------------------------------------------------------------
def load_embedding(name, imagenum):
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


def collect(pairs):
    """(margin, flip) cho mọi bit, cả hai chiều enroll/verify — giống bước 2."""
    if MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
    rel_all, flip_all = [], []
    for e_enroll, e_verify in pairs:
        b_en, rel_en = bits_and_reliability(e_enroll)
        b_ve, rel_ve = bits_and_reliability(e_verify)
        flip = (b_en != b_ve).astype(np.uint8)
        rel_all.append(rel_ve)
        flip_all.append(flip)
        rel_all.append(rel_en)
        flip_all.append(flip)
    if not rel_all:
        raise RuntimeError("Không nạp được cặp nào — kiểm tra load_embedding().")
    return np.concatenate(rel_all), np.concatenate(flip_all)


# ------------------------------------------------------------------
# 3. ISOTONIC REGRESSION (PAVA thuần numpy, không cần sklearn)
# ------------------------------------------------------------------
def pava_decreasing(y, w):
    """Pool-Adjacent-Violators cho ràng buộc KHÔNG TĂNG (y giảm dần theo x).
    y, w : mảng trung bình / trọng số mỗi bin, đã sắp theo x tăng dần.
    Trả về mảng fitted cùng độ dài, đơn điệu không tăng."""
    blocks = []  # mỗi block: [tổng-có-trọng-số, tổng-trọng-số, số-bin-gộp]
    for yi, wi in zip(y, w):
        blocks.append([yi * wi, wi, 1])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) < (
            blocks[-1][0] / blocks[-1][1]
        ):
            b2 = blocks.pop()
            b1 = blocks.pop()
            blocks.append([b1[0] + b2[0], b1[1] + b2[1], b1[2] + b2[2]])
    fitted = []
    for sw, wsum, cnt in blocks:
        fitted.extend([sw / wsum] * cnt)
    return np.array(fitted)


def fit_isotonic_calibration(margin, flip, n_bins=N_FIT_BINS):
    """Bin theo quantile của margin (~equal-count) rồi ép đơn điệu bằng PAVA.
    Trả về (bin_centers, fitted_p) đã sắp margin tăng dần, fitted_p không tăng."""
    order = np.argsort(margin)
    margin_sorted = margin[order]
    flip_sorted = flip[order]

    edges = np.quantile(margin_sorted, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(margin_sorted, edges[1:-1]), 0, n_bins - 1)

    centers, means, weights = [], [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        centers.append(margin_sorted[m].mean())
        means.append(flip_sorted[m].mean())
        weights.append(int(m.sum()))

    centers = np.array(centers)
    means = np.array(means)
    weights = np.array(weights, dtype=np.float64)

    fitted_p = pava_decreasing(means, weights)
    return centers, fitted_p


def margin_to_p_from_table(margin, bp_margin, bp_p, eps=EPS_P):
    margin = np.asarray(margin, dtype=np.float64)
    p = np.interp(margin, bp_margin, bp_p, left=bp_p[0], right=bp_p[-1])
    return np.clip(p, eps, 0.5 - eps)


def margin_to_llr_from_table(margin, bp_margin, bp_p, eps=EPS_P):
    p = margin_to_p_from_table(margin, bp_margin, bp_p, eps=eps)
    return np.log((1.0 - p) / p)


# ------------------------------------------------------------------
# 4. VALIDATION: so dự đoán (fit trên train) với thực nghiệm trên val
# ------------------------------------------------------------------
def validate_on_val(margin_val, flip_val, bp_margin, bp_p, n_bins=N_REPORT_BINS):
    import pandas as pd
    from scipy.stats import spearmanr

    p_pred = margin_to_p_from_table(margin_val, bp_margin, bp_p)

    # Brier score: dự đoán calibrated vs baseline "hằng số = BER trung bình"
    brier_model = float(np.mean((p_pred - flip_val) ** 2))
    baseline_p = float(flip_val.mean())
    brier_baseline = float(np.mean((baseline_p - flip_val) ** 2))

    # Bảng so sánh theo bin (report thô để đọc bằng mắt)
    edges = np.quantile(margin_val, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(margin_val, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            dict(
                bin=b,
                margin_mean=float(margin_val[m].mean()),
                n=int(m.sum()),
                emp_BER_val=float(flip_val[m].mean()),
                pred_p_mean=float(p_pred[m].mean()),
            )
        )
    stats = pd.DataFrame(rows)
    stats["abs_calib_error"] = (stats["emp_BER_val"] - stats["pred_p_mean"]).abs()

    rho, p_sp = spearmanr(margin_val, flip_val)
    ber_low = stats.iloc[0]["emp_BER_val"]
    ber_high = stats.iloc[-1]["emp_BER_val"]
    ratio = ber_low / max(ber_high, 1e-9)

    print("\n============  VALIDATION TRÊN tune_val (không dùng để fit)  ============")
    print(
        stats.to_string(
            index=False,
            formatters={
                "margin_mean": "{:.4f}".format,
                "emp_BER_val": "{:.4f}".format,
                "pred_p_mean": "{:.4f}".format,
                "abs_calib_error": "{:.4f}".format,
            },
        )
    )
    print("==========================================================================")
    print(f"Brier score (model, trên val)     = {brier_model:.6f}")
    print(f"Brier score (baseline hằng số p)  = {brier_baseline:.6f}")
    improve = (
        (brier_baseline - brier_model) / brier_baseline * 100
        if brier_baseline > 0
        else 0.0
    )
    print(f"-> Cải thiện Brier so baseline     = {improve:.1f}%")
    print(f"Spearman(margin, flip) trên val    = {rho:+.4f}  (p={p_sp:.1e})")
    print(f"Ratio BER thấp/cao trên val        = {ratio:.2f}×")
    print(f"Sai số hiệu chỉnh TB |emp - pred|  = {stats['abs_calib_error'].mean():.4f}")

    if brier_model < brier_baseline and stats["abs_calib_error"].mean() < 0.03:
        print(
            "\n✅ Calibration GENERALIZE TỐT sang val — có thể dùng LLR này cho decoder."
        )
    elif brier_model < brier_baseline:
        print(
            "\n🟡 Calibration cải thiện Brier nhưng sai số hiệu chỉnh còn cao — "
            "cân nhắc giảm N_FIT_BINS (đỡ overfit) trước khi dùng cho decoder."
        )
    else:
        print(
            "\n❌ Calibration KHÔNG generalize (Brier tệ hơn baseline trên val) — "
            "kiểm tra lại N_FIT_BINS / lượng dữ liệu train trước khi sang bước 4."
        )

    return stats, brier_model, brier_baseline


# ------------------------------------------------------------------
# 5. GHI OUTPUT: bảng lookup + module Python cắm thẳng vào pipeline
# ------------------------------------------------------------------
def save_lookup_and_module(bp_margin, bp_p):
    npz_path = os.path.join(OUT_DIR, "reliability_lookup.npz")
    np.savez(
        npz_path,
        margin_breakpoints=bp_margin,
        p_breakpoints=bp_p,
        eps=np.array(EPS_P),
    )
    print(f"[out] bảng lookup -> {npz_path}")

    module_path = os.path.join(OUT_DIR, "empirical_llr.py")
    module_code = '''"""
Module tự sinh bởi 19_empirical_llr_fit.py — KHÔNG sửa tay.
Thay thế trực tiếp cho công thức tham số scale=60/sigma cũ.

Dùng:
    from empirical_llr import margin_to_llr, margin_to_p
    llr = margin_to_llr(margin_array)   # cắm thẳng vào input soft cho Neural-MS
"""
import os
import numpy as np

_DATA = np.load(os.path.join(os.path.dirname(__file__), "reliability_lookup.npz"))
_MARGIN_BP = _DATA["margin_breakpoints"]
_P_BP = _DATA["p_breakpoints"]
_EPS = float(_DATA["eps"])


def margin_to_p(margin):
    """margin (khoang cach toi nguong, cung dinh nghia voi buoc 2) -> xac suat lat bit,
    da hieu chinh thuc nghiem (isotonic, fit tren tune_train) va clip vao [eps, 0.5-eps]."""
    margin = np.asarray(margin, dtype=np.float64)
    p = np.interp(margin, _MARGIN_BP, _P_BP, left=_P_BP[0], right=_P_BP[-1])
    return np.clip(p, _EPS, 0.5 - _EPS)


def margin_to_llr(margin):
    """LLR = log((1-p)/p) — thay the hoan toan cong thuc scale=60 cu."""
    p = margin_to_p(margin)
    return np.log((1.0 - p) / p)
'''
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(module_code)
    print(f"[out] module pluggable -> {module_path}")


def plot_train_vs_val(bp_margin, bp_p, margin_val, flip_val, n_bins=N_REPORT_BINS):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        edges = np.quantile(margin_val, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(margin_val, edges[1:-1]), 0, n_bins - 1)
        val_x, val_y = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() == 0:
                continue
            val_x.append(margin_val[m].mean())
            val_y.append(flip_val[m].mean())

        plt.figure(figsize=(7, 5))
        plt.plot(bp_margin, bp_p, "-", lw=2, label="Fitted isotonic (tune_train)")
        plt.plot(val_x, val_y, "o", ms=7, label="Empirical BER (tune_val, giữ riêng)")
        plt.xlabel("Margin (|khoảng cách tới ngưỡng|)")
        plt.ylabel("Xác suất lật bit")
        plt.title("Empirical-LLR calibration: train fit vs. val hold-out")
        plt.grid(alpha=0.3)
        plt.legend()
        png = os.path.join(OUT_DIR, "calibration_train_vs_val.png")
        plt.savefig(png, dpi=130, bbox_inches="tight")
        print(f"[out] biểu đồ -> {png}")
    except Exception as ex:
        print(f"[warn] không vẽ được biểu đồ: {ex}")


# ------------------------------------------------------------------
def main():
    print("=" * 66)
    print("BƯỚC 3 — Empirical-LLR calibration (fit train / validate val)")
    print("=" * 66)

    train_pairs, val_pairs = load_tune_genuine_pairs()
    print(f"[pairs] tune_train={len(train_pairs)}  tune_val={len(val_pairs)}")

    self_check_against_original(train_pairs + val_pairs)

    print("\n[fit] thu thập (margin, flip) trên tune_train...")
    margin_train, flip_train = collect(train_pairs)
    print(f"[fit] {len(margin_train):,} quan sát bit trên train.")

    print(f"[fit] isotonic PAVA trên {N_FIT_BINS} bin quantile...")
    bp_margin, bp_p = fit_isotonic_calibration(
        margin_train, flip_train, n_bins=N_FIT_BINS
    )
    print(f"[fit] xong — {len(bp_margin)} breakpoint đơn điệu không tăng.")
    print(f"      p tại margin nhỏ nhất  = {bp_p[0]:.4f}")
    print(f"      p tại margin lớn nhất  = {bp_p[-1]:.4f}")

    print("\n[val] thu thập (margin, flip) trên tune_val (giữ riêng, không fit)...")
    margin_val, flip_val = collect(val_pairs)
    print(f"[val] {len(margin_val):,} quan sát bit trên val.")

    validate_on_val(margin_val, flip_val, bp_margin, bp_p)

    save_lookup_and_module(bp_margin, bp_p)
    plot_train_vs_val(bp_margin, bp_p, margin_val, flip_val)

    print("\n[next] Cắm module vừa sinh vào pipeline sinh soft-input cho decoder:")
    print("       from empirical_llr import margin_to_llr")
    print("       (thay hoàn toàn công thức scale=60 cũ)")
    print("       Sau đó fine-tune/retrain Neural-MS TRÊN đúng phân phối LLR này")
    print("       (vẫn chỉ trên tune — select/final để dành cho đánh giá cuối cùng).")


if __name__ == "__main__":
    main()
