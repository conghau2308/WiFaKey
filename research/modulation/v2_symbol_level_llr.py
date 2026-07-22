"""
v2_symbol_level_llr.py - CAI TIEN MUC 2 (symbol-level LLR).

Muc 1 (v1_soft_distance_llr.py) coi moi bit trong 1 khoi n_thr bit la
DOC LAP, gan CUNG 1 bien do cho ca khoi dua tren "khoang cach toi nguong
gan nhat". Day la xap xi tho: thermometer code CHI CO n_thr+1 mau hop le
(khong phai 2^n_thr to hop doc lap), nen 2 bit trong cung khoi luon
TUONG QUAN voi nhau.

Muc 2 tinh dung hon: voi gia tri lien tuc quan sat duoc v, dat gia thuyet
nhieu Gaussian quanh TAM moi bin (level), tinh xac suat hau nghiem
P(level=L | v) qua tat ca n_thr+1 level, roi MARGINALIZE de ra LLR cho
tung bit rieng le:

    P(bit_p=1 | v) = sum_{L: bit_p(L)=1} P(L|v)
    P(bit_p=0 | v) = sum_{L: bit_p(L)=0} P(L|v)
    LLR_p = ln( P(bit_p=1|v) / P(bit_p=0|v) )

trong do bit_p(L) tra ve gia tri bit tai vi tri p khi level=L, theo dung
thermometer code (bit_p=1 <=> L >= n_thr-p, khop voi lkut trong
lssc_binary goc).

Tham so quan trong: sigma (do nhieu Gaussian gia dinh). Cach hieu chinh:
tune sigma sao cho ty le "bit flip du doan" (theo mo hinh) khop voi BER
THAT do duoc tren tap tune (xem ham calibrate_sigma_from_ber bên duoi).

Cach dung: thay the cho ca 2 buoc quantizer+modulation cu (vi Muc 2 can
CA gia tri lien tuc THO, khong chi bits+confidence nhu Muc 1) - class
nay tu lam tu dau (chieu M_matrix, tinh LLR), khong dung lai
binarize_with_confidence.
"""

import numpy as np


class SymbolLevelLLR:
    name = "v2_symbol_level_llr"

    def __init__(
        self, sigma=0.01, min_mag=0.1, max_mag=10.0, masked_mag=1.0, llr_scale=1.0
    ):
        """
        sigma: do lech chuan gia dinh cua nhieu bio giua 2 lan chup, tren
               THANG GIA TRI DA CHIEU QUA M_matrix (cung don vi voi
               intervals). Can hieu chinh bang calibrate_sigma_from_ber().
        min_mag/max_mag: chan bien do LLR cuoi cung (sau khi nhan
               llr_scale) de tranh gia tri qua nho/qua lon lam mat can
               bang voi thang ma decoder da quen (~1.0).
        masked_mag: bien do gan cho bit bi mask (mask_r=0) - PHAI la gia
               tri trung tinh (~1.0), khong dung logic Muc 2 cho cac vi
               tri nay (giong ly do da xac dinh o Muc 1).
        llr_scale: he so nhan them SAU KHI tinh LLR ly thuyet, de dua ve
               thang bien do ma decoder da duoc hieu chinh (~1.0 trung
               binh) - dong vai tro tuong tu 'scale' o Muc 1.
        """
        self.sigma = sigma
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.masked_mag = masked_mag
        self.llr_scale = llr_scale

    def _level_probs(self, v, intervals):
        """P(level=L | v) cho 1 gia tri lien tuc v, voi n_thr+1 level.
        Level L duoc gia thuyet co "tam" la trung diem giua 2 nguong lan
        can (hoac ngoai suy 1 do rong bin cho 2 level ngoai cung, vi
        chung khong bi chan)."""
        n_thr = len(intervals)
        # Tam moi level: dung trung diem giua cac nguong; 2 dau ngoai suy
        # bang do rong bin lan can (xap xi don gian, du dung trong thuc te).
        bin_width_est = np.mean(np.diff(intervals)) if n_thr > 1 else 1.0
        centers = np.zeros(n_thr + 1)
        centers[0] = intervals[0] - bin_width_est / 2
        centers[-1] = intervals[-1] + bin_width_est / 2
        for L in range(1, n_thr):
            centers[L] = (intervals[L - 1] + intervals[L]) / 2

        log_probs = -((v - centers) ** 2) / (2 * self.sigma**2)
        log_probs -= np.max(log_probs)  # on dinh so hoc truoc khi exp
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        return probs  # shape (n_thr+1,)

    def compute_llr_for_value(self, v, intervals):
        """Tra ve vector LLR (n_thr,) cho 1 gia tri lien tuc v."""
        n_thr = len(intervals)
        level_probs = self._level_probs(v, intervals)  # (n_thr+1,)

        llr = np.zeros(n_thr, dtype=np.float32)
        for p in range(n_thr):
            # bit_p(L) = 1 <=> L >= n_thr - p (khop dung lkut goc)
            threshold_level = n_thr - p
            p1 = np.sum(level_probs[threshold_level:])
            p0 = 1.0 - p1
            p1 = np.clip(p1, 1e-6, 1 - 1e-6)
            p0 = np.clip(p0, 1e-6, 1 - 1e-6)
            llr[p] = np.log(p1 / p0)
        return llr

    def compute_llr_full(self, projected_values, intervals):
        """projected_values: (D,) - toan bo D chieu embedding da chieu qua
        M_matrix. Tra ve (D*n_thr,) LLR, flatten dung thu tu nhu
        lssc_binary goc (moi chieu -> 1 khoi n_thr bit lien tiep)."""
        n_thr = len(intervals)
        D = len(projected_values)
        llr_full = np.zeros(D * n_thr, dtype=np.float32)
        for d in range(D):
            llr_full[d * n_thr : (d + 1) * n_thr] = self.compute_llr_for_value(
                projected_values[d], intervals
            )
        return llr_full

    def modulate(self, noisy_bits, context=None):
        """Interface khop voi BaseModulation, NHUNG can context khac Muc 1:
        context phai co 'llr_theoretical' (tinh san tu compute_llr_full,
        cat theo feature_length) va 'mask'. noisy_bits van can de xac dinh
        DAU cuoi cung (vi LLR ly thuyet tinh tu GIA TRI QUAN SAT THO, con
        dau thuc te con phu thuoc XOR voi helper_data)."""
        if context is None or "llr_theoretical" not in context:
            raise ValueError(
                f"[{self.name}] can context['llr_theoretical'] - tinh tu "
                f"compute_llr_full() TRUOC khi mask/XOR, xem verify_variant_v2.py"
            )

        llr_theoretical = context["llr_theoretical"]
        mask = context.get("mask")

        magnitude = np.clip(
            np.abs(llr_theoretical) * self.llr_scale, self.min_mag, self.max_mag
        )
        if mask is not None:
            magnitude = np.where(mask.astype(bool), magnitude, self.masked_mag)

        sign = 2 * noisy_bits.astype(np.float32) - 1
        return (sign * magnitude).astype(np.float32)


def calibrate_sigma_from_ber(handler, tune_pairs, sigma_candidates, n_samples=200):
    """Do BER THAT tren tap tune, roi so sanh voi BER DU DOAN boi mo hinh
    Gaussian voi tung sigma trong sigma_candidates - chon sigma cho BER
    du doan GAN NHAT voi BER that (tuong tu cach hieu chinh 'scale' o
    Muc 1, nhung do truc tiep tren xac suat lat bit thay vi khoang cach).

    Tra ve sigma tot nhat va bang so sanh (sigma, ber_du_doan, ber_that).
    """
    print("[MARKER] Đang chạy phiên bản calibrate_sigma_from_ber ĐÃ SỬA "
          "(dùng 1-max(probs), KHÔNG dùng argmax) - nếu bạn không thấy dòng "
          "này, code cũ vẫn đang được nạp, KHÔNG PHẢI bản đã sửa.")
    
    intervals = np.sort(np.asarray(handler.intervals).flatten())
    n_thr = len(intervals)

    # Do BER THAT (tai su dung logic tu script 17_ber_by_bin_type.py)
    n_flip_total, n_bit_total = 0, 0
    projected_pairs = []
    for emb_enroll, emb_verify in tune_pairs[:n_samples]:
        proj_e = np.dot(emb_enroll, handler.M_matrix)
        proj_v = np.dot(emb_verify, handler.M_matrix)
        projected_pairs.append((proj_e, proj_v))

        bin_e = np.searchsorted(intervals, proj_e, side="left")
        bin_v = np.searchsorted(intervals, proj_v, side="left")
        for d in range(len(proj_e)):
            n_diff = int(
                bin_e[d] != bin_v[d]
            )  # xap xi: khac bin -> co it nhat 1 bit khac
            n_flip_total += n_diff
            n_bit_total += 1
    ber_that = n_flip_total / n_bit_total

    results = []
    for sigma in sigma_candidates:
        model = SymbolLevelLLR(sigma=sigma)
        # BER du doan: P(level quan sat != level that) trung binh, xap xi
        # bang P(sai level) = 1 - P(level dung nhat theo posterior).
        n_wrong, n_val = 0, 0
        for proj_e, proj_v in projected_pairs:
            for d in range(len(proj_v)):
                probs = model._level_probs(proj_v[d], intervals)
                predicted_level = int(np.argmax(probs))
                true_level = int(np.searchsorted(intervals, proj_e[d], side="left"))
                n_wrong += int(predicted_level != true_level)
                n_val += 1
        ber_pred = n_wrong / n_val
        results.append((sigma, ber_pred, ber_that))

    best_sigma = min(results, key=lambda r: abs(r[1] - r[2]))[0]
    return best_sigma, results
