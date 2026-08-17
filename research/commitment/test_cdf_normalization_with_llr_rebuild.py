"""
test_cdf_normalization_with_llr_rebuild.py  (DA SUA theo doi chieu source that
cua EmpiricalLLR va SecureWiFaKeyHandler)

Sua so voi ban truoc:
  1. Dau LLR: sign = (2*noisy_bits - 1)  [noisy_bit=1 -> +, noisy_bit=0 -> -]
     KHOP CHINH XAC voi EmpiricalLLR.modulate() va quy uoc Modulation.BPSK
     ma SecureWiFaKeyHandler.verify() dang dung.
  2. Clip xac suat loi p_err trong [eps, 0.5-eps], KHONG phai [eps, 1-eps]
     - khop dung EmpiricalLLR._margin_to_llr_magnitude().
  3. Dung handler.N / handler.Z / handler.feature_length / handler.key_length
     (thuoc tinh that tren object) thay vi hang so cung, tranh lech dinh nghia.

GIAI DOAN:
  0. Sanity check: rebuild LLR tren pipeline GOC (khong CDF), so GMR voi
     ~88.9% da biet truoc. Neu khong khop -> DUNG LAI.
  1. Fit Gaussianization tung chieu tren TRAIN, giu nguyen handler.intervals.
  2. Rebuild LLR tren pipeline CDF-transform, dung TRAIN.
  3. Danh gia GMR + do lai entropy tren TEST (tach biet theo NGUOI DUNG).
"""

import os, sys, csv, hashlib
import numpy as np
from scipy.stats import norm
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)

# ===================== Hang so khong phu thuoc handler =====================
N_DIMS = 512
N_TRAIN_PAIRS_FOR_LLR = 400
N_TEST_PAIRS = 200
N_TEST_USERS_HELD_OUT = 60
N_MARGIN_BINS = 25
MIN_SAMPLES_PER_BIN = 20
KNOWN_BASELINE_GMR_PCT = 88.9  # Empirical LLR goc tren LFW 881 cap
EPS = 1e-4
SEED = 777


# ===================== Data loading =====================


def load_embedding(name, imagenum):
    path = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "embeddings_cache",
        f"{name}_{int(imagenum):04d}.npy",
    )
    return np.load(path)


def load_all_genuine_pairs():
    pairs_csv = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "pairs",
        "tune_genuine.csv",
    )
    pairs = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                (
                    row["name_enroll"],
                    int(row["imagenum_enroll"]),
                    row["name_verify"],
                    int(row["imagenum_verify"]),
                )
            )
    return pairs


# ===================== Tier 1: Gaussianization =====================


class PerDimGaussianizer:
    """Probability Integral Transform tung chieu, uoc luong tren TRAIN.
    KHONG doi threshold - chi doi thang do cua projected value truoc khi
    so nguong, giu nguyen handler.intervals."""

    def __init__(self, train_projected):
        self.sorted_train = np.sort(train_projected, axis=0)
        self.n_train = train_projected.shape[0]

    def transform(self, projected):
        out = np.zeros_like(projected, dtype=np.float64)
        for d in range(N_DIMS):
            rank = np.searchsorted(self.sorted_train[:, d], projected[d])
            u = np.clip((rank + 0.5) / (self.n_train + 1), 1e-6, 1 - 1e-6)
            out[d] = norm.ppf(u)
        return out


# ===================== Rebuilt Empirical LLR (khop dung source that) =====================


class RebuiltEmpiricalLLR:
    """Tu xay bang tra cuu margin -> LLR tu du lieu TRAIN. Khop dung cong
    thuc/quy uoc dau cua research/modulation/v2_empirical_llr.EmpiricalLLR
    (xem doi chieu o docstring dau file), chi khac o cho dung bang binned
    thay vi noi suy tuyen tinh tren breakpoint tu isotonic regression."""

    def __init__(self):
        self.bin_edges = None
        self.bin_magnitude = None

    def fit(self, margins, error_flags, n_bins=N_MARGIN_BINS):
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(margins, quantiles)
        edges[0], edges[-1] = -np.inf, np.inf
        self.bin_edges = edges

        bin_magnitude = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            mask = (margins >= edges[b]) & (margins < edges[b + 1])
            n_in_bin = int(mask.sum())
            bin_counts[b] = n_in_bin
            if n_in_bin < MIN_SAMPLES_PER_BIN:
                bin_magnitude[b] = 0.0
                continue
            p_err = np.clip(
                error_flags[mask].mean(), EPS, 0.5 - EPS
            )  # FIX: chan tren 0.5
            bin_magnitude[b] = np.log((1.0 - p_err) / p_err)  # luon >= 0
        self.bin_magnitude = bin_magnitude
        print(
            f"    [LLR fit] so bin co du du lieu: {(bin_counts >= MIN_SAMPLES_PER_BIN).sum()}/{n_bins}, "
            f"tong sample: {len(margins)}"
        )
        return self

    def modulate(self, noisy_bits, context):
        margins = context["margin"]
        assert margins.shape == noisy_bits.shape, "margin va noisy_bits phai cung shape"
        bin_idx = np.clip(
            np.searchsorted(self.bin_edges, margins) - 1, 0, len(self.bin_magnitude) - 1
        )
        magnitude = self.bin_magnitude[bin_idx]
        sign = (
            2 * noisy_bits.astype(np.float32) - 1
        )  # FIX: bit=1 -> +1, bit=0 -> -1 (khop EmpiricalLLR that)
        return (sign * magnitude).astype(np.float32)


# ===================== Pipeline dung chung (goc / CDF) =====================


def binarize_transformed(handler, emb, gaussianizer=None):
    projected = np.dot(emb, handler.M_matrix)
    if gaussianizer is not None:
        projected = gaussianizer.transform(projected)
    b_full, margin = binarize_with_perbit_confidence(projected, handler.intervals)
    return b_full.astype(np.uint8), margin


def select_positions(handler, margin_enroll):
    idx = np.argpartition(-margin_enroll, handler.feature_length)[
        : handler.feature_length
    ]
    idx.sort()
    return idx


def build_llr_training_data(handler, train_pairs, gaussianizer=None):
    all_margins, all_errors = [], []
    for name_e, img_e, name_v, img_v in train_pairs:
        b_full_e, margin_e = binarize_transformed(
            handler, load_embedding(name_e, img_e), gaussianizer
        )
        b_full_v, margin_v = binarize_transformed(
            handler, load_embedding(name_v, img_v), gaussianizer
        )

        sel_idx = select_positions(handler, margin_e)
        b_sel_e = b_full_e[sel_idx]
        b_sel_v = b_full_v[sel_idx]
        margin_sel_v = margin_v[sel_idx]

        error_flag = (b_sel_e != b_sel_v).astype(np.uint8)
        all_margins.append(margin_sel_v)
        all_errors.append(error_flag)

    return np.concatenate(all_margins), np.concatenate(all_errors)


def evaluate_gmr(handler, test_pairs, gaussianizer, llr_model, label):
    n_pass = 0
    for name_e, img_e, name_v, img_v in test_pairs:
        b_full_e, margin_e = binarize_transformed(
            handler, load_embedding(name_e, img_e), gaussianizer
        )
        b_full_v, margin_v = binarize_transformed(
            handler, load_embedding(name_v, img_v), gaussianizer
        )

        sel_idx = select_positions(handler, margin_e)
        b_sel_e = b_full_e[sel_idx]
        b_sel_v = b_full_v[sel_idx]
        margin_sel_v = margin_v[sel_idx]

        rng = np.random.default_rng()
        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper = np.logical_xor(b_sel_e, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        noisy = np.logical_xor(b_sel_v, helper).astype(np.uint8)
        llr = llr_model.modulate(noisy, context={"margin": margin_sel_v})
        llr = llr.reshape(1, handler.N, handler.Z).astype(np.float32)

        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            n_pass += 1

    pct = 100 * n_pass / len(test_pairs)
    print(f"  [{label}] GMR: {n_pass}/{len(test_pairs)} ({pct:.2f}%)")
    return n_pass, pct


def measure_entropy(handler, identities, gaussianizer, label):
    bit_counts = np.zeros(1536)
    selected_counts = np.zeros(1536)
    for name, imagenum in identities:
        b_full, margin = binarize_transformed(
            handler, load_embedding(name, imagenum), gaussianizer
        )
        sel_idx = select_positions(handler, margin)
        selected_counts[sel_idx] += 1
        bit_counts[sel_idx] += b_full[sel_idx]

    valid = selected_counts > 0
    p1 = np.divide(bit_counts, selected_counts, out=np.full(1536, 0.5), where=valid)
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    ent = -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
    avg = float(np.mean(ent[valid]))
    print(
        f"  [{label}] Entropy trung binh tai vi tri duoc chon: {avg:.4f} bit (ro ri {100*(1-avg):.1f}%)"
    )
    return avg


# ===================== Main =====================


def main():
    handler = SecureWiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )
    all_pairs = load_all_genuine_pairs()

    rng = np.random.default_rng(SEED)
    unique_names = []
    seen = set()
    for name_e, img_e, _, _ in all_pairs:
        if name_e not in seen:
            seen.add(name_e)
            unique_names.append(name_e)
    rng.shuffle(unique_names)
    test_names = set(unique_names[:N_TEST_USERS_HELD_OUT])

    train_pairs = [p for p in all_pairs if p[0] not in test_names][
        :N_TRAIN_PAIRS_FOR_LLR
    ]
    test_pairs = [p for p in all_pairs if p[0] in test_names][:N_TEST_PAIRS]
    test_identities = list({(p[0], p[1]) for p in test_pairs})

    print(f"TRAIN pairs (xay LLR/Gaussianizer): {len(train_pairs)}")
    print(f"TEST pairs (danh gia): {len(test_pairs)}")
    print(f"TEST identities (danh gia entropy): {len(test_identities)}\n")

    print("=== GIAI DOAN 0: Sanity check - rebuild LLR tren pipeline GOC ===")
    margins_0, errors_0 = build_llr_training_data(
        handler, train_pairs, gaussianizer=None
    )
    llr_original_rebuilt = RebuiltEmpiricalLLR().fit(margins_0, errors_0)
    _, gmr0_pct = evaluate_gmr(
        handler,
        test_pairs,
        None,
        llr_original_rebuilt,
        f"GOC + LLR rebuild (ky vong ~{KNOWN_BASELINE_GMR_PCT}%)",
    )
    measure_entropy(handler, test_identities, None, "GOC")

    if abs(gmr0_pct - KNOWN_BASELINE_GMR_PCT) > 15:
        print(
            f"\n!!! CANH BAO: GMR Giai doan 0 ({gmr0_pct:.1f}%) van lech xa so voi "
            f"~{KNOWN_BASELINE_GMR_PCT}%. Gui log day du de toi doi chieu tiep - "
            f"co the con diem khac trong _binarize_full/encode_LDPC can kiem tra.\n"
        )
    else:
        print(f"\n=> Giai doan 0 khop hop ly, tin duoc cac giai doan sau.\n")

    print("=== GIAI DOAN 1: Fit CDF/Gaussianization tren TRAIN ===")
    train_projected = []
    for name_e, img_e, name_v, img_v in train_pairs:
        train_projected.append(np.dot(load_embedding(name_e, img_e), handler.M_matrix))
        train_projected.append(np.dot(load_embedding(name_v, img_v), handler.M_matrix))
    train_projected = np.array(train_projected)
    gaussianizer = PerDimGaussianizer(train_projected)
    print(f"  Da fit tren {train_projected.shape[0]} embedding, {N_DIMS} chieu.\n")

    print("=== GIAI DOAN 2: Rebuild LLR tren pipeline CDF-transform ===")
    margins_1, errors_1 = build_llr_training_data(
        handler, train_pairs, gaussianizer=gaussianizer
    )
    llr_cdf = RebuiltEmpiricalLLR().fit(margins_1, errors_1)
    print()

    print("=== GIAI DOAN 3: Danh gia pipeline CDF-transform tren TEST ===")
    evaluate_gmr(
        handler, test_pairs, gaussianizer, llr_cdf, "CDF-transform + LLR rebuild"
    )
    measure_entropy(handler, test_identities, gaussianizer, "CDF-transform")

    print("\n=== TOM TAT ===")
    print(f"GOC: GMR {gmr0_pct:.2f}% | CDF-transform: xem Giai doan 3 o tren")

    handler.sess.close()


if __name__ == "__main__":
    main()
