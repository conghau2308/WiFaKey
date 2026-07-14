"""
WiFaKey calibration aligned with wifakey_handler:

  embedding (512) -> dot M_matrix (512) -> LSSC -> b_full (512 * n_thresholds bits)
  -> κ search and gen/imp stats on the first ``feature_length`` bits only (same as LDPC input).

Re-run after changing M_matrix.npy or INTERVAL_NUM.

FAR (impostor accepts): default κ mode is ``joint`` — requires masked impostor Hamming
on the LDPC bit slice to stay high on the calibration pairs (see ``look4noncerate_joint``).
Zero FAR on every future dataset cannot be guaranteed; if ``test_FAR`` still shows accepts,
lower κ slightly: set env ``WIFAKEY_KAPPA_SUB=0.01`` (or ``0.02``) when starting the server
or test, then re-check TAR/FAR trade-off.
"""
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from wifakey_module.wifakey_lib.utils import (
    equal_probable,
    lssc_binary,
    look4noncerate,
    look4noncerate_joint,
    computeGenImp,
)

EMBEDDINGS_FILE = './embeddings/adaface_lfw.csv'
ISSAME_FILE = './embeddings/lfw_issame.csv'

OUTPUT_DIR = './wifakey_module/data'
M_MATRIX_PATH = os.path.join(OUTPUT_DIR, 'M_matrix.npy')

INTERVALS_PATH = os.path.join(OUTPUT_DIR, 'binarization_intervals.npy')
KAPPA_PATH = os.path.join(OUTPUT_DIR, 'kappa.npy')

INTERVAL_NUM = 4

# Must match WiFaKeyHandler: code_n * Z (bits fed to LDPC after mask + b[:feature_length])
N_LDPC = 52
Z_LDPC = 16
FEATURE_LENGTH = N_LDPC * Z_LDPC  # 832

# κ search mode: "joint" (default) = minimize FAR proxy on calib set; "genuine" = paper Eq.5 only
KAPPA_MODE = os.environ.get("WIFAKEY_KAPPA_MODE", "joint").strip().lower()
GEN_THRESHOLD = float(os.environ.get("WIFAKEY_GEN_THRESHOLD", "0.1762"))
GEN_CONFIDENCE = float(os.environ.get("WIFAKEY_GEN_CONFIDENCE", "0.95"))
# Impostor: require masked BER >= imp_thr on (imp_conf) fraction of impostor pairs; 1.0 = all pairs
IMP_CONFIDENCE = float(os.environ.get("WIFAKEY_IMP_CONFIDENCE", "1.0"))
# If joint fails, try these imp thresholds from strict to looser (higher = easier to satisfy)
IMP_THRESHOLD_CANDIDATES = [
    float(x)
    for x in os.environ.get(
        "WIFAKEY_IMP_THRESHOLDS",
        "0.30,0.28,0.26,0.24,0.22,0.215,0.21,0.205,0.20,0.195,0.19,0.18",
    ).split(",")
    if x.strip()
]


def run_calibration():

    print("=" * 60)
    print("WIFAKEY CALIBRATION (handler-aligned pipeline)")
    print("=" * 60)

    if not os.path.exists(M_MATRIX_PATH):
        raise FileNotFoundError(
            f"Missing {M_MATRIX_PATH}. Generate or copy M_matrix.npy into wifakey_module/data."
        )

    print("[1/5] Loading embeddings")
    embeddings_o = np.loadtxt(EMBEDDINGS_FILE, delimiter=',')
    print("Embeddings shape:", embeddings_o.shape)

    dim = embeddings_o.shape[1]
    if dim != 512:
        print(f"Warning: expected 512-D AdaFace embeddings, got dim={dim}")

    print("[2/5] Loading issame labels")
    issame = np.loadtxt(ISSAME_FILE, delimiter=',').astype(int)
    print("issame shape:", issame.shape)

    print("[3/5] Loading M_matrix and projecting (same as _binarize_full)")
    M_matrix = np.load(M_MATRIX_PATH)
    if M_matrix.shape != (dim, dim):
        raise ValueError(
            f"M_matrix shape {M_matrix.shape} incompatible with embedding dim {dim}"
        )
    projected = np.dot(embeddings_o, M_matrix)
    print("Projected shape:", projected.shape)

    print("[4/5] BLuT intervals on projected features + LSSC binarization")
    intervals = equal_probable(projected, intervalnum=INTERVAL_NUM)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    np.save(INTERVALS_PATH, intervals)
    print("Intervals:", intervals)

    embeddings_bin = lssc_binary(projected, interval=intervals)
    print("Full binary shape (before LDPC slice):", embeddings_bin.shape)

    n_bits_full = embeddings_bin.shape[1]
    if n_bits_full < FEATURE_LENGTH:
        raise ValueError(
            f"LSSC width {n_bits_full} < FEATURE_LENGTH {FEATURE_LENGTH}; "
            "check INTERVAL_NUM vs handler."
        )

    embeddings_channel = embeddings_bin[:, :FEATURE_LENGTH]
    print(f"κ calibration slice: first {FEATURE_LENGTH} bits (LDPC channel), of {n_bits_full} total")

    gen, imp = computeGenImp(embeddings_channel, issame)
    print("Mean genuine BER (unmasked, channel bits):", float(np.mean(gen)))
    print("Std genuine BER:", float(np.std(gen)))
    print("Mean impostor BER:", float(np.mean(imp)))

    print(f"Searching κ — mode={KAPPA_MODE!r} (set WIFAKEY_KAPPA_MODE=joint|genuine|impostor)")

    kappa = None
    if KAPPA_MODE == "genuine":
        _embeddings_nonce, kappa, _nonce = look4noncerate(
            embeddings_channel.astype(np.uint8),
            issame,
            threshold=GEN_THRESHOLD,
            genorimp=1,
            confidence=GEN_CONFIDENCE,
        )
    elif KAPPA_MODE == "impostor":
        _embeddings_nonce, kappa, _nonce = look4noncerate(
            embeddings_channel.astype(np.uint8),
            issame,
            threshold=float(os.environ.get("WIFAKEY_IMP_THRESHOLD", "0.20")),
            genorimp=0,
            confidence=IMP_CONFIDENCE,
        )
    else:
        # joint (default): tighten impostor separation for lower FAR; relax imp threshold if needed
        for imp_thr in IMP_THRESHOLD_CANDIDATES:
            _embeddings_nonce, kappa, _nonce = look4noncerate_joint(
                embeddings_channel.astype(np.uint8),
                issame,
                gen_threshold=GEN_THRESHOLD,
                imp_threshold=imp_thr,
                gen_confidence=GEN_CONFIDENCE,
                imp_confidence=IMP_CONFIDENCE,
            )
            if kappa is not None:
                print(f"Joint κ satisfied with imp_threshold={imp_thr}")
                break
        if kappa is None:
            print("Joint search failed; falling back to genuine-only κ (higher FAR risk).")
            _embeddings_nonce, kappa, _nonce = look4noncerate(
                embeddings_channel.astype(np.uint8),
                issame,
                threshold=GEN_THRESHOLD,
                genorimp=1,
                confidence=GEN_CONFIDENCE,
            )

    np.save(KAPPA_PATH, np.array(kappa))
    print("Optimal κ:", kappa)

    print("Saved:")
    print(" -", os.path.abspath(INTERVALS_PATH))
    print(" -", os.path.abspath(KAPPA_PATH))

    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_calibration()
