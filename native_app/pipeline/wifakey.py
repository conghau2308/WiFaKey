"""
WiFaKey ONNX — replaces wifakey_handler.py (TF1.x) with onnxruntime.
Inlines encoder + LSSC so this module has zero dependency on
wifakey_lib (which pulls in sympy / scipy / matplotlib).

LDPC decoding now happens server-side (Authentication_Service) — the client
only computes and returns the noisy codeword c' = b_selected XOR helper_data,
so the trained decoder model never ships inside the distributed app.
"""
import hashlib
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (ported from wifakey_lib without scipy/sympy/tqdm)
# ──────────────────────────────────────────────────────────────────────────────

def _build_lkut(n_thr: int) -> np.ndarray:
    """Thermometer lookup table: lkut[i] is the binary code for bin i."""
    n_bins = n_thr + 1
    lkut = np.zeros((n_bins, n_thr), dtype=np.uint8)
    for i in range(1, n_bins):
        lkut[i, n_thr - i:] = 1
    return lkut


def _lssc_binary(projected: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """
    Vectorised LSSC binarization for a single 512-dim projected vector.
    Matches the batch version in wifakey_lib/utils.py exactly.
    """
    n_thr = len(intervals)
    lkut = _build_lkut(n_thr)
    # searchsorted 'right' == first position where interval > value (original logic)
    indices = np.searchsorted(intervals, projected, side="right")
    codes = lkut[indices]                         # (512, n_thr)
    out = np.zeros(512 * n_thr, dtype=np.uint8)
    n = min(len(projected), 512)
    out[: n * n_thr] = codes[:n].flatten()
    return out


class _LDPCEncoder:
    """Minimal Proto_LDPC encoder — GF(2) matrix multiplication only."""

    def __init__(self, Z: int, data_dir: Path):
        gm_file = data_dir / "BaseGraph_GM" / f"LDPC_GM_BG2_{Z}.txt"
        self._G = np.loadtxt(str(gm_file), dtype=int, delimiter=",")

    def encode(self, key_bits: np.ndarray) -> np.ndarray:
        """key_bits: (1, key_length) int → codeword (1, feature_length) int."""
        return np.dot(key_bits, self._G) % 2


# ──────────────────────────────────────────────────────────────────────────────
# Main handler
# ──────────────────────────────────────────────────────────────────────────────

class WiFaKeyONNX:
    N = 52
    M = 42
    Z = 16
    FEATURE_LEN = N * Z    # 832
    KEY_LEN = (N - M) * Z  # 160
    KAPPA = 0.3125

    def __init__(self, data_dir: Path):
        self._encoder  = _LDPCEncoder(self.Z, data_dir)
        self._M_matrix = np.load(str(data_dir / "M_matrix.npy"))
        self._intervals = np.load(str(data_dir / "binarization_intervals.npy"))

    # ── public ──────────────────────────────────────────────────────────────

    def enroll(self, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray, bytes]:
        """
        Returns (helper_data uint8[832], mask uint8[full_len], key_hash bytes[32]).
        """
        b_full = self._binarize(embedding)

        u = np.random.uniform(0.0, 1.0, size=len(b_full))
        mask = (u >= self.KAPPA).astype(np.uint8)
        b_masked = (b_full & mask).astype(np.uint8)
        b_sel = b_masked[: self.FEATURE_LEN]

        key = np.random.randint(0, 2, size=(1, self.KEY_LEN), dtype=np.uint8)
        codeword = self._encoder.encode(key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_sel, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(key.flatten().tobytes()).digest()

        return helper_data, mask, key_hash

    def get_noisy_codeword(
        self,
        embedding: np.ndarray,
        helper_data: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute c' = b_selected XOR helper_data (uint8[FEATURE_LEN]).

        LDPC decoding + key reconstruction + hashing now happen server-side
        (Authentication_Service), so the client stops here and never sees
        the reconstructed key.
        """
        b_full = self._binarize(embedding)
        b_masked = (b_full & mask[: len(b_full)]).astype(np.uint8)
        b_sel = b_masked[: self.FEATURE_LEN]
        noisy = np.logical_xor(b_sel, helper_data[: self.FEATURE_LEN])
        return noisy.astype(np.uint8)

    # ── private ─────────────────────────────────────────────────────────────

    def _binarize(self, embedding: np.ndarray) -> np.ndarray:
        projected = np.dot(embedding, self._M_matrix)
        return _lssc_binary(projected, self._intervals)
