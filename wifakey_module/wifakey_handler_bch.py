"""WiFaKey handler that uses a binary BCH code (galois.BCH) for the ECC.

Mirrors ``wifakey_module.wifakey_handler.WiFaKeyHandler`` API but plugs the
hard-decision Berlekamp-Massey BCH decoder from
``wifakey_module.decoders.bch_decoder.BCHDecoder``.

Default ``BCH(1023, 193)`` (rate ~0.1887, t=118) is the closest valid
``galois.BCH(1023, k)`` rate to Neural-MS (160/832 ~= 0.1923).
Because n=1023 > 832, this handler consumes 1023
bits from the LSSC output (default LSSC is 512*3 = 1536 bits, so it fits).
"""

from __future__ import annotations

import hashlib
import os
from typing import Tuple

import numpy as np

from .decoders.bch_decoder import BCHDecoder
from .wifakey_lib import utils


class WiFaKeyBCHHandler:
    """End-to-end WiFaKey pipeline backed by ``BCHDecoder`` (galois)."""

    name = "bch"

    def __init__(
        self,
        data_path: str = "wifakey_module/data",
        n: int = 1023,
        k: int = 193,
        kappa_sub: float = 0.009,
    ) -> None:
        print(f"[WiFaKey-BCH] Initializing (n={n}, k={k})...")

        self.data_path = data_path

        # kappa_path = os.path.join(self.data_path, "kappa.npy")
        # if os.path.exists(kappa_path):
        #     self.kappa = float(np.load(kappa_path))
        # else:
        #     self.kappa = 0.0
        # if kappa_sub is not None:
        #     self.kappa = float(np.clip(self.kappa - float(kappa_sub), 0.0, 0.999))
        self.kappa = 0.3125
        print(f"[WiFaKey-BCH] kappa = {self.kappa:.4f}")

        m_matrix_path = os.path.join(self.data_path, "M_matrix.npy")
        if not os.path.exists(m_matrix_path):
            raise FileNotFoundError(f"Missing M_matrix.npy in {self.data_path}.")
        self.M_matrix = np.load(m_matrix_path)

        interval_path = os.path.join(self.data_path, "binarization_intervals.npy")
        self.intervals = np.load(interval_path)
        n_thr = int(np.asarray(self.intervals).size)
        self.full_binary_length = self.M_matrix.shape[0] * n_thr

        print("[WiFaKey-BCH] Building BCH encoder/decoder (galois)...")
        self.ecc = BCHDecoder(n=n, k=k)
        self.feature_length = int(self.ecc.n)
        self.key_length = int(self.ecc.k)
        self.t = int(self.ecc.t)
        self.rate = float(self.ecc.rate)

        if self.full_binary_length < self.feature_length:
            raise ValueError(
                f"LSSC output ({self.full_binary_length} bits) shorter than "
                f"BCH n ({self.feature_length}); need a longer LSSC config or "
                f"a smaller BCH code."
            )

        print(
            f"[WiFaKey-BCH] Ready. n={self.feature_length}, k={self.key_length}, "
            f"t={self.t}, rate={self.rate:.4f}"
        )

    def _binarize_full(self, feature_vector_float: np.ndarray) -> np.ndarray:
        projected = np.dot(feature_vector_float, self.M_matrix)
        feature_vector_2d = np.expand_dims(projected, axis=0)
        binary_vector_expanded = utils.lssc_binary(
            feature_vector_2d, interval=self.intervals
        ).flatten()
        return binary_vector_expanded

    def enroll(
        self, feature_vector_float: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bytes]:
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        if b_full.size < self.feature_length:
            raise ValueError(
                f"b_full has {b_full.size} bits, need >= {self.feature_length} for BCH."
            )

        u = np.random.uniform(0.0, 1.0, size=len(b_full))
        mask_r = (u >= self.kappa).astype(np.uint8)
        b_masked = (b_full & mask_r).astype(np.uint8)
        b_selected = b_masked[: self.feature_length]

        random_key = np.random.randint(
            0, 2, size=self.key_length, dtype=np.int64
        ).astype(np.uint8)
        codeword = self.ecc.encode(random_key).astype(np.uint8)

        helper_data = np.bitwise_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.tobytes()).digest()
        return helper_data, mask_r, key_hash

    def verify(
        self,
        feature_vector_float: np.ndarray,
        helper_data: np.ndarray,
        mask_r: np.ndarray,
        stored_key_hash: bytes,
    ) -> bool:
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        if b_full.size < self.feature_length:
            return False
        b_masked = (b_full & mask_r.astype(np.uint8)).astype(np.uint8)
        b_selected = b_masked[: self.feature_length]

        y_noisy = np.bitwise_xor(b_selected, helper_data.astype(np.uint8)).astype(np.uint8)

        try:
            reconstructed_key = self.ecc.decode(y_noisy)
        except Exception:
            return False
        reconstructed_key = np.asarray(reconstructed_key, dtype=np.uint8).reshape(
            self.key_length
        )

        recon_hash = hashlib.sha256(reconstructed_key.tobytes()).digest()
        return recon_hash == stored_key_hash
