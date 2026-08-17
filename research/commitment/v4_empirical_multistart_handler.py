"""
v4_empirical_multistart_handler.py

Handler an toàn thay thế mask_r:
- Empirical LLR (v2) cho LLR đầu vào chính xác.
- Multi‑start Decoding (tùy chọn) để cứu thêm ca fail.
- Bảo mật: OTP hoàn hảo, khóa 160‑bit không suy giảm.
"""

import hashlib
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR
from wifakey_module.wifakey_lib import Modulation


class EmpiricalMultiStartHandler(SecureWiFaKeyHandler):
    def __init__(
        self, lookup_path=None, multi_start_K=0, multi_start_sigma=0.2, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if lookup_path is None:
            lookup_path = os.path.join(
                _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
            )
        self.empirical_llr = EmpiricalLLR(lookup_path=lookup_path)
        self.multi_start_K = multi_start_K
        self.multi_start_sigma = multi_start_sigma

    def _compute_llr(self, feature_vector_float, helper_data, selection_mask):
        """Tính LLR 832‑bit từ Empirical LLR."""
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        idx = np.where(selection_mask == 1)[0]
        b_sel = b_full[idx]
        noisy = np.logical_xor(b_sel, helper_data).astype(np.uint8)

        _, margin_all = binarize_with_perbit_confidence(
            np.dot(feature_vector_float, self.M_matrix), self.intervals
        )
        margin_sel = margin_all[idx]

        llr = self.empirical_llr.modulate(noisy, context={"margin": margin_sel})
        return llr.astype(np.float32)

    def verify(
        self, feature_vector_float, helper_data, selection_mask, stored_key_hash
    ):
        # Lần đầu tiên
        llr = self._compute_llr(feature_vector_float, helper_data, selection_mask)
        if self._try_decode(llr, stored_key_hash):
            return True
        if self.multi_start_K <= 0:
            return False

        # Multi‑start
        llr_clean = llr.reshape(1, self.N, self.Z)
        for _ in range(self.multi_start_K):
            noise = np.random.normal(
                0, self.multi_start_sigma, size=llr_clean.shape
            ).astype(np.float32)
            llr_noisy = llr_clean + noise
            y_pred = self.sess.run(
                self.decoder_output, feed_dict={self.xa: llr_noisy}
            ).flatten()
            decoded_key = (y_pred > 0).astype(int)[: self.key_length]
            if hashlib.sha256(decoded_key.tobytes()).digest() == stored_key_hash:
                return True
        return False

    def _try_decode(self, llr_flat, key_hash):
        """Giải mã một lần, kiểm tra hash."""
        llr = llr_flat.reshape(1, self.N, self.Z)
        y_pred = self.sess.run(self.decoder_output, feed_dict={self.xa: llr}).flatten()
        decoded_key = (y_pred > 0).astype(int)[: self.key_length]
        return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash
