"""
v1_multisample.py

Majority-vote enrollment (C.2): nhận K embedding của CÙNG 1 người, binarize
từng cái riêng biệt, vote per-bit (giá trị xuất hiện nhiều nhất trong K bản)
ra 1 vector b_full "đồng thuận" duy nhất, rồi tiếp tục enroll như v1
(SecureWiFaKeyHandler: uniform selection, an toàn, đã vá lỗ hổng AND-mask).

Chỉ khác v1 ở NGUỒN của b_full (vote từ K ảnh thay vì binarize 1 ảnh) —
phần selection/commitment phía sau giữ nguyên logic v1 để so sánh công bằng
(cô lập đúng 1 biến: có vote hay không).

verify() giữ nguyên như v1 — verify luôn dùng ĐÚNG 1 ảnh (đúng thực tế sử
dụng, người dùng không quét nhiều ảnh mỗi lần xác thực).
"""

import hashlib

import numpy as np

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation


class MultisampleWiFaKeyHandler(WiFaKeyHandler):
    def _majority_vote_binarize(self, feature_vectors_float: list) -> np.ndarray:
        binarized = [
            self._binarize_full(fv).astype(np.uint8) for fv in feature_vectors_float
        ]
        stacked = np.stack(binarized, axis=0)  # (K, full_binary_length)
        K = stacked.shape[0]
        votes_sum = stacked.sum(axis=0)

        majority = (votes_sum * 2 > K).astype(np.uint8)

        if K % 2 == 0:  # K chẵn -> có thể hòa phiếu, cần xử lý riêng
            ties = votes_sum * 2 == K
            if ties.any():
                rng = np.random.default_rng()
                majority[ties] = rng.integers(
                    0, 2, size=int(ties.sum()), dtype=np.uint8
                )

        return majority

    def enroll_multisample(self, feature_vectors_float: list):
        if len(feature_vectors_float) < 1:
            raise ValueError("Cần ít nhất 1 embedding để enroll.")

        b_full = self._majority_vote_binarize(feature_vectors_float)

        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full), size=self.feature_length, replace=False
        )
        selection_indices.sort()

        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1

        b_selected = b_full[selection_indices]

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)

        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        return helper_data, selection_mask, key_hash

    def verify(
        self,
        feature_vector_float: np.ndarray,
        helper_data,
        selection_mask,
        stored_key_hash,
    ) -> bool:
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        selection_indices = np.where(selection_mask == 1)[0]

        b_selected = b_full[selection_indices]
        y_noisy_bits = np.logical_xor(b_selected, helper_data)

        y_llr = (
            Modulation.BPSK(y_noisy_bits)
            .astype(np.float32)
            .reshape((1, self.N, self.Z))
        )
        y_pred_llr = self.sess.run(self.decoder_output, feed_dict={self.xa: y_llr})

        decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
        reconstructed_key = decoded_codeword[: self.key_length]
        recon_hash = hashlib.sha256(reconstructed_key.tobytes()).digest()

        return recon_hash == stored_key_hash
