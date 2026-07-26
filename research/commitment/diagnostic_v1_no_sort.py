"""
diagnostic_v1_no_sort.py

Giống hệt v1_selection_puncturing.py (SecureWiFaKeyHandler) — chọn 832 vị trí
ngẫu nhiên đều trong 1536 — CHỈ khác duy nhất 1 điểm: KHÔNG gọi
selection_indices.sort() trước khi gán vào codeword. Giữ nguyên thứ tự
permutation ngẫu nhiên từ rng.choice().

MỤC ĐÍCH: kiểm tra giả thuyết còn treo — việc luôn SẮP TĂNG DẦN thứ tự vị
trí trước khi gán vào codeword position 0..831 có ảnh hưởng đến khả năng
decode của Neural-MS hay không (so với thứ tự ngẫu nhiên hoàn toàn).

Cách đọc: nếu GMR ~ bằng v1_uniform (42.45%) => sort không ảnh hưởng, loại
giả thuyết này hẳn. Nếu khác biệt rõ rệt => sort có ảnh hưởng thật, cần điều
tra sâu hơn về cách Neural-MS phụ thuộc thứ tự gán bit.
"""

import hashlib

import numpy as np

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation


class NoSortSelectionWiFaKeyHandler(WiFaKeyHandler):
    def enroll(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full), size=self.feature_length, replace=False
        )
        # KHÔNG sort — giữ nguyên thứ tự ngẫu nhiên từ rng.choice()

        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1

        b_selected = b_full[selection_indices]  # thứ tự KHÔNG sort

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)

        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        # QUAN TRỌNG: phải lưu selection_indices (thứ tự gốc, KHÔNG sort)
        # để verify() dùng lại đúng thứ tự — không thể tái tạo từ
        # selection_mask (vì mask chỉ biết TẬP vị trí, không biết THỨ TỰ).
        # Vì vậy trả về selection_indices thay vì selection_mask ở đây.
        return helper_data, selection_indices, key_hash

    def verify(
        self, feature_vector_float, helper_data, selection_indices, stored_key_hash
    ):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        b_selected = b_full[selection_indices]  # đúng thứ tự đã lưu lúc enroll

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
