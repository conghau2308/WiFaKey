"""
v1_reduced_key_length.py

C.5 — Hạ "code rate hiệu dụng" bằng cách CHỈ yêu cầu đúng 128/160 bit khoá
sau decode, thay vì đòi hỏi cả 160 bit đúng như hiện tại. KHÔNG đổi gì ở
encoder/decoder/G matrix (không cần retrain Neural-MS) — chỉ đổi cách tính
và so khớp hash.

TRÁNH kỹ thuật "shortening" kiểu cố định 32 bit message về hằng số (0) --
vì G là systematic ([I_160|P], đã xác nhận qua verify_encoder_leak.py),
cố định bit message sẽ làm lộ trực tiếp bit sinh trắc thật tại các vị trí
codeword tương ứng (cùng họ lỗi với lỗ hổng AND-mask đã vá, chỉ nhẹ hơn --
lộ 1 phần biometric, không lộ key).

Cách làm AN TOÀN ở đây: vẫn sinh đủ 160 bit random_key THẬT (CSPRNG, không
cố định gì), codeword/helper_data giữ nguyên 100% logic v1 (OTP đầy đủ,
không lộ gì mới) -- chỉ đổi việc hash: key_hash chỉ tính trên 128 bit đầu
của random_key, 32 bit còn lại vẫn ngẫu nhiên thật nhưng không được kiểm
tra đúng/sai lúc verify. Vì P(128 bit cụ thể đúng) >= P(cả 160 bit đúng),
GMR tăng hợp lệ mà không cần đổi gì về mã hoá/giải mã.

Entropy còn lại: 2^128 -- vẫn thừa an toàn cho brute-force (theo đúng nhận
định ban đầu của đề xuất C.5).

Kế thừa trực tiếp SecureWiFaKeyHandler (v1) để cô lập đúng 1 biến: chỉ khác
số bit được hash, mọi thứ khác (selection, encoder, decoder) giữ nguyên.
"""

import hashlib

import numpy as np

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler


class ReducedKeyLengthWiFaKeyHandler(SecureWiFaKeyHandler):
    def __init__(self, *args, effective_key_length: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        if effective_key_length > self.key_length:
            raise ValueError(
                f"effective_key_length ({effective_key_length}) không được lớn hơn "
                f"key_length gốc ({self.key_length})."
            )
        self.effective_key_length = effective_key_length

    def enroll(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full), size=self.feature_length, replace=False
        )
        selection_indices.sort()

        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1

        b_selected = b_full[selection_indices]

        # random_key vẫn ĐỦ key_length bit, HOÀN TOÀN ngẫu nhiên thật --
        # không cố định bit nào -- giữ nguyên tính OTP của helper_data.
        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)

        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)

        # CHỈ hash effective_key_length bit đầu -- đây là điểm khác biệt
        # duy nhất so với v1.
        effective_key = random_key.flatten()[: self.effective_key_length]
        key_hash = hashlib.sha256(effective_key.tobytes()).digest()

        return helper_data, selection_mask, key_hash

    def verify(
        self, feature_vector_float, helper_data, selection_mask, stored_key_hash
    ) -> bool:
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        selection_indices = np.where(selection_mask == 1)[0]

        b_selected = b_full[selection_indices]
        y_noisy_bits = np.logical_xor(b_selected, helper_data)

        from wifakey_module.wifakey_lib import Modulation

        y_llr = (
            Modulation.BPSK(y_noisy_bits)
            .astype(np.float32)
            .reshape((1, self.N, self.Z))
        )
        y_pred_llr = self.sess.run(self.decoder_output, feed_dict={self.xa: y_llr})

        decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
        reconstructed_key = decoded_codeword[: self.key_length]

        # CHỈ so khớp effective_key_length bit đầu -- điểm khác biệt duy nhất.
        effective_reconstructed = reconstructed_key[: self.effective_key_length]
        recon_hash = hashlib.sha256(effective_reconstructed.tobytes()).digest()

        return recon_hash == stored_key_hash
