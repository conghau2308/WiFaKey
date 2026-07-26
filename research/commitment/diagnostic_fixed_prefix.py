"""
diagnostic_fixed_prefix.py

CHỈ DÙNG ĐỂ CHẨN ĐOÁN, KHÔNG DEPLOY — handler này luôn chọn CỐ ĐỊNH đúng
832 vị trí đầu tiên [0:832) của không gian nhị phân (giống hệt cách baseline
cũ định vị trí, KHÔNG random hoá mỗi lần enroll), nhưng dùng bit sinh trắc
THẬT thay vì AND-về-0 (giữ đúng tính chất OTP như v1/v2). Vì selection không
đổi giữa các lần enroll, handler này KHÔNG có cancelability/unlinkability —
tuyệt đối không dùng cho sản phẩm thật.

MỤC ĐÍCH DUY NHẤT: cô lập câu hỏi — "GMR thấp của v1 (42.45%) có phải một
phần do việc ÁNH XẠ VỊ TRÍ THAY ĐỔI ngẫu nhiên mỗi lần enroll (bất kể chất
lượng bit), hay hoàn toàn do đặc tính nhiễu của chính 832 bit được chọn?"

Cách đọc kết quả (so với v1_uniform=42.45%, baseline_and_mask=95.69% đã có):
  - Nếu GMR của handler này gần bằng v1 (~42%) => ánh xạ vị trí cố định hay
    ngẫu nhiên KHÔNG quan trọng; vấn đề nằm thuần ở đặc tính nhiễu/kích
    thước tập vị trí [0:832) này (dù cố định hay không, decoder xử lý y hệt).
  - Nếu GMR của handler này CAO HƠN RÕ RỆT v1 (gần baseline hơn) => ánh xạ
    vị trí có ảnh hưởng thật đến decoder (có thể do đặc tính riêng của
    chính dải [0:832) — ví dụ ít nhiễu hơn phần còn lại của không gian —
    hoặc do bản thân việc thay đổi ánh xạ mỗi lần làm decoder khó xử lý
    hơn) => cần điều tra thêm CÁI GÌ trong dải [0:832) khiến nó đặc biệt.
"""

import hashlib

import numpy as np

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation


class FixedPrefixWiFaKeyHandler(WiFaKeyHandler):
    def enroll(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        selection_indices = np.arange(
            self.feature_length
        )  # LUÔN CỐ ĐỊNH — chỉ để chẩn đoán
        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1

        b_selected = b_full[selection_indices]

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)

        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        return helper_data, selection_mask, key_hash

    def verify(
        self, feature_vector_float, helper_data, selection_mask, stored_key_hash
    ):
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
