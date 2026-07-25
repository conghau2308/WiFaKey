"""
wifakey_handler_secure.py

Bản vá lỗ hổng "AND-mask-về-0 rồi XOR" đã xác nhận thực nghiệm (G systematic
=> delta lộ trực tiếp c tại các vị trí mask=0 => khôi phục k 100% qua Gaussian
elimination, xem verify_encoder_leak.py).

Nguyên tắc vá: chuyển từ MASKING (AND về 0, giữ nguyên độ dài, một số vị trí bị
ép về hằng số biết trước) sang SELECTION/PUNCTURING (chọn đúng feature_length
vị trí THẬT từ b_full, không có vị trí nào bị ép về hằng số). Nhờ vậy:

    helper_data_i = b_selected_i XOR codeword_i

luôn là XOR giữa 1 bit sinh trắc thật (chưa biết với attacker) và 1 bit mã hóa,
tại MỌI vị trí — tức One-Time-Pad đúng nghĩa toàn phần, không còn vị trí nào
lộ trực tiếp giá trị codeword như bản cũ.

File này KHÔNG sửa wifakey_handler.py gốc — chỉ kế thừa WiFaKeyHandler và
override enroll()/verify(). Toàn bộ hạ tầng khác (encoder LDPC, TF session,
Neural-MS decoder, binarize, M_matrix, intervals...) được tái sử dụng nguyên
vẹn từ lớp cha qua super().__init__(), không load lại, không tốn thêm chi phí
khởi tạo GPU/model.

Đặt file này trong research/commitment/ — category mới, ngang hàng với
research/modulation/, research/quantizer/, research/decoder/ đã có sẵn trong
project của bạn, theo đúng convention v0_baseline / v1_variant:

    project_root/
        wifakey_module/
            wifakey_handler.py      <- gốc, không đổi
            wifakey_lib/
            data/
        research/
            commitment/
                v1_selection_puncturing.py   <- file này
            modulation/
            quantizer/
            decoder/

File import wifakey_module tuyệt đối (from wifakey_module.wifakey_handler
import WiFaKeyHandler) vì giờ nằm khác package, không còn quan hệ package
tương đối (relative import) với wifakey_module/ nữa. Cần chạy python với
project_root là working directory / trên PYTHONPATH để import này resolve
đúng — giống cách research/modulation/, research/decoder/ hiện tại của bạn
chắc cũng đang import wifakey_module theo kiểu tuyệt đối tương tự.

LƯU Ý VỀ TƯƠNG THÍCH NGƯỢC:
- API thay đổi: enroll() giờ trả về (helper_data, selection_mask, key_hash)
  thay vì (helper_data, mask_r, key_hash). selection_mask có cùng độ dài với
  b_full (full_binary_length), nhưng ý nghĩa khác: 1 = vị trí này được CHỌN
  tham gia commit, 0 = vị trí này không dùng (không phải "bị che về 0" như
  trước — nó đơn giản là không được chọn vào tập feature_length).
- selection_mask vẫn có thể lưu/truyền công khai như mask_r cũ (không vi phạm
  Kerckhoffs) vì nó chỉ cho biết VỊ TRÍ nào tham gia, không tiết lộ GIÁ TRỊ bit
  nào — khác bản chất với mask_r cũ, nơi vị trí mask=0 tất định bằng 0 và làm
  lộ trực tiếp c qua XOR.
- Do đó, endpoint/DB nào đang lưu "mask_b64" theo API cũ cần đổi tên field
  (gợi ý: "selection_mask_b64") để tránh nhầm lẫn khi đọc lại — xem
  main_secure.py đi kèm.

GIỚI HẠN CẦN LƯU Ý:
- Cần full_binary_length > feature_length để có "dư" vị trí mà chọn ngẫu
  nhiên — nếu không, mọi lần enroll đều chọn y hệt toàn bộ b_full (selection
  suy biến về full-length, mất tác dụng làm giảm noise/BER mà mask cũ vốn có
  qua tham số kappa). Kiểm tra self.full_binary_length > self.feature_length
  khi khởi tạo nếu muốn chắc chắn.
- kappa (thuộc tính kế thừa từ lớp cha) KHÔNG còn được dùng trong enroll/verify
  của lớp này — tỷ lệ "giữ lại" giờ là feature_length / full_binary_length cố
  định theo cấu hình hệ thống (N=52, Z=16 => feature_length=832), không còn
  điều khiển qua kappa nữa. Nếu muốn tái tạo hiệu ứng "kappa cao hơn = giữ ít
  hơn, chọn lọc hơn", có thể mở rộng bằng cách trước tiên chọn một tập ứng
  viên lớn hơn feature_length theo trọng số reliability (Fisher/percentile-
  rank đã bàn trước đó), rồi mới chọn ngẫu nhiên feature_length trong tập đó
  — đây là điểm mở rộng tự nhiên cho các đề xuất reliability-weighting sau
  này, không nằm trong phạm vi bản vá tối thiểu này.
"""

import hashlib

import numpy as np

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation


class SecureWiFaKeyHandler(WiFaKeyHandler):
    """
    Drop-in thay thế cho WiFaKeyHandler, chỉ khác cơ chế enroll()/verify().
    Khởi tạo giống hệt lớp cha:

        handler = SecureWiFaKeyHandler(
            data_path=...,
            weights_path=...,
            biases_path=...,
        )
    """

    def enroll(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        if len(b_full) <= self.feature_length:
            raise ValueError(
                f"full_binary_length ({len(b_full)}) phải LỚN HƠN feature_length "
                f"({self.feature_length}) để selection có ý nghĩa. Kiểm tra lại "
                f"M_matrix / binarization_intervals."
            )

        rng = np.random.default_rng()
        # Chọn ngẫu nhiên, KHÔNG lặp lại, đúng feature_length vị trí thật.
        selection_indices = rng.choice(
            len(b_full), size=self.feature_length, replace=False
        )
        selection_indices.sort()

        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1

        b_selected = b_full[selection_indices]  # toàn bộ là bit sinh trắc thật

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)

        # OTP tại MỌI vị trí -- không còn delta_i == codeword_i tất định.
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)

        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        return helper_data, selection_mask, key_hash

    def verify(
        self,
        feature_vector_float: np.ndarray,
        helper_data: np.ndarray,
        selection_mask: np.ndarray,
        stored_key_hash: bytes,
    ) -> bool:
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        if len(selection_mask) != len(b_full):
            raise ValueError(
                f"selection_mask length {len(selection_mask)} != "
                f"full_binary_length {len(b_full)}. Dữ liệu client gửi có thể "
                f"không khớp cấu hình server (M_matrix/intervals khác nhau?)."
            )

        selection_indices = np.where(selection_mask == 1)[0]
        if len(selection_indices) != self.feature_length:
            raise ValueError(
                f"selection_mask có {len(selection_indices)} vị trí được chọn, "
                f"kỳ vọng đúng {self.feature_length}. Có thể selection_mask bị "
                f"hỏng/giả mạo."
            )

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

        if recon_hash == stored_key_hash:
            print("[SecureWiFaKey] Verify SUCCESS.")
            return True
        else:
            print("[SecureWiFaKey] Verify FAILED.")
            return False
