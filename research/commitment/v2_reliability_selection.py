"""
v2_reliability_selection.py

Nâng cấp research/commitment/v1_selection_puncturing.py với 2 thay đổi:

1. CSPRNG cho random_key (thay np.random.randint / Mersenne Twister bằng
   secrets.token_bytes) — vì đây là khóa bí mật thật của hệ thống.
   (selection_indices vẫn dùng PRNG thường (np.random.default_rng) vì
   selection_mask công khai, không cần CSPRNG.)

2. Pool reliability-weighted selection: thay vì rng.choice đều trên toàn bộ
   full_binary_length vị trí, trước tiên lọc ra pool M vị trí có Fisher-score
   population cao nhất (M > feature_length), rồi chọn ngẫu nhiên
   feature_length vị trí TRONG pool đó. Tham số pool_size thay thế vai trò
   của kappa cũ — pool_size càng gần feature_length thì chọn lọc càng mạnh
   (BER thấp hơn) nhưng reroll ít hơn; pool_size càng lớn thì reroll nhiều
   hơn nhưng BER nhích lên.

reliability_scores (Fisher F_i, xem compute_reliability_scores.py đi kèm)
PHẢI được tính 1 lần offline trên tập calibration (population-level), không
bao giờ dùng dữ liệu riêng của từng user tại thời điểm enroll — nếu không sẽ
làm lộ đặc điểm phương sai sinh trắc riêng của user qua chính selection_mask
công khai (phá unlinkability).

Giữ nguyên interface enroll()/verify() như v1 để main_secure.py không cần
đổi gì thêm ngoài import.
"""

import hashlib

import numpy as np
import secrets

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation


class ReliabilitySelectionWiFaKeyHandler(WiFaKeyHandler):
    """
    Kế thừa WiFaKeyHandler gốc (không sửa file gốc). So với
    SecureWiFaKeyHandler (v1), thêm reliability_scores + pool_size để chọn
    vị trí có chủ đích thay vì đều.

    Khởi tạo:
        handler = ReliabilitySelectionWiFaKeyHandler(
            data_path=..., weights_path=..., biases_path=...,
            reliability_scores_path="path/to/reliability.npy",  # array (full_binary_length,)
            pool_size=1200,   # M > feature_length; tune trên tune_*.csv
        )
    """

    def __init__(self, *args, reliability_scores_path: str, pool_size: int, **kwargs):
        super().__init__(*args, **kwargs)

        F = np.load(reliability_scores_path)
        if F.shape[0] != self.full_binary_length:
            raise ValueError(
                f"reliability_scores length {F.shape[0]} != "
                f"full_binary_length {self.full_binary_length}. "
                f"Kiểm tra lại tập calibration dùng để tính F_i có cùng "
                f"M_matrix/intervals với handler hiện tại không."
            )
        if pool_size <= self.feature_length:
            raise ValueError(
                f"pool_size ({pool_size}) phải LỚN HƠN feature_length "
                f"({self.feature_length}) để còn khoảng ngẫu nhiên (reroll)."
            )
        if pool_size > self.full_binary_length:
            raise ValueError(
                f"pool_size ({pool_size}) không được vượt quá "
                f"full_binary_length ({self.full_binary_length})."
            )

        self.reliability_scores = F
        self.pool_size = pool_size
        # Pool cố định: M vị trí có F_i cao nhất, tính 1 lần lúc khởi tạo.
        # Đây là tham số population-level, công khai, giống hệt binarization_intervals.npy.
        self.reliability_pool = np.argsort(-F)[:pool_size]

    def _sample_selection_indices(self) -> np.ndarray:
        rng = np.random.default_rng()  # PRNG thường đủ dùng — mask công khai
        selection_indices = rng.choice(
            self.reliability_pool, size=self.feature_length, replace=False
        )
        selection_indices.sort()
        return selection_indices

    @staticmethod
    def _generate_csprng_key(key_length: int) -> np.ndarray:
        """Sinh random_key bằng CSPRNG (secrets), không dùng Mersenne Twister."""
        num_bytes = (key_length + 7) // 8
        raw = secrets.token_bytes(num_bytes)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:key_length]
        return bits.reshape(1, -1).astype(int)

    def enroll(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        if len(b_full) != self.full_binary_length:
            raise ValueError(
                f"b_full length {len(b_full)} != full_binary_length "
                f"{self.full_binary_length} — kiểm tra lại cấu hình."
            )

        selection_indices = self._sample_selection_indices()
        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1

        b_selected = b_full[selection_indices]

        random_key = self._generate_csprng_key(self.key_length)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        assert len(codeword) == self.feature_length, (
            f"codeword length {len(codeword)} != feature_length "
            f"{self.feature_length} — lỗi cấu hình encoder."
        )

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
                f"full_binary_length {len(b_full)}."
            )

        selection_indices = np.where(selection_mask == 1)[0]
        if len(selection_indices) != self.feature_length:
            raise ValueError(
                f"selection_mask có {len(selection_indices)} vị trí, kỳ vọng "
                f"đúng {self.feature_length}."
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
            print("[ReliabilitySelectionWiFaKey] Verify SUCCESS.")
            return True
        else:
            print("[ReliabilitySelectionWiFaKey] Verify FAILED.")
            return False
