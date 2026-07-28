"""
wifakey_handler_lda_variant.py

Wrapper KHÔNG đụng vào wifakey_module/wifakey_handler.py gốc — chỉ kế thừa
và hoán đổi self.M_matrix SAU KHI __init__ gốc đã chạy xong (đã load xong
LDPC encoder, decoder Neural-MS, binarization_intervals — những phần này
giữ nguyên, không đổi gì).

CHỈ dùng để TEST — nếu kết quả FRR/FAR thật khả quan, hãy cân nhắc việc
chính thức thay M_matrix.npy gốc (và re-enroll toàn hệ thống) như 1 bước
riêng, có chủ đích, không lặng lẽ qua wrapper này.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from wifakey_module.wifakey_handler import (
    WiFaKeyHandler,
)


class WiFaKeyHandlerLDAVariant(WiFaKeyHandler):
    """
    Giống hệt WiFaKeyHandler gốc (LDPC, decoder, intervals, kappa, mask —
    KHÔNG đổi gì), CHỈ khác đúng 1 điểm: self.M_matrix được nạp từ file mới
    (512x512 trực giao, do build_oracle_lda_matrix.py sinh ra) thay vì
    M_matrix.npy gốc trong data_path.
    """

    def __init__(self, m_matrix_override_path: str, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )  # load mọi thứ như bình thường, kể cả M_matrix gốc

        new_matrix = np.load(m_matrix_override_path)
        assert (
            new_matrix.shape == self.M_matrix.shape
        ), f"Shape M_matrix mới {new_matrix.shape} phải khớp bản gốc {self.M_matrix.shape}"
        # Self-check: xác nhận vẫn trực giao trước khi thay — tránh silently
        # dùng 1 ma trận hỏng làm lệch toàn bộ kết quả benchmark.
        ortho_err = float(
            np.max(np.abs(new_matrix @ new_matrix.T - np.eye(new_matrix.shape[0])))
        )
        assert ortho_err < 1e-6, (
            f"M_matrix mới không đủ trực giao (err={ortho_err:.2e}) — "
            f"kiểm tra lại build_oracle_lda_matrix.py trước khi dùng."
        )

        self.M_matrix = (
            new_matrix  # ghi đè ĐÚNG 1 thuộc tính này, mọi thứ khác giữ nguyên
        )
        print(
            f"[WiFaKeyHandlerLDAVariant] Đã hoán đổi M_matrix -> {m_matrix_override_path}"
        )
