"""
wifakey_handler_secure_lda_variant.py

Giống hệt wifakey_handler_lda_variant.py về mục đích (hoán đổi self.M_matrix
sau khi __init__ gốc chạy xong), nhưng kế thừa SecureWiFaKeyHandler (selection-
puncturing, đã vá lỗ hổng AND-mask-về-0) thay vì WiFaKeyHandler gốc.

KHÔNG đụng vào wifakey_handler.py lẫn wifakey_handler_secure.py — chỉ kế thừa.

M_matrix_oracle_lda_regeps*.npy (do build_oracle_lda_matrix.py sinh ra) VẪN
DÙNG LẠI ĐƯỢC nguyên vẹn ở đây — ma trận đó được fit trên chính không gian
embedding AdaFace liên tục (512-d), không phụ thuộc gì vào cơ chế mask/
selection phía sau, nên không cần build lại.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler


class SecureWiFaKeyHandlerLDAVariant(SecureWiFaKeyHandler):
    """
    Giống hệt SecureWiFaKeyHandler (selection-puncturing, LDPC, decoder,
    intervals, kappa KHÔNG dùng tới — mọi thứ khác giữ nguyên), CHỈ khác
    đúng 1 điểm: self.M_matrix được nạp từ file mới thay vì M_matrix.npy gốc.
    """

    def __init__(self, m_matrix_override_path: str, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )  # load mọi thứ như SecureWiFaKeyHandler bình thường

        new_matrix = np.load(m_matrix_override_path)
        assert (
            new_matrix.shape == self.M_matrix.shape
        ), f"Shape M_matrix mới {new_matrix.shape} phải khớp bản gốc {self.M_matrix.shape}"
        ortho_err = float(
            np.max(np.abs(new_matrix @ new_matrix.T - np.eye(new_matrix.shape[0])))
        )
        assert ortho_err < 1e-6, (
            f"M_matrix mới không đủ trực giao (err={ortho_err:.2e}) — "
            f"kiểm tra lại build_oracle_lda_matrix.py trước khi dùng."
        )

        self.M_matrix = new_matrix
        print(
            f"[SecureWiFaKeyHandlerLDAVariant] Đã hoán đổi M_matrix -> {m_matrix_override_path}"
        )
