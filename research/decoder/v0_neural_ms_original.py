"""
v0_neural_ms_original — BASELINE decoder, dùng lại NGUYÊN VẸN session TF1.x
đã build sẵn trong WiFaKeyHandler gốc. Không khởi tạo model mới, không sửa
trọng số — chỉ expose lại qua interface .decode(llr).

Việc "wrap" thay vì khởi tạo mới rất quan trọng: tránh tốn thêm VRAM khi
chạy A/B test (chỉ 1 session TF chạy cho cả baseline lẫn các bản so sánh
dùng chung decoder gốc).
"""

import numpy as np


class NeuralMSOriginal:
    name = "v0_neural_ms_original"

    def __init__(self, wifakey_handler):
        """
        wifakey_handler: instance CÓ SẴN của wifakey_module.WiFaKeyHandler
                         (đã load xong session + trọng số).
        """
        self._h = wifakey_handler

    def decode(self, llr: np.ndarray) -> np.ndarray:
        y_llr = llr.reshape((1, self._h.N, self._h.Z)).astype(np.float32)
        y_pred_llr = self._h.sess.run(
            self._h.decoder_output, feed_dict={self._h.xa: y_llr}
        )
        decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
        return decoded_codeword[: self._h.key_length]
