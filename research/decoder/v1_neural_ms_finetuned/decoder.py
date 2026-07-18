"""
decoder.py - Wrapper cho decoder ĐÃ FINE-TUNE, cùng interface với
NeuralMSOriginal (research/decoder/v0_neural_ms_original.py) để có thể
hoán đổi trực tiếp trong pipeline A/B test (verify_variant.py) mà không
sửa gì thêm.

Khác với NeuralMSOriginal (dùng lại session của WiFaKeyHandler đã có sẵn),
class này tự dựng session RIÊNG với trọng số đã fine-tune (không đụng gì
đến handler gốc).
"""

import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
FINETUNED_WEIGHTS_PATH = os.path.join(_HERE, "weights", "Weights_Var_MS_finetuned")
FINETUNED_BIASES_PATH = os.path.join(_HERE, "weights", "Biases_Var_MS_finetuned")


class NeuralMSFinetuned:
    name = "v1_neural_ms_finetuned"

    def __init__(self, handler):
        """
        handler: WiFaKeyHandler gốc - chỉ dùng để lấy N/Z/key_length/
        code_PCM_base (thông số cấu trúc mã, không dùng session/trọng số
        gốc của nó).
        """
        from train import build_trainable_decoder  # reuse graph-builder

        self.N = handler.N
        self.Z = handler.Z
        self.key_length = handler.key_length

        tg = build_trainable_decoder(
            batch_size=1,
            init_weights_path=FINETUNED_WEIGHTS_PATH,
            init_biases_path=FINETUNED_BIASES_PATH,
        )
        self._xa = tg["xa"]
        self._decoder_output = tg["net_dict"][f"ya_output24"]  # ITERS_MAX-1 = 24

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=tg["graph"], config=config)
        self.sess.run(tg["init_op"])

    def decode(self, llr: np.ndarray) -> np.ndarray:
        y_llr = llr.reshape((1, self.N, self.Z)).astype(np.float32)
        y_pred_llr = self.sess.run(self._decoder_output, feed_dict={self._xa: y_llr})
        decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
        return decoded_codeword[: self.key_length]

    def __del__(self):
        if hasattr(self, "sess"):
            self.sess.close()
