"""
verify_variant — tái hiện lại đúng luồng `WiFaKeyHandler.verify()` gốc,
NHƯNG cho phép hoán đổi bước modulation (và quantizer nếu cần) để A/B test.

Không sửa file wifakey_handler.py gốc. File này REUSE:
  - handler.M_matrix, handler.intervals  (dữ liệu, không phải logic)
  - handler.encoder, handler.sess, handler.xa, handler.decoder_output (qua decoder wrapper)

Chỉ thay thế bước: noisy_bits -> LLR (trước đây luôn là BPSK cứng).
"""

import hashlib
import numpy as np


def verify_with_variant(
    handler,  # WiFaKeyHandler gốc, đã init sẵn
    quantizer_fn,  # vd: binarize_with_confidence
    modulation,  # instance của BaseModulation (v0 hoặc v1...)
    decoder,  # instance NeuralMSOriginal
    feature_vector_float: np.ndarray,
    helper_data: np.ndarray,
    mask_r: np.ndarray,
    stored_key_hash: bytes,
):
    projected = np.dot(feature_vector_float, handler.M_matrix)

    bits, confidence = quantizer_fn(projected, handler.intervals)
    b_full = bits.astype(np.uint8)

    b_masked = (b_full & mask_r).astype(np.uint8)
    b_selected = b_masked[: handler.feature_length]

    # confidence cũng cần mask + cắt giống hệt b_full, để khớp vị trí bit
    conf_masked = confidence[: len(mask_r)] * mask_r
    conf_selected = conf_masked[: handler.feature_length]

    y_noisy_bits = np.logical_xor(b_selected, helper_data).astype(np.uint8)

    llr = modulation(y_noisy_bits, context={"distance": conf_selected})

    reconstructed_key = decoder.decode(llr)
    recon_hash = hashlib.sha256(reconstructed_key.astype(np.uint8).tobytes()).digest()

    return recon_hash == stored_key_hash, reconstructed_key
