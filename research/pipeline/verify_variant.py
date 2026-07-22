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
    # QUAN TRỌNG: KHÔNG nhân confidence với mask_r ở đây. Bit bị mask (mask_r=0)
    # bị ép về 0 một cách TẤT ĐỊNH (không phụ thuộc embedding), nên y_noisy tại
    # đó = helper_data (biết trước, đáng tin cậy TUYỆT ĐỐI) - ngược lại hoàn
    # toàn với "confidence thấp". Để modulation tự xử lý đúng ý nghĩa này,
    # truyền cả confidence gốc lẫn mask_r riêng biệt.
    confidence_selected = confidence[: handler.feature_length]
    mask_selected = mask_r[: handler.feature_length]

    y_noisy_bits = np.logical_xor(b_selected, helper_data)

    # LƯU Ý: key "distance"/"margin" ở đây chỉ là 2 TÊN GỌI cho CÙNG MỘT mảng
    # confidence_selected mà quantizer_fn trả về - ý nghĩa thật của mảng đó
    # (block-shared confidence hay per-bit margin) do quantizer_fn truyền vào
    # quyết định, không phải do verify_with_variant. Giữ cả 2 key để tương
    # thích ngược với v0/v1 (đọc "distance") lẫn v2_empirical_llr (đọc
    # "margin") mà không phải viết 2 nhánh code riêng.
    llr = modulation(
        y_noisy_bits,
        context={
            "distance": confidence_selected,
            "margin": confidence_selected,
            "mask": mask_selected,
        },
    )

    reconstructed_key = decoder.decode(llr)
    # KHÔNG ép kiểu uint8 ở đây - phải khớp CHÍNH XÁC cách handler.enroll() hash
    # random_key (random_key.flatten().tobytes(), giữ nguyên dtype=int gốc, int32
    # trên Windows). Ép sang uint8 làm đổi số byte/phần tử khi serialize, khiến
    # hash luôn lệch dù bit-value giống hệt nhau.
    recon_hash = hashlib.sha256(reconstructed_key.tobytes()).digest()

    return recon_hash == stored_key_hash, reconstructed_key
