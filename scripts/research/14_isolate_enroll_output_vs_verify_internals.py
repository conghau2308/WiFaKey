"""
11_isolate_enroll_output_vs_verify_internals.py

Mục tiêu: xác định CHÍNH XÁC verify() thật khác với tái hiện thủ công ở
bước nào, bằng cách dùng CÙNG helper_data/mask_r/key_hash (lấy từ MỘT
lần gọi enroll() thật DUY NHẤT), sau đó:

  (A) Gọi handler.verify() THẬT với các giá trị này.
  (B) Tái hiện THỦ CÔNG (copy nguyên văn từng dòng của verify()) với
      CÙNG helper_data/mask_r/key_hash, in ra từng giá trị trung gian.

Nếu (A) và (B) cho kết quả KHÁC NHAU dù dùng chung input, nghĩa là có gì
đó bên trong lời gọi hàm thật (self.sess, self.decoder_output...) khác
với những gì đoạn code thủ công (copy y hệt) tính ra - dấu hiệu của
non-determinism (GPU/TF) hoặc trạng thái ẩn nào đó.

Cách chạy:
    python scripts/research/11_isolate_enroll_output_vs_verify_internals.py
"""

import os
import sys
import hashlib
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)


def manual_verify_with_trace(
    handler, feature_vector_float, helper_data, mask_r, stored_key_hash
):
    """Copy NGUYÊN VĂN từng dòng của handler.verify(), chỉ thêm print."""
    b_full = handler._binarize_full(feature_vector_float).astype(np.uint8)
    b_masked = (b_full & mask_r).astype(np.uint8)
    b_selected = b_masked[: handler.feature_length]
    y_noisy_bits = np.logical_xor(b_selected, helper_data)

    y_llr = Modulation.BPSK(y_noisy_bits).reshape((1, handler.N, handler.Z))
    y_pred_llr = handler.sess.run(handler.decoder_output, feed_dict={handler.xa: y_llr})

    decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
    reconstructed_key = decoded_codeword[: handler.key_length]
    recon_hash = hashlib.sha256(reconstructed_key.tobytes()).digest()

    print(
        f"    [thủ công] y_noisy_bits sum(bit=1): {y_noisy_bits.sum()}/{len(y_noisy_bits)}"
    )
    print(
        f"    [thủ công] reconstructed_key sum(bit=1): {reconstructed_key.sum()}/{len(reconstructed_key)}"
    )
    print(
        f"    [thủ công] recon_hash == stored_key_hash: {recon_hash == stored_key_hash}"
    )

    return recon_hash == stored_key_hash, y_noisy_bits, reconstructed_key


def main():
    handler = WiFaKeyHandler()
    emb = np.load(os.path.join(CACHE_DIR, "Aaron_Guiel_0001.npy"))

    for trial in range(5):
        print(f"\n--- Trial {trial+1} ---")

        # MỘT lần enroll() THẬT DUY NHẤT cho trial này
        helper_data, mask_r, key_hash = handler.enroll(emb)

        # (A) Gọi verify() THẬT
        result_real = handler.verify(emb, helper_data, mask_r, key_hash)
        print(f"    [verify() THẬT] kết quả: {result_real}")

        # (B) Tái hiện thủ công với CÙNG helper_data/mask_r/key_hash
        result_manual, y_noisy_bits, reconstructed_key = manual_verify_with_trace(
            handler, emb, helper_data, mask_r, key_hash
        )

        print(
            f"    => (A) THẬT={result_real}  vs  (B) THỦ CÔNG={result_manual}  "
            f"{'✅ KHỚP' if result_real == result_manual else '❌ KHÁC NHAU!'}"
        )

        if result_real != result_manual:
            print(
                "    ❌❌❌ PHÁT HIỆN: verify() THẬT và tái hiện thủ công (cùng input)"
            )
            print("         cho ra 2 kết quả KHÁC NHAU -> non-determinism thực sự")
            print("         (rất có thể từ GPU/TensorFlow sess.run()).")


if __name__ == "__main__":
    main()
