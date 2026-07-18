"""
09_manual_enroll_verify_ground_truth.py

Test trước cho thấy nghịch lý: b_full deterministic, mask_r khớp độ dài,
nhưng self-match vẫn fail. Về lý thuyết y_noisy_bits PHẢI bằng đúng
codeword nội bộ của enroll() - nhưng ta không thể thấy codeword đó vì
handler.enroll() không trả nó ra (chỉ trả helper_data/mask_r/key_hash).

Script này TỰ TÁI HIỆN thủ công từng dòng của enroll() và verify()
(dùng CHÍNH các hàm/thuộc tính nội bộ của handler: _binarize_full,
handler.encoder, handler.kappa...), để lấy được codeword/random_key
THẬT SỰ được dùng, rồi so sánh trực tiếp:

  1. b_selected(enroll) so với b_selected(verify) - phải giống hệt.
  2. y_noisy_bits so với codeword gốc - phải giống hệt NẾU (1) đúng.
  3. Nếu (2) sai dù (1) đúng -> lỗi nằm ở helper_data hoặc phép XOR.
  4. Giải mã y_noisy_bits và so sánh TRỰC TIẾP (không qua SHA256) với
     random_key gốc - để loại trừ khả năng lỗi nằm ở khâu hash/so sánh.

Cách chạy:
    python scripts/research/09_manual_enroll_verify_ground_truth.py
"""

import os
import sys
import hashlib
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation as orig_modulation

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)


def load_one_embedding():
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    return np.load(os.path.join(CACHE_DIR, files[0])), files[0]


def main():
    np.random.seed(42)  # cố định để tái lập được
    handler = WiFaKeyHandler()
    emb, fname = load_one_embedding()
    print(f"Dùng embedding: {fname}\n")

    # ==================== TÁI HIỆN THỦ CÔNG enroll() ====================
    b_full = handler._binarize_full(emb).astype(np.uint8)

    u = np.random.uniform(0.0, 1.0, size=len(b_full))
    mask_r = (u >= handler.kappa).astype(np.uint8)

    b_masked_enroll = (b_full & mask_r).astype(np.uint8)
    b_selected_enroll = b_masked_enroll[: handler.feature_length]

    random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
    codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)

    helper_data = np.logical_xor(b_selected_enroll, codeword).astype(np.uint8)
    key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

    print(f"random_key dtype gốc: {random_key.dtype}")
    print(f"codeword: {len(codeword)} bit, số bit=1: {codeword.sum()}")
    print(
        f"b_selected_enroll: {len(b_selected_enroll)} bit, số bit=1: {b_selected_enroll.sum()}"
    )
    print(f"helper_data: {len(helper_data)} bit\n")

    # ==================== TÁI HIỆN THỦ CÔNG verify() (self-match) ====================
    b_full_v = handler._binarize_full(emb).astype(np.uint8)
    b_masked_verify = (b_full_v & mask_r).astype(np.uint8)
    b_selected_verify = b_masked_verify[: handler.feature_length]

    print(
        f"b_selected(enroll) == b_selected(verify): "
        f"{np.array_equal(b_selected_enroll, b_selected_verify)}"
    )

    y_noisy_bits = np.logical_xor(b_selected_verify, helper_data)
    print(
        f"y_noisy_bits == codeword gốc: {np.array_equal(y_noisy_bits.astype(np.uint8), codeword)}"
    )

    if not np.array_equal(y_noisy_bits.astype(np.uint8), codeword):
        diff = np.sum(y_noisy_bits.astype(np.uint8) != codeword)
        print(
            f"❌ KHÁC NHAU ở {diff}/{len(codeword)} vị trí - ĐÂY LÀ BẰNG CHỨNG TRỰC TIẾP"
        )
        print("   dù b_selected(enroll)==b_selected(verify), y_noisy_bits vẫn KHÔNG")
        print("   bằng codeword -> lỗi nằm ở chính phép XOR hoặc ở helper_data,")
        print("   KHÔNG phải ở b_full/mask/binarization nữa.")
    else:
        print("✅ y_noisy_bits khớp CHÍNH XÁC với codeword gốc.")

    # ==================== Giải mã trực tiếp, so sánh KHÔNG qua SHA256 ====================
    from research.decoder.v0_neural_ms_original import NeuralMSOriginal

    decoder = NeuralMSOriginal(handler)

    llr = orig_modulation.BPSK(y_noisy_bits).astype(np.float32)
    reconstructed_key = decoder.decode(llr)

    print(
        f"\nreconstructed_key == random_key gốc (so sánh TRỰC TIẾP, không qua hash): "
        f"{np.array_equal(reconstructed_key, random_key.flatten())}"
    )

    recon_hash = hashlib.sha256(reconstructed_key.astype(np.uint8).tobytes()).digest()
    print(f"recon_hash == key_hash gốc (so sánh qua SHA256): {recon_hash == key_hash}")

    if (
        np.array_equal(reconstructed_key, random_key.flatten())
        and recon_hash != key_hash
    ):
        print("\n❌ Key giải mã ĐÚNG hệt random_key, nhưng SHA256 hash KHÔNG khớp!")
        print("   -> Lỗi nằm ở KHÂU HASH/so sánh dtype khi .tobytes() (rất có thể do")
        print("      random_key.dtype khác reconstructed_key.dtype khi serialize).")


if __name__ == "__main__":
    main()
