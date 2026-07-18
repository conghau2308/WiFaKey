"""
13_compare_real_enroll_vs_manual_same_seed.py

Test 11 (gọi handler.enroll() THẬT) fail 5/5. Test 12 (tái hiện thủ công
CẢ enroll lẫn verify) thành công 10/10. Cùng logic, khác kết quả.

Script này CỐ ĐỊNH SEED trước khi gọi handler.enroll() THẬT, lưu lại
(helper_data, mask_r, key_hash). Sau đó RESET LẠI CÙNG SEED ĐÓ, tái hiện
thủ công enroll() để lấy (helper_data', mask_r', key_hash', codeword,
random_key). So sánh TRỰC TIẾP xem handler.enroll() THẬT có cho ra kết
quả giống hệt bản thủ công với CÙNG random draws hay không.

Cách chạy:
    python scripts/research/13_compare_real_enroll_vs_manual_same_seed.py
"""

import os
import sys
import hashlib
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)


def main():
    handler = WiFaKeyHandler()
    emb = np.load(os.path.join(CACHE_DIR, "Aaron_Guiel_0001.npy"))

    for trial in range(5):
        seed = 1000 + trial
        print(f"\n--- Trial {trial+1} (seed={seed}) ---")

        # ==== Gọi handler.enroll() THẬT với seed cố định ====
        np.random.seed(seed)
        helper_data_real, mask_r_real, key_hash_real = handler.enroll(emb)

        # ==== RESET lại CÙNG seed, tái hiện thủ công ====
        np.random.seed(seed)
        b_full = handler._binarize_full(emb).astype(np.uint8)
        u = np.random.uniform(0.0, 1.0, size=len(b_full))
        mask_r_manual = (u >= handler.kappa).astype(np.uint8)
        b_masked = (b_full & mask_r_manual).astype(np.uint8)
        b_selected = b_masked[: handler.feature_length]
        random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data_manual = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash_manual = hashlib.sha256(random_key.flatten().tobytes()).digest()

        # ==== So sánh ====
        mask_match = np.array_equal(mask_r_real, mask_r_manual)
        helper_match = np.array_equal(helper_data_real, helper_data_manual)
        hash_match = key_hash_real == key_hash_manual

        print(f"  mask_r:       THẬT == THỦ CÔNG? {mask_match}")
        print(f"  helper_data:  THẬT == THỦ CÔNG? {helper_match}")
        print(f"  key_hash:     THẬT == THỦ CÔNG? {hash_match}")

        if not (mask_match and helper_match and hash_match):
            print("  ❌❌❌ handler.enroll() THẬT cho kết quả KHÁC bản thủ công")
            print("       dù dùng CÙNG SEED! Có sự khác biệt ẩn trong cách gọi qua")
            print("       instance method so với tái hiện thủ công.")
        else:
            print("  ✅ handler.enroll() THẬT khớp hoàn toàn với thủ công (cùng seed).")

        # ==== Bây giờ verify bằng handler.verify() THẬT với output THẬT của enroll ====
        result = handler.verify(emb, helper_data_real, mask_r_real, key_hash_real)
        print(f"  handler.verify() với output THẬT của enroll(): {result}")


if __name__ == "__main__":
    main()
