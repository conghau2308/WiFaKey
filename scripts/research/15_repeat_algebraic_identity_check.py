"""
12_repeat_algebraic_identity_check.py

Test 09 (1 lần, seed=42) cho thấy y_noisy_bits == codeword (True) và
reconstructed_key == random_key (True). Nhưng test 11 (5 lần, không cố
định seed) cho thấy verify() luôn FAILED. Cần kiểm tra: đẳng thức đại số
"y_noisy_bits phải bằng codeword khi self-match" có LUÔN đúng qua nhiều
lần thử hay chỉ đúng tình cờ ở seed=42?

Script này lặp lại CHÍNH XÁC logic của test 09 (tái hiện thủ công enroll+
verify, KHÔNG qua handler.enroll()/verify() như hộp đen) qua N lần thử
với random KHÔNG cố định seed, in ra kết quả từng lần.

Cách chạy:
    python scripts/research/12_repeat_algebraic_identity_check.py
"""

import os
import sys
import hashlib
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation as orig_modulation
from research.decoder.v0_neural_ms_original import NeuralMSOriginal

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)


def main():
    handler = WiFaKeyHandler()
    decoder = NeuralMSOriginal(handler)
    emb = np.load(os.path.join(CACHE_DIR, "Aaron_Guiel_0001.npy"))

    n_trials = 10
    n_identity_ok = 0
    n_decode_ok = 0

    for trial in range(n_trials):
        # ==== Tái hiện thủ công enroll() (KHÔNG cố định seed) ====
        b_full = handler._binarize_full(emb).astype(np.uint8)
        u = np.random.uniform(0.0, 1.0, size=len(b_full))
        mask_r = (u >= handler.kappa).astype(np.uint8)
        b_masked_enroll = (b_full & mask_r).astype(np.uint8)
        b_selected_enroll = b_masked_enroll[: handler.feature_length]

        random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
        codeword = handler.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected_enroll, codeword).astype(np.uint8)

        # ==== Tái hiện thủ công verify() (self-match, CÙNG emb) ====
        b_full_v = handler._binarize_full(emb).astype(np.uint8)
        b_masked_verify = (b_full_v & mask_r).astype(np.uint8)
        b_selected_verify = b_masked_verify[: handler.feature_length]

        selected_match = np.array_equal(b_selected_enroll, b_selected_verify)

        y_noisy_bits = np.logical_xor(b_selected_verify, helper_data)
        identity_ok = np.array_equal(y_noisy_bits.astype(np.uint8), codeword)
        n_identity_ok += int(identity_ok)

        llr = orig_modulation.BPSK(y_noisy_bits).astype(np.float32)
        reconstructed_key = decoder.decode(llr)
        decode_ok = np.array_equal(reconstructed_key, random_key.flatten())
        n_decode_ok += int(decode_ok)

        print(
            f"Trial {trial+1}: b_selected khớp={selected_match}  "
            f"y_noisy==codeword={identity_ok}  decode_đúng={decode_ok}"
        )

    print(f"\n=== TỔNG KẾT ({n_trials} lần) ===")
    print(f"Đẳng thức y_noisy_bits==codeword đúng: {n_identity_ok}/{n_trials}")
    print(f"Decode đúng random_key: {n_decode_ok}/{n_trials}")

    if n_identity_ok < n_trials:
        print("\n❌ Đẳng thức đại số KHÔNG LUÔN đúng - test 09 (seed=42) chỉ là")
        print("   trường hợp may mắn. Cần tìm hiểu vì sao b_selected(enroll) không")
        print("   luôn bằng b_selected(verify) dù cùng embedding + cùng mask_r.")
    elif n_decode_ok < n_identity_ok:
        print("\n❌ y_noisy_bits LUÔN bằng codeword (đẳng thức đúng), NHƯNG decoder")
        print("   không phải lúc nào cũng giải mã đúng ngay cả với input KHÔNG NHIỄU.")
        print("   => Đây là bằng chứng quyết định: NEURAL-MS DECODER không đáng tin")
        print("   cậy 100% ngay cả với codeword hợp lệ không nhiễu - có thể do:")
        print("   - Trọng số Weights_Var_MS chỉ được train cho 1 SỐ ÍT vòng lặp/điều")
        print("     kiện nhất định, không tổng quát hóa tốt cho MỌI codeword hợp lệ.")
        print("   - Cần kiểm tra lại xem trọng số này CÓ THỰC SỰ được train cho đúng")
        print("     N=52/m=42/Z=16 hay là file ví dụ/placeholder chưa hoàn chỉnh.")
    else:
        print("\n✅ Mọi thứ nhất quán 100% - nếu vậy kết quả 'FAILED' trước đó cần")
        print("   xem lại có phải do nguyên nhân khác (vd: helper_data/mask_r bị")
        print("   truyền sai lệch khi đi qua nhiều lớp gọi hàm).")


if __name__ == "__main__":
    main()
