"""
08_selfmatch_step_by_step_debug.py

Test 1 (decoder thuần) PASS 100%. Self-match (embedding thật) FAIL 100%.
Về lý thuyết, self-match phải quy giản đúng về Test 1 (verify bằng chính
embedding vừa enroll -> y_noisy_bits phải == codeword hệt như Test 1).

Script này soi TỪNG BƯỚC của MỘT trường hợp self-match cụ thể, để xác
định chính xác b_full/mask/helper_data lệch nhau ở đâu.

Cách chạy:
    python scripts/research/08_selfmatch_step_by_step_debug.py
"""

import os
import sys
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


def load_one_embedding():
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    return np.load(os.path.join(CACHE_DIR, files[0])), files[0]


def main():
    handler = WiFaKeyHandler()
    emb, fname = load_one_embedding()
    print(f"Dùng embedding: {fname}, shape={emb.shape}, dtype={emb.dtype}\n")

    print(
        f"M_matrix.shape = {handler.M_matrix.shape}  "
        f"(kỳ vọng: (512 hoặc len(embedding), n_features_sau_chieu))"
    )
    print(
        f"intervals.shape = {handler.intervals.shape}  (n_thr = {handler.intervals.shape[0]})"
    )
    print(f"feature_length (self.feature_length) = {handler.feature_length}")
    print(
        f"full_binary_length (dùng validate ở API) = "
        f"{handler.M_matrix.shape[0] * handler.intervals.shape[0]}\n"
    )

    # ---- Gọi _binarize_full 2 LẦN với CÙNG 1 embedding ----
    b_full_1 = handler._binarize_full(emb).astype(np.uint8)
    b_full_2 = handler._binarize_full(emb).astype(np.uint8)

    print(f"len(b_full) lần 1: {len(b_full_1)}")
    print(f"len(b_full) lần 2: {len(b_full_2)}")
    print(
        f"b_full lần 1 == lần 2 (identical không?): {np.array_equal(b_full_1, b_full_2)}"
    )
    if not np.array_equal(b_full_1, b_full_2):
        diff = np.sum(b_full_1 != b_full_2)
        print(
            f"❌ KHÁC NHAU ở {diff}/{len(b_full_1)} vị trí dù CÙNG 1 embedding đầu vào!"
        )
        print("   => _binarize_full() không deterministic - đây LÀ nguyên nhân gốc.")
        print("   Kiểm tra lại hàm lssc_binary/utils.py có dùng random ở đâu không,")
        print("   hoặc có phụ thuộc biến global/state nào bị thay đổi giữa 2 lần gọi.")
        return
    else:
        print(
            "✅ b_full giống hệt nhau qua 2 lần gọi -> _binarize_full deterministic, OK.\n"
        )

    projected = np.dot(emb, handler.M_matrix)
    print(f"projected.shape = {projected.shape}  (= embedding @ M_matrix)")
    print(f"len(b_full) thực tế = {len(b_full_1)}")
    print(
        f"Kỳ vọng theo lssc_binary: len(projected) * n_thr = "
        f"{projected.shape[0] * handler.intervals.shape[0]}"
    )

    if len(b_full_1) != projected.shape[0] * handler.intervals.shape[0]:
        print("❌ ĐỘ DÀI b_full KHÔNG khớp với len(projected)*n_thr - có khả năng hàm")
        print("   lssc_binary gốc bị hard-code kích thước (vd: giả định 512 chiều)")
        print("   khác với chiều thực tế của M_matrix, gây cắt/đệm sai âm thầm.")

    if len(b_full_1) != handler.full_binary_length:
        print(
            f"❌ len(b_full)={len(b_full_1)} KHÁC full_binary_length="
            f"{handler.full_binary_length} (dùng để tính mask_r) - đây có thể là"
        )
        print("   nguồn gốc gây lệch vị trí bit giữa b_full và mask_r!")

    # ---- Enroll và verify thủ công, in từng bước ----
    print("\n=== Enroll + Verify thủ công (in từng bước) ===")
    helper_data, mask_r, key_hash = handler.enroll(emb)
    print(f"len(mask_r) = {len(mask_r)}")
    print(f"len(helper_data) = {len(helper_data)}")

    b_full_verify = handler._binarize_full(emb).astype(np.uint8)
    print(
        f"b_full (verify) == b_full (đã tính ở trên, lần 1): "
        f"{np.array_equal(b_full_verify, b_full_1)}"
    )

    if len(mask_r) != len(b_full_verify):
        print(f"❌ len(mask_r)={len(mask_r)} KHÁC len(b_full)={len(b_full_verify)}")
        print("   -> phép AND 'b_full & mask_r' trong verify() sẽ lỗi hoặc numpy tự")
        print("   broadcast sai. ĐÂY RẤT CÓ THỂ LÀ NGUYÊN NHÂN CHÍNH.")
    else:
        b_masked_verify = (b_full_verify & mask_r)[: handler.feature_length]
        b_masked_enroll_reconstructed = (
            None  # không có sẵn, nhưng b_full giống nhau nên suy ra giống
        )
        y_noisy_bits = np.logical_xor(b_masked_verify, helper_data)
        codeword_all_zero_check = np.sum(y_noisy_bits)
        print(
            f"y_noisy_bits có {codeword_all_zero_check} bit = 1 trên tổng "
            f"{len(y_noisy_bits)} (nếu self-match hoàn hảo, y_noisy_bits phải"
        )
        print(f"   CHÍNH XÁC bằng codeword ban đầu, không nhất thiết toàn 0 - nhưng")
        print(f"   phải decode được).")

    success = handler.verify(emb, helper_data, mask_r, key_hash)
    print(f"\nhandler.verify() kết quả: {'THÀNH CÔNG' if success else 'THẤT BẠI'}")


if __name__ == "__main__":
    main()
