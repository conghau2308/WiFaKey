"""
10_real_handler_call_same_embedding.py

Test 09 (tái hiện thủ công) đã CHỨNG MINH: reconstructed_key == random_key
(so sánh trực tiếp) cho embedding Aaron_Peirsol_0001.npy - nghĩa là thuật
toán lõi hoạt động đúng. Lỗi hash ở test 09 là do CHÍNH TÔI thêm nhầm
.astype(np.uint8) khi tự tính recon_hash để so sánh (không phải bug
trong wifakey_handler.py gốc).

Nhưng test 07/10 (gọi THẲNG handler.enroll()/handler.verify() gốc, không
tái hiện thủ công) lại báo FAILED trên 20/20 embedding. Script này gọi
handler.enroll()/handler.verify() gốc TRÊN CHÍNH embedding Aaron_Peirsol_0001
đã biết là "tốt" - nếu vẫn FAILED, nghĩa là có gì đó khác biệt giữa việc
GỌI HÀM THẬT và TÁI HIỆN THỦ CÔNG mà ta chưa biết.

Cách chạy:
    python scripts/research/10_real_handler_call_same_embedding.py
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


def main():
    handler = WiFaKeyHandler()

    emb_path = os.path.join(CACHE_DIR, "Aaron_Peirsol_0001.npy")
    emb = np.load(emb_path)
    print(f"Dùng embedding: Aaron_Peirsol_0001.npy\n")

    n_ok = 0
    n_trials = 10
    for trial in range(n_trials):
        # Gọi THẲNG handler.enroll()/handler.verify() GỐC, không sửa gì,
        # không tái hiện thủ công - đúng như cách main.py (API) sẽ gọi.
        helper_data, mask_r, key_hash = handler.enroll(emb)
        success = handler.verify(emb, helper_data, mask_r, key_hash)
        n_ok += int(success)
        print(f"  Lần {trial+1}: {'THÀNH CÔNG' if success else 'THẤT BẠI'}")

    print(f"\nTổng: {n_ok}/{n_trials} thành công")

    if n_ok == 0:
        print(
            "\n❌ Gọi hàm THẬT vẫn fail 100% dù test 09 (tái hiện thủ công cùng logic,"
        )
        print("   cùng embedding) đã CHỨNG MINH thuật toán đúng. Có sự khác biệt giữa")
        print("   lời gọi hàm thật và bản tái hiện thủ công. Khả năng cao nhất:")
        print("   phiên bản wifakey_handler.py ĐANG CHẠY THỰC TẾ trên máy bạn KHÁC với")
        print("   phiên bản mã nguồn đã dán cho tôi xem lúc đầu cuộc trò chuyện.")
        print("   -> Hãy dán lại NGUYÊN VĂN nội dung HIỆN TẠI của")
        print("      wifakey_module/wifakey_handler.py (đặc biệt hàm verify()) để tôi")
        print("      đối chiếu từng dòng với bản gốc đã phân tích.")
    elif n_ok == n_trials:
        print("\n✅ Gọi hàm THẬT cũng thành công 100% - vậy 07/10_selfmatch trước đó")
        print(
            "   thất bại có thể do MỘT SỐ EMBEDDING KHÁC (không phải Aaron_Peirsol_0001)"
        )
        print(
            "   có vấn đề riêng (ảnh lỗi, embedding suy biến...), không phải bug hệ thống."
        )
        print("   Cần chạy lại 07/10 và in ra TÊN FILE cụ thể của từng embedding fail")
        print("   để kiểm tra riêng các trường hợp đó.")
    else:
        print(
            f"\n⚠️  Kết quả KHÔNG ỔN ĐỊNH ({n_ok}/{n_trials}) dù dùng CÙNG 1 embedding -"
        )
        print(
            "   đây là dấu hiệu NON-DETERMINISM thực sự (rất có thể do GPU/TF gây sai"
        )
        print(
            "   số dấu phẩy động không ổn định giữa các lần chạy sess.run() khác nhau)."
        )
        print("   Thử chạy lại với CUDA_VISIBLE_DEVICES=-1 (ép CPU) để xem có ổn định")
        print(
            "   100% không - nếu có, đây là vấn đề GPU non-determinism cần xử lý riêng."
        )


if __name__ == "__main__":
    main()
