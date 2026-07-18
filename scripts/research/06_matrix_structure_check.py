"""
03_matrix_structure_check.py

Test 02 cho thấy output SAI 100% VÀ có biên độ khổng lồ (~500-1500) NGAY
TỪ VÒNG LẶP ĐẦU TIÊN. Nghi vấn cụ thể: hằng số "10000" (dùng để đánh dấu
"không phải cạnh thật" trong Tanner graph, để reduce_min bỏ qua) đang bị
RÒ RỈ vào phép tính vì ma trận W_even2odd đánh dấu SAI cạnh nào là thật.

Script này KHÔNG cần TensorFlow/GPU - chỉ dùng NumPy để dựng lại đúng
ma trận W_even2odd (và W_odd2even, W_output để đối chiếu) từ base graph,
rồi kiểm tra: số "cạnh thật" (giá trị 1) trên mỗi hàng của W_even2odd
có khớp với "bậc của check-node đó trừ 1" (loại trừ chính cạnh đang xét)
hay không - đây là bất biến đại số BẮT BUỘC của bất kỳ Tanner graph nào.

Cách chạy (không cần activate GPU, chạy rất nhanh):
    python scripts/research/03_matrix_structure_check.py
"""

import os
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PCM_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "BaseGraph", "BaseGraph2_Set0.txt"
)


def main():
    code_PCM_raw = np.loadtxt(PCM_PATH, int, delimiter=None)
    # QUAN TRỌNG: -1 = không có cạnh; MỌI giá trị khác (kể cả 0, vốn là shift
    # hợp lệ) đều là cạnh thật. Dùng mask thay vì gán tuần tự để tránh nhầm
    # lẫn giữa "vốn dĩ = 0 (cạnh thật)" và "vừa chuyển từ -1 (không phải cạnh)".
    code_PCM = (code_PCM_raw != -1).astype(int)

    sum_edge_c = np.sum(code_PCM, axis=1)  # bậc mỗi check-node (số biến kết nối)
    sum_edge_v = np.sum(code_PCM, axis=0)  # bậc mỗi variable-node
    sum_edge = int(np.sum(sum_edge_v))

    print(
        f"Base graph shape: {code_PCM.shape} (check_rows={code_PCM.shape[0]}, var_cols={code_PCM.shape[1]})"
    )
    print(
        f"Bậc check-node (sum_edge_c): min={sum_edge_c.min()}, max={sum_edge_c.max()}, "
        f"trung bình={sum_edge_c.mean():.2f}"
    )
    print(
        f"Bậc variable-node (sum_edge_v): min={sum_edge_v.min()}, max={sum_edge_v.max()}"
    )
    print(f"Tổng số cạnh (sum_edge): {sum_edge}\n")

    if sum_edge_c.min() <= 1:
        print(
            f"⚠️  Có check-node bậc <= 1 (min={sum_edge_c.min()}) -> sau khi loại trừ"
        )
        print(
            f"   chính cạnh đang xét, KHÔNG còn hàng xóm nào -> x3 buộc phải nhận giá trị"
        )
        print(
            f"   10000 giả -> đây CÓ THỂ là nguồn gốc rò rỉ hằng số 10000 vào phép tính.\n"
        )

    # ---- Dựng lại W_odd2even (thứ tự cạnh: variable-major, giống bản gốc) ----
    W_odd2even_edge_check_id = np.full(
        sum_edge, -1, dtype=int
    )  # edge -> check-node index
    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM[i, j] == 1:
                W_odd2even_edge_check_id[k] = i
                k += 1

    # ---- Dựng lại W_even2odd giống HỆT vòng lặp gốc trong _build_decoder() ----
    W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM[i, j] == 1:
                idx_row = np.cumsum(code_PCM[i, 0 : j + 1])[-1] - 1
                c1, c2 = 0, np.cumsum(sum_edge_c[0 : i + 1])[-1]
                if i > 0:
                    c1 = np.cumsum(sum_edge_c[0:i])[-1]
                W_even2odd[k, c1:c2] = 1.0
                W_even2odd[k, c1 + idx_row] = 0.0
                k += 1

    row_sums = W_even2odd.sum(axis=1)

    print(
        "=== Kiểm tra: mỗi hàng của W_even2odd phải có đúng (bậc check-node - 1) giá trị 1 ==="
    )
    n_mismatch = 0
    for edge_id in range(sum_edge):
        check_id = W_odd2even_edge_check_id[edge_id]
        expected = sum_edge_c[check_id] - 1  # trừ chính cạnh đang xét
        actual = row_sums[edge_id]
        if actual != expected:
            n_mismatch += 1

    print(f"Số hàng KHỚP đúng bậc lý thuyết: {sum_edge - n_mismatch}/{sum_edge}")
    print(f"Số hàng SAI: {n_mismatch}/{sum_edge}")

    if n_mismatch > 0:
        print("\n❌ XÁC NHẬN: W_even2odd bị xây dựng SAI - đây chính là nguồn gốc")
        print("   khiến hằng số giả '10000' rò rỉ vào phép tính ngay từ vòng lặp 1,")
        print("   gây ra biên độ bùng nổ (~500-1500) và sai dấu 100% quan sát được.")
        print("   -> File cần sửa: wifakey_module/wifakey_handler.py")
        print("      Hàm: _build_decoder()")
        print("      Đoạn xây dựng W_even2odd (vòng lặp có dòng")
        print(
            "      'W_even2odd[k, c1:c2] = 1.0' và 'W_even2odd[k, c1 + idx_row] = 0.0')"
        )
        print("      Cần rà lại công thức tính c1/c2/idx_row cho khớp với sum_edge_c")
        print("      thực tế của BaseGraph2_Set0.txt hiện tại (có thể do offset/cumsum")
        print("      lệch 1 vị trí - lỗi kinh điển 'off-by-one' trong loại code này).")
    else:
        print("\n✅ W_even2odd khớp đúng lý thuyết - lỗi không nằm ở đây,")
        print("   cần kiểm tra tiếp W_odd2even hoặc W_output bằng logic tương tự.")


if __name__ == "__main__":
    main()
