"""
04_check_wodd2even_woutput.py

W_even2odd đã xác nhận ĐÚNG (197/197). Kiểm tra tiếp 2 ma trận còn lại
theo đúng thứ tự xây dựng gốc trong _build_decoder():

  W_odd2even (sum_edge x sum_edge): cột k (thuộc edge của variable-node j)
      phải có đúng (bậc(j) - 1) giá trị 1, đánh dấu CÁC CẠNH KHÁC cùng
      thuộc variable-node j (trừ chính nó).

  W_output (sum_edge x N): mỗi HÀNG (1 cạnh) phải có ĐÚNG 1 giá trị 1
      (cạnh đó thuộc về đúng 1 variable-node). Tổng theo CỘT j phải bằng
      bậc(j).

  W_skipconn2even (N x sum_edge): mỗi CỘT (1 cạnh) phải có đúng 1 giá trị 1
      ở đúng hàng = variable-node của cạnh đó. Tổng theo hàng j = bậc(j).

Thuần NumPy, không cần TF/GPU, chạy trong vài giây.

Cách chạy:
    python scripts/research/04_check_wodd2even_woutput.py
"""

import os
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PCM_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "BaseGraph", "BaseGraph2_Set0.txt"
)


def main():
    code_PCM_raw = np.loadtxt(PCM_PATH, int, delimiter=None)
    code_PCM = (code_PCM_raw != -1).astype(
        int
    )  # -1 = không cạnh, mọi giá trị khác (kể cả 0) = cạnh thật

    sum_edge_c = np.sum(code_PCM, axis=1)
    sum_edge_v = np.sum(code_PCM, axis=0)
    sum_edge = int(np.sum(sum_edge_v))
    N = code_PCM.shape[1]

    print(f"sum_edge={sum_edge}, N={N}\n")

    # ==================== Dựng W_odd2even (đúng thứ tự vòng lặp gốc) ====================
    W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    edge_variable_id = np.full(sum_edge, -1, dtype=int)  # edge k -> variable-node j

    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM[i, j] == 1:
                num_of_conn = int(np.sum(code_PCM[:, j]))
                idx = np.argwhere(code_PCM[:, j] == 1)
                for l in range(num_of_conn):
                    vec_tmp = np.zeros(sum_edge, dtype=np.float32)
                    for r in range(code_PCM.shape[0]):
                        if code_PCM[r, j] == 1 and idx[l][0] != r:
                            idx_row = np.cumsum(code_PCM[r, 0 : j + 1])[-1] - 1
                            cnt = 0
                            if r > 0:
                                cnt = np.cumsum(sum_edge_c[0:r])[-1]
                            vec_tmp[idx_row + cnt] = 1
                    W_odd2even[:, k] = vec_tmp.transpose()
                    edge_variable_id[k] = j
                    k += 1
                break

    print(
        "=== Kiểm tra W_odd2even: mỗi CỘT phải có đúng (bậc(variable) - 1) giá trị 1 ==="
    )
    col_sums = W_odd2even.sum(axis=0)
    n_mismatch_odd2even = 0
    for edge_id in range(sum_edge):
        j = edge_variable_id[edge_id]
        expected = sum_edge_v[j] - 1
        if col_sums[edge_id] != expected:
            n_mismatch_odd2even += 1
    print(f"Số cột KHỚP: {sum_edge - n_mismatch_odd2even}/{sum_edge}")
    print(f"Số cột SAI:  {n_mismatch_odd2even}/{sum_edge}\n")

    # ==================== Dựng W_output (đúng thứ tự vòng lặp gốc) ====================
    W_output = np.zeros((sum_edge, N), dtype=np.float32)
    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM[i, j] == 1:
                idx_row = np.cumsum(code_PCM[i, 0 : j + 1])[-1] - 1
                cnt = 0
                if i > 0:
                    cnt = np.cumsum(sum_edge_c[0:i])[-1]
                W_output[cnt + idx_row, k] = 1.0
        k += 1  # LƯU Ý: dòng này nằm NGOÀI vòng lặp "for i" - giống hệt code gốc

    print("=== Kiểm tra W_output: mỗi HÀNG (1 cạnh) phải có ĐÚNG 1 giá trị 1 ===")
    row_sums = W_output.sum(axis=1)
    n_rows_with_0 = int(np.sum(row_sums == 0))
    n_rows_with_1 = int(np.sum(row_sums == 1))
    n_rows_with_more = int(np.sum(row_sums > 1))
    print(f"Số hàng có ĐÚNG 1 giá trị 1: {n_rows_with_1}/{sum_edge}")
    print(
        f"Số hàng có 0 giá trị 1 (SAI - cạnh không map vào biến nào cả): {n_rows_with_0}/{sum_edge}"
    )
    print(
        f"Số hàng có >1 giá trị 1 (SAI - cạnh map vào nhiều biến): {n_rows_with_more}/{sum_edge}"
    )

    col_sums_out = W_output.sum(axis=0)
    n_col_mismatch = int(np.sum(col_sums_out != sum_edge_v))
    print(
        f"Số cột (variable-node) có tổng KHÔNG khớp bậc lý thuyết: {n_col_mismatch}/{N}\n"
    )

    # ==================== Kết luận ====================
    if (
        n_mismatch_odd2even == 0
        and n_rows_with_0 == 0
        and n_rows_with_more == 0
        and n_col_mismatch == 0
    ):
        print("✅ Cả W_odd2even và W_output đều khớp lý thuyết.")
        print(
            "   => Nghi vấn chuyển sang: Lift_M1/Lift_M2 (dù đây chỉ hoán vị, khó gây"
        )
        print("      ra biên độ hàng trăm/nghìn), HOẶC file trọng số Weights_Var_MS/")
        print(
            "      Biases_Var_MS không thực sự tương thích với N=52/m=42/Z=16 hiện tại"
        )
        print("      (ví dụ được train cho Z khác, dù shape trùng hợp khớp 197).")
    else:
        print("❌ Phát hiện SAI LỆCH cụ thể ở trên - đây là vị trí cần sửa trong")
        print("   wifakey_module/wifakey_handler.py, hàm _build_decoder().")


if __name__ == "__main__":
    main()
