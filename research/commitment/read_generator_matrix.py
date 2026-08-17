"""
read_generator_matrix.py

Đọc và kiểm tra kích thước của M_matrix.npy.
Nếu kích thước là (160, 832) → đây chính là ma trận sinh G của LDPC.
"""

import numpy as np
import os


def main():
    # Tìm đường dẫn tới M_matrix.npy
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    matrix_path = os.path.join(project_root, "wifakey_module", "data", "M_matrix.npy")

    if not os.path.exists(matrix_path):
        print(f"Không tìm thấy M_matrix.npy tại {matrix_path}")
        return

    M = np.load(matrix_path)
    print(f"Kích thước M_matrix: {M.shape}")
    print(f"Kiểu dữ liệu: {M.dtype}")

    # Nếu kích thước là (160, 832), lưu ra file text để Rust dùng
    if M.shape == (160, 832):
        print("Đây chính là ma trận sinh G của LDPC (160x832).")
        output_path = os.path.join(os.path.dirname(__file__), "generator_matrix.txt")
        np.savetxt(output_path, M, fmt="%d")
        print(f"Đã lưu ma trận vào {output_path} (dùng cho Rust).")
    else:
        print(f"Kích thước lạ: {M.shape}. Không phải ma trận sinh G (160, 832).")


if __name__ == "__main__":
    main()
