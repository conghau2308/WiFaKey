"""
convert_m_matrix.py

Chuyển M_matrix.npy (512x512) thành file text để Rust load.
"""

import numpy as np
import os


def main():
    # Đường dẫn tới M_matrix.npy
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    matrix_path = os.path.join(project_root, "wifakey_module", "data", "M_matrix.npy")

    if not os.path.exists(matrix_path):
        print(f"Không tìm thấy M_matrix.npy tại {matrix_path}")
        return

    M = np.load(matrix_path)
    print(f"Kích thước M_matrix: {M.shape}")
    print(f"Kiểu dữ liệu: {M.dtype}")

    # Lưu ra file text
    output_path = os.path.join(os.path.dirname(__file__), "M_matrix.txt")
    np.savetxt(output_path, M, fmt="%.10f")
    print(f"Đã lưu M_matrix.txt tại {output_path}")
    print(f"Số dòng: {M.shape[0]}, số cột: {M.shape[1]}")


if __name__ == "__main__":
    main()
