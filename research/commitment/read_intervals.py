"""
read_intervals.py

Đọc và in ra các ngưỡng (intervals) dùng trong binarization.
Chạy script này từ thư mục gốc của dự án WiFaKey.
"""

import numpy as np
import os


def find_intervals_file():
    """Tìm file binarization_intervals.npy từ thư mục hiện tại trở lên."""
    current_dir = os.getcwd()
    for _ in range(5):
        candidate = os.path.join(
            current_dir, "wifakey_module", "data", "binarization_intervals.npy"
        )
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent
    return None


def main():
    intervals_path = find_intervals_file()
    if intervals_path is None:
        # Thử tìm trực tiếp từ PROJECT_ROOT
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        intervals_path = os.path.join(
            project_root, "wifakey_module", "data", "binarization_intervals.npy"
        )

    if not os.path.exists(intervals_path):
        print("Không tìm thấy file binarization_intervals.npy")
        print("Hãy đảm bảo bạn đã chạy 01_extract_embeddings.py hoặc file này tồn tại.")
        return

    intervals = np.load(intervals_path)
    print(f"File: {intervals_path}")
    print(f"Intervals (ngưỡng thermometer code): {intervals}")
    print(f"Kiểu dữ liệu: {intervals.dtype}")
    print(f"Shape: {intervals.shape}")
    print(f"Giá trị (dạng list): {intervals.tolist()}")

    # Lưu ra file text để dễ dùng trong Rust
    output_path = os.path.join(os.path.dirname(__file__), "intervals_values.txt")
    with open(output_path, "w") as f:
        f.write(str(intervals.tolist()))
    print(f"Đã lưu giá trị vào {output_path}")


if __name__ == "__main__":
    main()
