"""
export_llr_table.py

Xuất bảng Empirical LLR từ reliability_lookup.npz sang định dạng text cho Rust.
"""

import numpy as np
import os


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    lookup_path = os.path.join(
        project_root, "experiments", "out_step3", "reliability_lookup.npz"
    )

    if not os.path.exists(lookup_path):
        print(f"Không tìm thấy {lookup_path}")
        return

    data = np.load(lookup_path)
    margin_bp = data["margin_breakpoints"]  # mảng 1D: các điểm chia margin
    p_bp = data["p_breakpoints"]  # mảng 1D: xác suất lỗi tại các điểm chia

    print(f"Số breakpoints: {len(margin_bp)}")
    print(f"margin range: [{margin_bp[0]:.6f}, {margin_bp[-1]:.6f}]")
    print(f"p range: [{p_bp[0]:.6f}, {p_bp[-1]:.6f}]")

    # Lưu ra file text: mỗi dòng "margin p_error"
    output_path = os.path.join(os.path.dirname(__file__), "empirical_llr_table.txt")
    with open(output_path, "w") as f:
        f.write(f"{len(margin_bp)}\n")  # dòng đầu: số breakpoints
        for m, p in zip(margin_bp, p_bp):
            f.write(f"{m:.10f} {p:.10f}\n")

    print(f"Đã lưu bảng LLR vào {output_path}")


if __name__ == "__main__":
    main()
