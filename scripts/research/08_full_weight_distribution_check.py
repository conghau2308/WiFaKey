"""
05_full_weight_distribution_check.py

Test A trước đó chỉ in MEAN của Biases - có thể ẩn giấu các giá trị ngoại
lệ (outlier) cực lớn ở một vài vị trí cụ thể, đủ để giải thích biên độ
bùng nổ (~500-1500) quan sát được dù cấu trúc 3 ma trận W_even2odd/
W_odd2even/W_output đã được xác nhận ĐÚNG 100%.

Script này in đầy đủ min/max/std/số lượng outlier của CẢ Weights và Biases
qua TẤT CẢ 25 vòng lặp - thuần NumPy, không cần TF/GPU.

Cách chạy:
    python scripts/research/05_full_weight_distribution_check.py
"""

import os
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "wifakey_module", "data", "Weights_Var_MS")
BIASES_PATH = os.path.join(_PROJECT_ROOT, "wifakey_module", "data", "Biases_Var_MS")
ITERS_MAX = 25
OUTLIER_THRESHOLD = 5.0  # trọng số/bias vượt ngưỡng này bị coi là bất thường (dữ liệu thường ở cỡ ~0.1-1)


def main():
    print(
        f"{'Iter':>5} | {'W_min':>10} | {'W_max':>10} | {'W_std':>8} | "
        f"{'B_min':>10} | {'B_max':>10} | {'B_std':>8} | {'#outlier':>9}"
    )
    print("-" * 90)

    any_outlier = False
    for i in range(ITERS_MAX):
        w = np.loadtxt(
            os.path.join(WEIGHTS_PATH, f"Weights_Var{i}.txt"),
            delimiter=",",
            dtype=np.float32,
        )
        b = np.loadtxt(
            os.path.join(BIASES_PATH, f"Biases_Var{i}.txt"),
            delimiter=",",
            dtype=np.float32,
        )

        n_outlier = int(
            np.sum(np.abs(w) > OUTLIER_THRESHOLD)
            + np.sum(np.abs(b) > OUTLIER_THRESHOLD)
        )
        if n_outlier > 0:
            any_outlier = True

        print(
            f"{i:>5} | {w.min():>10.4f} | {w.max():>10.4f} | {w.std():>8.4f} | "
            f"{b.min():>10.4f} | {b.max():>10.4f} | {b.std():>8.4f} | {n_outlier:>9}"
        )

    print()
    if any_outlier:
        print(
            f"❌ Có giá trị vượt ngưỡng |x| > {OUTLIER_THRESHOLD} ở ít nhất 1 vòng lặp."
        )
        print(
            "   Đây rất có thể là nguồn gốc trực tiếp của biên độ bùng nổ quan sát được"
        )
        print(
            "   ở Test B/02_single_iteration_test - không cần nghi ngờ thêm về graph/matrix nữa."
        )
        print(
            "   -> Cần xác nhận: các file Weights_Var*.txt/Biases_Var*.txt này có đúng là"
        )
        print(
            "      bộ trọng số được TRAIN RIÊNG cho N=52/m=42/Z=16 hay là file từ 1 cấu hình"
        )
        print(
            "      khác (Z khác, hoặc thậm chí ví dụ/placeholder từ repo gốc chưa train lại)?"
        )
    else:
        print(f"✅ Không có outlier nào vượt ngưỡng {OUTLIER_THRESHOLD}.")
        print("   => Vấn đề không nằm ở giá trị trọng số cực đoan đơn lẻ.")
        print(
            "   Bước tiếp theo cần làm: chèn print() TRỰC TIẾP vào từng bước trung gian"
        )
        print("   bên trong vòng lặp iteration 1 của _build_decoder() (x2, x2_1, x3,")
        print(
            "   x_output_0, x_output_1, LLRa1) để xem giá trị bắt đầu phình to ở ĐÚNG"
        )
        print(
            "   phép tính nào - lúc này cần debug interactive thay vì kiểm tra thống kê tổng."
        )


if __name__ == "__main__":
    main()
