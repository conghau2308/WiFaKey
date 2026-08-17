"""
generate_summary.py

Quét thư mục comprehensive_results, đọc tất cả file JSON và tạo summary.csv.
"""

import os
import json
import csv

# Đường dẫn tới thư mục chứa kết quả
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "comprehensive_results_v2")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "summary.csv")


def main():
    rows = []

    # Duyệt qua tất cả file JSON trong thư mục
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Trích xuất thông tin cần thiết
        dataset = data.get("dataset", "unknown")
        label = data.get("label", "unknown")
        gmr = data.get("GMR", 0) * 100
        far = data.get("FAR", 0) * 100
        gen_succ = data.get("genuine_success", 0)
        gen_total = data.get("genuine_total", 0)
        imp_succ = data.get("impostor_success", 0)
        imp_total = data.get("impostor_total", 0)

        rows.append(
            [
                dataset,
                label,
                f"{gmr:.2f}",
                f"{far:.4f}",
                gen_succ,
                gen_total,
                imp_succ,
                imp_total,
            ]
        )

    # Sắp xếp theo dataset và method
    rows.sort(key=lambda r: (r[0], r[1]))

    # Ghi ra CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Dataset",
                "Method",
                "GMR",
                "FAR",
                "Genuine Success",
                "Genuine Total",
                "Impostor Success",
                "Impostor Total",
            ]
        )
        writer.writerows(rows)

    print(
        f"✅ Đã tạo {OUTPUT_CSV} với {len(rows)} dòng dữ liệu từ {len(os.listdir(RESULTS_DIR))} file JSON."
    )


if __name__ == "__main__":
    main()
