"""
run_full_margin_benchmark.py

Tự động chạy benchmark cho cả Baseline và Margin Selection trên LFW và CPLFW.
Gọi run_final_benchmark_extended.py trong các process riêng biệt.
"""

import subprocess
import sys
import os
import json

SCRIPT = os.path.join(os.path.dirname(__file__), "run_final_benchmark_extended.py")
LOOKUP = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "experiments",
    "out_step3",
    "reliability_lookup.npz",
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = [
    {
        "name": "LFW",
        "folder": "labeled_faces_in_the_wild",
        "tier": "tune",
    },
    {
        "name": "CPLFW",
        "folder": "cplfw",
        "tier": "select",
    },
]

MODES = [
    {"mode": "baseline", "flag": None, "suffix": "baseline"},
    {"mode": "margin", "flag": "--margin-selection", "suffix": "margin"},
]


def run_benchmark(dataset, mode_info):
    output_file = os.path.join(
        RESULTS_DIR, f"{dataset['name'].lower()}_{mode_info['suffix']}.json"
    )
    cmd = [
        sys.executable,
        SCRIPT,
        "--lookup",
        LOOKUP,
        "--dataset-folder",
        dataset["folder"],
        "--tier",
        dataset["tier"],
        "--output",
        output_file,
    ]
    if mode_info["flag"]:
        cmd.append(mode_info["flag"])

    print(f"\n{'='*60}")
    print(f"Chạy {mode_info['mode']} trên {dataset['name']}...")
    print(f"Lệnh: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_file


def main():
    results = {}

    for dataset in DATASETS:
        for mode_info in MODES:
            output_file = run_benchmark(dataset, mode_info)
            with open(output_file) as f:
                results[f"{dataset['name']}_{mode_info['mode']}"] = json.load(f)

    # In bảng tổng hợp
    print("\n" + "=" * 80)
    print("TỔNG HỢP KẾT QUẢ")
    print("=" * 80)
    print(f"{'Dataset':<10} {'Phương pháp':<20} {'GMR':<15} {'FAR':<15}")
    print("-" * 60)

    for key, data in results.items():
        dataset, method = key.rsplit("_", 1)
        gmr = (
            data["genuine_success"] / data["genuine_total"] * 100
            if data["genuine_total"]
            else 0
        )
        far = (
            data["impostor_success"] / data["impostor_total"] * 100
            if data["impostor_total"]
            else 0
        )
        print(
            f"{dataset:<10} {method:<20} {gmr:>6.2f}% ({data['genuine_success']}/{data['genuine_total']})   {far:>8.4f}% ({data['impostor_success']}/{data['impostor_total']})"
        )

    print("\n✅ Hoàn tất. Kết quả chi tiết được lưu trong thư mục:", RESULTS_DIR)


if __name__ == "__main__":
    main()
