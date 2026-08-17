"""
run_comprehensive_benchmark.py

Quét qua tất cả tổ hợp phương pháp (bao gồm Oracle‑LDA và Neural Correction) trên LFW và CPLFW.
"""

import subprocess, sys, os, json, csv

SCRIPT = os.path.join(os.path.dirname(__file__), "benchmark_single_config.py")
LOOKUP = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "experiments",
    "out_step3",
    "reliability_lookup.npz",
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "comprehensive_results_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Oracle‑LDA path (nếu có)
ORACLE_LDA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "research",
    "modulation",
    "dimension_selection",
    "M_matrix_oracle_lda_regeps2.0.npy",
)
if not os.path.exists(ORACLE_LDA_PATH):
    print(
        f"[WARN] Không tìm thấy Oracle‑LDA matrix tại {ORACLE_LDA_PATH}, sẽ bỏ qua cấu hình LDA."
    )
    ORACLE_LDA_PATH = None

# Neural Correction checkpoint (nếu có)
NEURAL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "checkpoints", "neural_llr_v3", "L3_H128_relu", "model"
)
if not os.path.exists(NEURAL_MODEL_PATH + ".index"):
    print(
        f"[WARN] Không tìm thấy Neural Correction checkpoint tại {NEURAL_MODEL_PATH}, sẽ bỏ qua cấu hình neural."
    )
    NEURAL_MODEL_PATH = None

# Định nghĩa tất cả các tổ hợp
# (selection, llr, multi_K, multi_sigma, oracle_lda, neural_model, label)
CONFIGS = [
    ("random", "bpsk", 0, 0, None, None, "random_bpsk"),
    ("random", "empirical", 0, 0, None, None, "random_empirical"),
    ("random", "empirical", 5, 0.2, None, None, "random_empirical_multi"),
    ("margin", "bpsk", 0, 0, None, None, "margin_bpsk"),
    ("margin", "empirical", 0, 0, None, None, "margin_empirical"),
    ("margin", "empirical", 5, 0.2, None, None, "margin_empirical_multi"),
]

if ORACLE_LDA_PATH:
    CONFIGS.extend(
        [
            ("random", "bpsk", 0, 0, ORACLE_LDA_PATH, None, "random_bpsk_lda"),
            (
                "random",
                "empirical",
                0,
                0,
                ORACLE_LDA_PATH,
                None,
                "random_empirical_lda",
            ),
            (
                "margin",
                "empirical",
                0,
                0,
                ORACLE_LDA_PATH,
                None,
                "margin_empirical_lda",
            ),
            (
                "margin",
                "empirical",
                5,
                0.2,
                ORACLE_LDA_PATH,
                None,
                "margin_empirical_multi_lda",
            ),
        ]
    )

if NEURAL_MODEL_PATH:
    CONFIGS.extend(
        [
            (
                "random",
                "empirical",
                0,
                0,
                None,
                NEURAL_MODEL_PATH,
                "random_empirical_neural",
            ),
            (
                "margin",
                "empirical",
                0,
                0,
                None,
                NEURAL_MODEL_PATH,
                "margin_empirical_neural",
            ),
        ]
    )

DATASETS = [
    # ("labeled_faces_in_the_wild", "tune", "LFW"),
    ("cplfw", "select", "CPLFW"),
]


def run_config(
    selection,
    llr,
    multi_K,
    multi_sigma,
    oracle_lda,
    neural_model,
    dataset,
    tier,
    label,
    max_pairs=None,
):
    output_file = os.path.join(RESULTS_DIR, f"{dataset}_{label}.json")
    cmd = [
        sys.executable,
        SCRIPT,
        "--selection",
        selection,
        "--llr",
        llr,
        "--multi-K",
        str(multi_K),
        "--multi-sigma",
        str(multi_sigma),
        "--lookup",
        LOOKUP,
        "--dataset",
        dataset,
        "--tier",
        tier,
        "--output",
        output_file,
        "--label",
        label,
    ]
    if oracle_lda:
        cmd.extend(["--oracle-lda", oracle_lda])
    if neural_model:
        cmd.extend(["--neural-model", neural_model])
    if max_pairs:
        cmd.extend(["--max-pairs", str(max_pairs)])
    print(f"\nChạy: {label} trên {dataset}")
    subprocess.run(cmd, check=True)
    return output_file


def main():
    csv_file = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_file, "w", newline="") as f:
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

        for dataset, tier, name in DATASETS:
            for (
                selection,
                llr,
                multi_K,
                multi_sigma,
                oracle_lda,
                neural_model,
                label,
            ) in CONFIGS:
                output_file = run_config(
                    selection,
                    llr,
                    multi_K,
                    multi_sigma,
                    oracle_lda,
                    neural_model,
                    dataset,
                    tier,
                    label,
                    # max_pairs=200,
                )
                with open(output_file) as f_json:
                    data = json.load(f_json)
                gmr = data["GMR"] * 100
                far = data["FAR"] * 100
                writer.writerow(
                    [
                        name,
                        label,
                        f"{gmr:.2f}",
                        f"{far:.4f}",
                        data["genuine_success"],
                        data["genuine_total"],
                        data["impostor_success"],
                        data["impostor_total"],
                    ]
                )
                print(f"  {label}: GMR={gmr:.2f}%, FAR={far:.4f}%")

    print(f"\n✅ Kết quả tổng hợp đã được lưu vào {csv_file}")


if __name__ == "__main__":
    main()
