"""
run_margin_vs_baseline.py

Gọi lần lượt diagnose_margin_selection_v2.py trong 2 process riêng biệt
để tránh OOM, sau đó in kết quả so sánh.
"""

import subprocess, json, sys, os

SCRIPT = os.path.join(os.path.dirname(__file__), "diagnose_margin_selection_v2.py")
LOOKUP = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "experiments",
    "out_step3",
    "reliability_lookup.npz",
)
OUT_BASELINE = "results_margin_baseline.json"
OUT_MARGIN = "results_margin_margin.json"
MAX_PAIRS = 200


def run_mode(mode, output):
    cmd = [
        sys.executable,
        SCRIPT,
        "--mode",
        mode,
        "--lookup",
        LOOKUP,
        "--output",
        output,
        "--max-pairs",
        str(MAX_PAIRS),
    ]
    print(f"Chạy: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_mode("baseline", OUT_BASELINE)
    run_mode("margin", OUT_MARGIN)

    # So sánh
    with open(OUT_BASELINE) as f:
        base = json.load(f)
    with open(OUT_MARGIN) as f:
        margin = json.load(f)

    n_base = sum(r["success"] for r in base)
    n_margin = sum(r["success"] for r in margin)
    n_total = len(base)
    print(f"\nBaseline: {n_base}/{n_total} ({100*n_base/n_total:.2f}%)")
    print(f"Margin  : {n_margin}/{n_total} ({100*n_margin/n_total:.2f}%)")
