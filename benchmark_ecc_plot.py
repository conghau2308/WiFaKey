"""Render BLER / BER / decode-time charts from benchmark_ecc.py results."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _label(run: Dict[str, Any]) -> str:
    name = run["decoder"]
    return f"{name}  (n={run['n']}, k={run['k']}, R={run['rate']:.3f})"


def plot_results(runs: List[Dict[str, Any]], out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax_bler, ax_ber, ax_time = axes

    for run in runs:
        rs = run["results"]
        ps = [r["p"] for r in rs]
        bler = [max(r["bler"], 1e-4) for r in rs]
        ber = [max(r["ber"], 1e-6) for r in rs]
        ax_bler.semilogy(ps, bler, marker="o", label=_label(run))
        ax_ber.semilogy(ps, ber, marker="s", label=_label(run))

    ax_bler.set_xlabel("BSC crossover probability p")
    ax_bler.set_ylabel("BLER (block error rate)")
    ax_bler.set_title("BLER vs p")
    ax_bler.grid(True, which="both", alpha=0.3)
    ax_bler.axhline(1e-2, color="grey", linestyle="--", linewidth=0.7,
                    label="threshold 1e-2")
    ax_bler.legend(loc="best", fontsize=8)

    ax_ber.set_xlabel("BSC crossover probability p")
    ax_ber.set_ylabel("BER (over message bits)")
    ax_ber.set_title("BER vs p")
    ax_ber.grid(True, which="both", alpha=0.3)
    ax_ber.legend(loc="best", fontsize=8)

    # Bar chart of mean decode time at the lowest p (typically p ~= 0.001).
    decoder_names = [run["decoder"] for run in runs]
    mean_times = []
    p95_times = []
    for run in runs:
        first = run["results"][0] if run["results"] else None
        mean_times.append(first["mean_ms"] if first else 0.0)
        p95_times.append(first["p95_ms"] if first else 0.0)
    x = range(len(decoder_names))
    width = 0.35
    ax_time.bar([i - width / 2 for i in x], mean_times, width=width, label="mean")
    ax_time.bar([i + width / 2 for i in x], p95_times, width=width, label="p95")
    ax_time.set_xticks(list(x))
    ax_time.set_xticklabels(decoder_names, rotation=15)
    ax_time.set_ylabel("decode time per codeword (ms)")
    ax_time.set_title("Decode latency (at smallest p)")
    ax_time.grid(True, axis="y", alpha=0.3)
    ax_time.legend()

    fig.suptitle("WiFaKey ECC decoder benchmark on BSC", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
