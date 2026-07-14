"""Benchmark the WiFaKey Neural-MS decoder against BCH on BSC.

Neural-MS here always loads ``Weights_Var_MS_BSC`` / ``Biases_Var_MS_BSC`` under
``wifakey_module/data`` (no fallback to legacy ``*_MS`` weights).

By default both decoders use **similar code rate**, with BCH block length in
the same order of magnitude as Neural-MS so BLER curves compare on similar
redundancy and channel uses per block. Defaults:

* Neural-MS: fixed at ``n=832, k=160``    -> ``R = 0.1923``
* BCH galois: ``BCH(1023, 193)`` -> ``R=0.1887, t=118`` (closest valid
  ``galois.BCH(1023, k)`` rate; ``n=1023`` is the same order of magnitude as
  Neural-MS ``n=832``).

Override BCH shape with ``--bch-n`` and ``--bch-k``.

Sweeps the BSC crossover probability ``p`` from low to high, measures
block-error-rate (BLER), bit-error-rate over the message bits (BER), and the
mean / p50 / p95 decode latency for each configured decoder.

Usage (from the WiFaKey_252 directory):

    python benchmark_ecc.py --trials 1000 --out results.json --plot results.png

To shrink the BCH workload (its decoder is the slowest):

    python benchmark_ecc.py --trials 1000 --bch-trials 200

To run everything on CPU only (apples-to-apples timing):

    python benchmark_ecc.py --trials 500 --cpu-only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _summarize_times(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p95_ms": float("nan")}
    s = sorted(values_ms)
    return {
        "mean_ms": round(sum(s) / len(s), 4),
        "p50_ms": round(_percentile(s, 0.50), 4),
        "p95_ms": round(_percentile(s, 0.95), 4),
    }


# Default coarse sweep covering the full BSC waterfall region.
DEFAULT_P_GRID = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.13,
    0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
]

# WiFaKey Neural-MS LDPC (fixed architecture).
NEURAL_MS_N = 832
NEURAL_MS_K = 160
NEURAL_MS_RATE = NEURAL_MS_K / NEURAL_MS_N


def refine_grid(coarse_results: List[Dict[str, Any]], n_extra: int = 5) -> List[float]:
    """Find waterfall regions and add evenly spaced p values inside them."""
    sorted_res = sorted(coarse_results, key=lambda r: r["p"])
    extra = []
    for a, b in zip(sorted_res, sorted_res[1:]):
        # Waterfall heuristic: a working point next to a failing one.
        if a["bler"] < 0.20 and b["bler"] > 0.80:
            for j in range(1, n_extra + 1):
                p_new = a["p"] + (b["p"] - a["p"]) * j / (n_extra + 1)
                extra.append(p_new)
    return extra


def _wifakey_data_dir() -> str:
    """Absolute path to ``wifakey_module/data`` (same layout as package defaults)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "wifakey_module", "data")


def make_decoder(name: str, force_cpu: bool, **kw: Any) -> Any:
    if name == "neural_ms":
        from wifakey_module.decoders.neural_ms_decoder import NeuralMSDecoder

        data_dir = _wifakey_data_dir()
        return NeuralMSDecoder(
            data_path=data_dir,
            weights_path=os.path.join(data_dir, "Weights_Var_MS_BSC"),
            biases_path=os.path.join(data_dir, "Biases_Var_MS_BSC"),
            force_cpu=force_cpu,
        )
    if name == "bch":
        from wifakey_module.decoders.bch_decoder import BCHDecoder

        return BCHDecoder(
            n=int(kw.get("bch_n", 1023)),
            k=int(kw.get("bch_k", 193)),
        )
    raise ValueError(f"Unknown decoder name: {name}")


def evaluate_at_p(
    decoder, p: float, n_trials: int, rng: np.random.Generator
) -> Dict[str, Any]:
    block_err = 0
    bit_err = 0
    bit_total = 0
    times_ms: List[float] = []

    for _ in range(n_trials):
        msg = rng.integers(0, 2, size=decoder.k, dtype=np.int64)
        cw = decoder.encode(msg)
        flips = (rng.random(decoder.n) < p).astype(np.uint8)
        rx = (cw.astype(np.uint8) ^ flips).astype(np.uint8)

        t0 = time.perf_counter()
        try:
            msg_hat = decoder.decode(rx, p=p)
        except TypeError:  # decoder doesn't accept p kwarg
            msg_hat = decoder.decode(rx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

        msg_hat = np.asarray(msg_hat, dtype=np.uint8).reshape(decoder.k)
        msg_u8 = msg.astype(np.uint8)
        diff = int(np.count_nonzero(msg_hat != msg_u8))
        bit_err += diff
        bit_total += decoder.k
        if diff > 0:
            block_err += 1

    bler = block_err / n_trials
    ber = bit_err / max(bit_total, 1)
    summary = _summarize_times(times_ms)
    return {
        "p": float(p),
        "n_trials": n_trials,
        "block_err": block_err,
        "bler": bler,
        "bit_err": bit_err,
        "bit_total": bit_total,
        "ber": ber,
        **summary,
    }


def run_for_decoder(
    decoder_name: str,
    n_trials: int,
    p_grid: List[float],
    seed: int,
    force_cpu: bool,
    early_stop_streak: int = 3,
    decoder_build_kw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print(f"\n{'=' * 70}")
    print(f"Building decoder: {decoder_name} ...")
    print("=" * 70)
    t_init = time.perf_counter()
    decoder = make_decoder(
        decoder_name, force_cpu=force_cpu, **(decoder_build_kw or {})
    )
    init_ms = (time.perf_counter() - t_init) * 1000.0
    print(
        f"  -> n={decoder.n}, k={decoder.k}, rate={decoder.rate:.4f}, "
        f"init={init_ms:.1f} ms"
    )

    # Coarse pass with early stop.
    rng = np.random.default_rng(seed)
    coarse_results: List[Dict[str, Any]] = []
    consecutive_one = 0
    for p in sorted(p_grid):
        res = evaluate_at_p(decoder, p, n_trials, rng)
        coarse_results.append(res)
        print(
            f"  p={p:.4f}  BLER={res['bler']:.4f}  BER={res['ber']:.4e}  "
            f"mean={res['mean_ms']:.2f}ms  p95={res['p95_ms']:.2f}ms"
        )
        if res["bler"] >= 0.999999:
            consecutive_one += 1
        else:
            consecutive_one = 0
        if consecutive_one >= early_stop_streak:
            print(f"  [early-stop] BLER stuck at 1.0 for {consecutive_one} steps, "
                  f"skipping remaining higher p values.")
            break

    # Refinement pass around any waterfall.
    extra_ps = refine_grid(coarse_results, n_extra=5)
    refine_results: List[Dict[str, Any]] = []
    if extra_ps:
        print(f"  [refine] Adding {len(extra_ps)} fine grid points around waterfall.")
        for p in extra_ps:
            res = evaluate_at_p(decoder, p, n_trials, rng)
            refine_results.append(res)
            print(
                f"   *p={p:.4f}  BLER={res['bler']:.4f}  BER={res['ber']:.4e}  "
                f"mean={res['mean_ms']:.2f}ms"
            )

    all_results = sorted(coarse_results + refine_results, key=lambda r: r["p"])

    # Find p* threshold.
    threshold_bler = 1e-2
    p_star = None
    for r in all_results:
        if r["bler"] <= threshold_bler:
            p_star = r["p"]
        else:
            # First failure -> stop tracking (curve assumed monotone in p).
            break

    return {
        "decoder": getattr(decoder, "name", decoder_name),
        "decoder_key": decoder_name,
        "n": int(decoder.n),
        "k": int(decoder.k),
        "rate": round(decoder.rate, 6),
        "rate_delta_vs_neural_ms": round(decoder.rate - NEURAL_MS_RATE, 6),
        "neural_ms_reference_rate": round(NEURAL_MS_RATE, 6),
        "init_ms": round(init_ms, 3),
        "extra_info": _decoder_extra_info(decoder),
        "p_star_bler_le_1e-2": p_star,
        "results": all_results,
    }


def _decoder_extra_info(decoder) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    for attr in ("d_v", "d_c", "max_iter", "iters_max", "t"):
        if hasattr(decoder, attr):
            info[attr] = getattr(decoder, attr)
    return info


def print_summary(all_runs: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Reference (Neural-MS architecture): n={NEURAL_MS_N}, k={NEURAL_MS_K}, "
        f"R={NEURAL_MS_RATE:.6f}"
    )
    header = (
        f"{'decoder':<14} {'n':>5} {'k':>5} {'rate':>7} {'dR':>7} "
        f"{'p* (BLER<=1e-2)':>17} {'mean ms (low-p)':>17}"
    )
    print(header)
    print("-" * len(header))
    for run in all_runs:
        first = run["results"][0] if run["results"] else None
        mean_ms = first["mean_ms"] if first else float("nan")
        p_star = run["p_star_bler_le_1e-2"]
        p_star_str = f"{p_star:.4f}" if p_star is not None else "  --  "
        dr = run.get("rate_delta_vs_neural_ms", 0.0)
        print(
            f"{run['decoder']:<14} {run['n']:>5} {run['k']:>5} "
            f"{run['rate']:>7.4f} {dr:>+7.4f} {p_star_str:>17} {mean_ms:>17.3f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--decoders",
        type=str,
        default="neural_ms,bch",
        help="Comma-separated list of decoders to benchmark "
        "(neural_ms, bch).",
    )
    p.add_argument("--trials", type=int, default=1000,
                   help="Number of Monte-Carlo trials per (decoder, p).")
    p.add_argument("--bch-trials", type=int, default=0,
                   help="Override --trials for BCH (slowest decoder). "
                        "0 means reuse --trials.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true",
                   help="Disable GPU (set CUDA_VISIBLE_DEVICES='').")
    p.add_argument("--p-grid", type=str, default="",
                   help="Comma-separated custom coarse p-grid. "
                        "Default = built-in.")
    p.add_argument("--out", type=str, default="ecc_benchmark_results.json")
    p.add_argument("--plot", type=str, default="ecc_benchmark_results.png",
                   help="Output PNG with BLER/BER/time charts. "
                        "Empty string disables plotting.")
    p.add_argument(
        "--bch-n", type=int, default=1023,
        help="BCH block length n (default 1023 = 2^10-1, closer to LDPC's n=832).",
    )
    p.add_argument(
        "--bch-k", type=int, default=193,
        help="BCH message dimension k. Default 193 matches Neural-MS rate ~0.19 "
             "among valid galois.BCH(1023,k) (R=0.1887, t=118). "
             "Use n=511,k=10 for the paper's illustrative low-rate example.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    if args.p_grid.strip():
        p_grid = sorted({float(v) for v in args.p_grid.split(",")})
    else:
        p_grid = list(DEFAULT_P_GRID)

    decoder_names = [d.strip() for d in args.decoders.split(",") if d.strip()]
    build_kw_common: Dict[str, Any] = {
        "bch_n": args.bch_n,
        "bch_k": args.bch_k,
    }
    runs: List[Dict[str, Any]] = []
    for name in decoder_names:
        n_trials = args.trials
        if name == "bch" and args.bch_trials > 0:
            n_trials = args.bch_trials
        # Use a per-decoder offset of the seed so each decoder sees an
        # independent noise realization (but reproducible across runs).
        run = run_for_decoder(
            decoder_name=name,
            n_trials=n_trials,
            p_grid=p_grid,
            seed=args.seed + hash(name) % 1000,
            force_cpu=args.cpu_only,
            decoder_build_kw=build_kw_common,
        )
        runs.append(run)

    print_summary(runs)

    out_path = os.path.join(here, args.out) if not os.path.isabs(args.out) else args.out
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "decoders": decoder_names,
                    "trials": args.trials,
                    "bch_trials": args.bch_trials,
                    "seed": args.seed,
                    "cpu_only": args.cpu_only,
                    "p_grid": p_grid,
                    "neural_ms_reference": {
                        "n": NEURAL_MS_N,
                        "k": NEURAL_MS_K,
                        "rate": NEURAL_MS_RATE,
                    },
                    "bch_params": {"n": args.bch_n, "k": args.bch_k},
                },
                "runs": runs,
            },
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o)
            if isinstance(o, (np.integer,)) else o,
        )
    print(f"\nResults written to {out_path}")

    if args.plot:
        plot_path = os.path.join(here, args.plot) if not os.path.isabs(args.plot) else args.plot
        try:
            from benchmark_ecc_plot import plot_results

            plot_results(runs, plot_path)
            print(f"Plot saved to {plot_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to render plot: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
