"""Measure real intra-class / inter-class bit error rate (BER) on a face dataset.

Pipeline per image:
    image -> FaceProcessor.process -> aligned RGB 112x112
          -> AdaFaceExtractor.get_feature_vector -> 512-d float
          -> WiFaKeyHandler._binarize_full -> long binary vector
          -> take first ``--bit-length`` bits (default 832 = WiFaKey n)

Then we compute Hamming-distance/n between bit vectors:
    * intra-class: pairs of images of the SAME user
    * inter-class: pairs of images of DIFFERENT users (random sample)

We also report **masked** BER that matches enroll/verify:
    * One random ``mask_r`` per user (same Bernoulli(kappa) rule as
      ``WiFaKeyHandler.enroll``), length ``len(b_full)``, seeded from
      ``--seed`` + user id so runs are reproducible.
    * Intra: compare ``(b_full_i & mask_r)[:n]`` vs ``(b_full_j & mask_r)[:n]``
      for the same user — same mask as stored at enrollment, new probe image.
    * Inter: pick a **victim** user ``U`` (the enrolled identity). Compare
      ``(b_full_probe & mask_r_U)[:n]`` vs ``(b_full_genuine_U & mask_r_U)[:n]``
      where the probe comes from a different user — same mask as at verify
      when an impostor presents.

The **raw** (unmasked) BER is an upper bound on Hamming friction; **masked**
numbers align with paper Eq. 4 and the bits fed to the ECC path.

Usage (from WiFaKey_252 directory):

    python measure_real_ber.py --max-users 20 --max-images 5 \
        --out real_ber_results.json --plot real_ber_histogram.png

Use ``--max-users 5 --max-images 3`` for a quick smoke run (~30s after init).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_DATASET_ROOT = r"G:\archive_3\Selfies ID Images dataset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root.  Each immediate subfolder = one user.",
    )
    p.add_argument(
        "--image-glob",
        type=str,
        default="*.jpg",
        help="Glob pattern for images inside each user folder.",
    )
    p.add_argument(
        "--max-users", type=int, default=0,
        help="Process at most N users (0 = all).",
    )
    p.add_argument(
        "--max-images", type=int, default=0,
        help="Process at most N images per user (0 = all).",
    )
    p.add_argument(
        "--inter-pairs", type=int, default=5000,
        help="Sample at most N random inter-user pairs.",
    )
    p.add_argument(
        "--bit-length", type=int, default=832,
        help="Number of leading bits taken from b_full (must match ECC n).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for inter-pair sampling.",
    )
    p.add_argument(
        "--out", type=str, default="real_ber_results.json",
        help="Where to write the raw + summary JSON.",
    )
    p.add_argument(
        "--plot", type=str, default="real_ber_histogram.png",
        help="Output PNG (empty string disables).",
    )
    p.add_argument(
        "--ctx-id", type=int, default=0,
        help="FaceProcessor ctx_id (0=GPU, -1=CPU).",
    )
    p.add_argument(
        "--adaface-device", type=str, default="cuda",
        help="AdaFace device: cuda or cpu.",
    )
    p.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="Face detection confidence threshold.",
    )
    return p.parse_args()


def init_modules(args: argparse.Namespace):
    print("[init] FaceProcessor ...")
    from vision_module.face_processor import FaceProcessor

    fp = FaceProcessor(
        ctx_id=args.ctx_id,
        confidence_threshold=args.confidence_threshold,
    )

    print("[init] AdaFaceExtractor ...")
    from feature_extractor.adaface_handler import AdaFaceExtractor

    ada = AdaFaceExtractor(device=args.adaface_device)

    print("[init] WiFaKeyHandler (only used for _binarize_full) ...")
    from wifakey_module.wifakey_handler import WiFaKeyHandler

    wifakey = WiFaKeyHandler()

    return fp, ada, wifakey


def list_user_folders(root: str, max_users: int) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")
    subs = [f.path for f in os.scandir(root) if f.is_dir()]
    # Match test_ACC.py: prefer numeric folder names sorted as int.
    numeric = [s for s in subs if os.path.basename(s).isdigit()]
    if numeric:
        numeric.sort(key=lambda p: int(os.path.basename(p)))
        subs = numeric
    else:
        subs.sort()
    if max_users > 0:
        subs = subs[:max_users]
    return subs


def collect_user_bits(
    user_folder: str,
    image_glob: str,
    max_images: int,
    bit_length: int,
    fp: Any,
    ada: Any,
    wifakey: Any,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """Return (list of bit vectors, status counts)."""
    paths = sorted(glob.glob(os.path.join(user_folder, image_glob)))
    if max_images > 0:
        paths = paths[:max_images]
    import cv2

    bits: List[np.ndarray] = []
    counts: Dict[str, int] = {}

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            counts["read_failed"] = counts.get("read_failed", 0) + 1
            continue

        aligned, status = fp.process(img)
        counts[status] = counts.get(status, 0) + 1
        if aligned is None:
            continue

        try:
            emb = ada.get_feature_vector(aligned).astype(np.float32)
        except Exception:
            counts["adaface_failed"] = counts.get("adaface_failed", 0) + 1
            continue

        try:
            b_full = wifakey._binarize_full(emb).astype(np.uint8)
        except Exception:
            counts["binarize_failed"] = counts.get("binarize_failed", 0) + 1
            continue

        if b_full.size < bit_length:
            counts["too_short"] = counts.get("too_short", 0) + 1
            continue

        # Keep full LSSC vector so masking matches WiFaKeyHandler (mask length = len(b_full)).
        bits.append(b_full.astype(np.uint8).copy())
        counts["used"] = counts.get("used", 0) + 1

    return bits, counts


def enroll_mask_rng_seed(global_seed: int, user_id: str) -> int:
    """Stable 64-bit seed for per-user mask RNG (reproducible across runs)."""
    digest = hashlib.sha256(f"{global_seed}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def make_enroll_mask(b_full_length: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Same rule as WiFaKeyHandler.enroll: mask_r = (u >= kappa) with u ~ Uniform(0,1)."""
    u = rng.random(b_full_length, dtype=np.float64)
    return (u >= float(kappa)).astype(np.uint8)


def masked_prefix(b_full: np.ndarray, mask_r: np.ndarray, bit_length: int) -> np.ndarray:
    return (b_full.astype(np.uint8) & mask_r.astype(np.uint8))[:bit_length]


def build_user_masks(
    users_b_full: Dict[str, List[np.ndarray]],
    kappa: float,
    global_seed: int,
) -> Dict[str, np.ndarray]:
    """One enroll-style mask per user (reproducible). Skips users with no vectors."""
    out: Dict[str, np.ndarray] = {}
    for uid, vec_list in users_b_full.items():
        if not vec_list:
            continue
        n = int(vec_list[0].size)
        rng = np.random.default_rng(enroll_mask_rng_seed(global_seed, uid))
        out[uid] = make_enroll_mask(n, kappa, rng)
    return out


def compute_intra_pairs_masked(
    users_b_full: Dict[str, List[np.ndarray]],
    masks: Dict[str, np.ndarray],
    bit_length: int,
) -> List[float]:
    out: List[float] = []
    for uid, vec_list in users_b_full.items():
        r = masks.get(uid)
        if r is None or len(vec_list) < 2:
            continue
        masked = [masked_prefix(b, r, bit_length) for b in vec_list]
        for i in range(len(masked)):
            for j in range(i + 1, len(masked)):
                out.append(hamming_ber(masked[i], masked[j]))
    return out


def compute_inter_pairs_masked(
    users_b_full: Dict[str, List[np.ndarray]],
    masks: Dict[str, np.ndarray],
    n_pairs: int,
    rng: random.Random,
    bit_length: int,
) -> List[float]:
    """
    Inter-user BER under the **victim's** enrolled mask (verify behavior).

    Each sample: pick distinct users (victim U, impostor W), pick images
    i from U and j from W, Hamming( (b_W[j] & r_U)[:L], (b_U[i] & r_U)[:L] ).
    """
    user_ids = [uid for uid, b in users_b_full.items() if len(b) > 0 and uid in masks]
    if len(user_ids) < 2:
        return []

    out: List[float] = []
    seen: set = set()
    attempts = 0
    target = max(n_pairs, 0)
    while len(out) < target and attempts < target * 40:
        attempts += 1
        victim, impostor = rng.sample(user_ids, 2)
        r_v = masks[victim]
        bu = users_b_full[victim]
        bw = users_b_full[impostor]
        i = rng.randrange(len(bu))
        j = rng.randrange(len(bw))
        key = (victim, i, impostor, j)
        if key in seen:
            continue
        seen.add(key)
        a = masked_prefix(bw[j], r_v, bit_length)
        b = masked_prefix(bu[i], r_v, bit_length)
        out.append(hamming_ber(a, b))
    return out


def hamming_ber(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a ^ b)) / float(a.size)


def compute_intra_pairs(
    users_bits: Dict[str, List[np.ndarray]],
    bit_length: int,
) -> List[float]:
    out: List[float] = []
    for _, bits in users_bits.items():
        if len(bits) < 2:
            continue
        for i in range(len(bits)):
            for j in range(i + 1, len(bits)):
                out.append(hamming_ber(bits[i][:bit_length], bits[j][:bit_length]))
    return out


def compute_inter_pairs(
    users_bits: Dict[str, List[np.ndarray]],
    n_pairs: int,
    rng: random.Random,
    bit_length: int,
) -> List[float]:
    user_ids = [uid for uid, b in users_bits.items() if len(b) > 0]
    if len(user_ids) < 2:
        return []

    out: List[float] = []
    seen: set = set()
    attempts = 0
    target = max(n_pairs, 0)
    while len(out) < target and attempts < target * 20:
        attempts += 1
        u1, u2 = rng.sample(user_ids, 2)
        i = rng.randrange(len(users_bits[u1]))
        j = rng.randrange(len(users_bits[u2]))
        key = (u1, i, u2, j) if (u1, i) < (u2, j) else (u2, j, u1, i)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            hamming_ber(
                users_bits[u1][i][:bit_length],
                users_bits[u2][j][:bit_length],
            )
        )
    return out


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _draw_ber_panel(
    ax: Any,
    intra: List[float],
    inter: List[float],
    bit_length: int,
    title: str,
) -> None:
    bins = np.linspace(0.0, 0.55, 56)
    if intra:
        ax.hist(
            intra,
            bins=bins,
            alpha=0.55,
            label=f"intra  (n={len(intra)})",
            density=True,
            color="tab:blue",
        )
    if inter:
        ax.hist(
            inter,
            bins=bins,
            alpha=0.55,
            label=f"inter  (n={len(inter)})",
            density=True,
            color="tab:orange",
        )
    ax.set_xlabel(f"BER (Hamming / {bit_length})")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def render_plot(
    out_path: str,
    intra: List[float],
    inter: List[float],
    n_users: int,
    bit_length: int,
    intra_m: Optional[List[float]] = None,
    inter_m: Optional[List[float]] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_masked = intra_m is not None and inter_m is not None
    if has_masked:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        _draw_ber_panel(
            axes[0],
            intra,
            inter,
            bit_length,
            title=f"Raw first-{bit_length} bits (no mask)",
        )
        _draw_ber_panel(
            axes[1],
            intra_m,
            inter_m,
            bit_length,
            title="Masked (victim enroll mask @ verify)",
        )
        fig.suptitle(
            f"Real BER distribution  |  users={n_users}, bit_len={bit_length}",
            fontsize=12,
        )
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        _draw_ber_panel(
            ax,
            intra,
            inter,
            bit_length,
            title=f"Real intra/inter BER (users={n_users}, bit_len={bit_length})",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    folders = list_user_folders(args.dataset_root, args.max_users)
    print(f"[info] Dataset root : {args.dataset_root}")
    print(f"[info] User folders : {len(folders)}")

    fp, ada, wifakey = init_modules(args)

    users_bits: Dict[str, List[np.ndarray]] = {}
    status_total: Dict[str, int] = {}
    images_per_user: List[int] = []

    t0 = time.perf_counter()
    for idx, folder in enumerate(folders, 1):
        uid = os.path.basename(folder)
        bits, counts = collect_user_bits(
            folder, args.image_glob, args.max_images,
            args.bit_length, fp, ada, wifakey,
        )
        users_bits[uid] = bits
        for k, v in counts.items():
            status_total[k] = status_total.get(k, 0) + v
        images_per_user.append(len(bits))
        if idx % 10 == 0 or idx == len(folders):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{idx:>4}/{len(folders)}] user {uid}: kept {len(bits)} imgs"
                f"  | total used so far: {sum(images_per_user)}"
                f"  | {elapsed:.1f}s"
            )

    print("\n[info] Per-image status counts:")
    for k in sorted(status_total.keys()):
        print(f"  - {k}: {status_total[k]}")

    intra = compute_intra_pairs(users_bits, args.bit_length)
    inter = compute_inter_pairs(users_bits, args.inter_pairs, rng, args.bit_length)

    print(f"\n[info] intra-class pairs (raw): {len(intra)}")
    print(f"[info] inter-class pairs (raw): {len(inter)}")

    intra_summary = summarize(intra)
    inter_summary = summarize(inter)
    print("\n=== intra-class BER summary (raw) ===")
    print(json.dumps(intra_summary, indent=2))
    print("\n=== inter-class BER summary (raw) ===")
    print(json.dumps(inter_summary, indent=2))

    masks = build_user_masks(users_bits, float(wifakey.kappa), args.seed)
    intra_m = compute_intra_pairs_masked(users_bits, masks, args.bit_length)
    inter_m = compute_inter_pairs_masked(
        users_bits, masks, args.inter_pairs, rng, args.bit_length
    )
    print(f"\n[info] intra-class pairs (masked): {len(intra_m)}")
    print(f"[info] inter-class pairs (masked, victim mask): {len(inter_m)}")
    intra_m_summary = summarize(intra_m)
    inter_m_summary = summarize(inter_m)
    print("\n=== intra-class BER summary (masked, same enroll mask) ===")
    print(json.dumps(intra_m_summary, indent=2))
    print("\n=== inter-class BER summary (masked, victim enroll mask) ===")
    print(json.dumps(inter_m_summary, indent=2))

    payload = {
        "config": {
            "dataset_root": args.dataset_root,
            "image_glob": args.image_glob,
            "max_users": args.max_users,
            "max_images": args.max_images,
            "inter_pairs_target": args.inter_pairs,
            "bit_length": args.bit_length,
            "seed": args.seed,
            "wifakey_kappa": float(wifakey.kappa),
            "masking_note": (
                "masked: one Bernoulli mask per user (enroll-style); "
                "intra = same mask on all probes; inter = victim user's mask on impostor bits"
            ),
        },
        "stats": {
            "users_total": len(folders),
            "users_with_bits": int(sum(1 for v in users_bits.values() if len(v) > 0)),
            "images_used_total": int(sum(images_per_user)),
            "images_per_user": images_per_user,
            "status_counts": status_total,
            "intra_pairs": len(intra),
            "inter_pairs": len(inter),
            "intra_pairs_masked": len(intra_m),
            "inter_pairs_masked": len(inter_m),
        },
        "intra_summary": intra_summary,
        "inter_summary": inter_summary,
        "intra_summary_masked": intra_m_summary,
        "inter_summary_masked": inter_m_summary,
        "intra_ber": intra,
        "inter_ber": inter,
        "intra_ber_masked": intra_m,
        "inter_ber_masked": inter_m,
    }

    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[info] JSON written to {out_path}")

    if args.plot:
        plot_path = args.plot if os.path.isabs(args.plot) else os.path.join(here, args.plot)
        render_plot(
            plot_path,
            intra,
            inter,
            n_users=int(payload["stats"]["users_with_bits"]),
            bit_length=args.bit_length,
            intra_m=intra_m,
            inter_m=inter_m,
        )
        print(f"[info] Plot saved to {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
