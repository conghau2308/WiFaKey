"""End-to-end TAR, FAR, and timing benchmark for WiFaKey ECC pipelines.

Default: two handlers only (same biometric front-end: FaceProcessor + AdaFace +
M_matrix + LSSC + Bernoulli kappa mask):

    * neural_ms : wifakey_module.wifakey_handler.WiFaKeyHandler
    * bch       : wifakey_module.wifakey_handler_bch.WiFaKeyBCHHandler

Phase 1 caches AdaFace embeddings once for the whole dataset; phase 2 then
runs each handler over the same embeddings so face detection + AdaFace cost
is paid only once.

Metrics per handler:
    * intra (genuine) pairs i<j of every user      -> TAR only (no FRR column)
    * inter (impostor) pairs sampled across users  -> FAR
    * mean / p50 / p95 of enroll() and verify() ms

Usage (from WiFaKey_252):

    python test_ACC_compare.py --max-users 5 --max-images 3 --inter-pairs 50 \
        --out test_ACC_compare.json --plot test_ACC_compare.png

For full dataset, drop --max-users / --max-images.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_DATASET_ROOT = r"G:\archive_3\Selfies ID Images dataset"
DEFAULT_HANDLERS = ["neural_ms", "bch"]
ALLOWED_HANDLERS = frozenset({"neural_ms", "bch"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--image-glob", type=str, default="*.jpg")
    p.add_argument("--max-users", type=int, default=0,
                   help="Process at most N users (0 = all).")
    p.add_argument("--max-images", type=int, default=0,
                   help="Process at most N images per user (0 = all).")
    p.add_argument("--inter-pairs", type=int, default=1000,
                   help="Sampled impostor pairs per handler (FAR).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--handlers", type=str, default=",".join(DEFAULT_HANDLERS),
        help="Comma-separated handlers (default: neural_ms,bch).",
    )
    p.add_argument("--cache", type=str, default="embeddings_cache.npz",
                   help="Path to cache AdaFace embeddings (.npz).")
    p.add_argument("--use-cache", action="store_true",
                   help="If set and --cache file exists, skip face/AdaFace step.")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force rerun face/AdaFace even if --cache file exists.")
    p.add_argument("--out", type=str, default="test_ACC_compare.json")
    p.add_argument("--plot", type=str, default="test_ACC_compare.png",
                   help="Output PNG (empty disables plotting).")
    p.add_argument("--ctx-id", type=int, default=0,
                   help="FaceProcessor ctx_id (0=GPU, -1=CPU).")
    p.add_argument("--adaface-device", type=str, default="cuda")
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--quiet-handler-prints", action="store_true",
                   help="Silence handler.verify built-in prints (Neural-MS).")
    return p.parse_args()


def list_user_folders(root: str, max_users: int) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")
    subs = [f.path for f in os.scandir(root) if f.is_dir()]
    numeric = [s for s in subs if os.path.basename(s).isdigit()]
    if numeric:
        numeric.sort(key=lambda p: int(os.path.basename(p)))
        subs = numeric
    else:
        subs.sort()
    if max_users > 0:
        subs = subs[:max_users]
    return subs


def extract_embeddings(
    folders: List[str],
    image_glob: str,
    max_images: int,
    fp: Any,
    ada: Any,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Returns ({user_id: [{name, vector}]}, status_counts)."""
    import cv2

    out: Dict[str, List[Dict[str, Any]]] = {}
    status_counts: Dict[str, int] = {}

    t0 = time.perf_counter()
    for idx, folder in enumerate(folders, 1):
        uid = os.path.basename(folder)
        paths = sorted(glob.glob(os.path.join(folder, image_glob)))
        if max_images > 0:
            paths = paths[:max_images]
        valid: List[Dict[str, Any]] = []

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                status_counts["read_failed"] = status_counts.get("read_failed", 0) + 1
                continue
            aligned, status = fp.process(img)
            status_counts[status] = status_counts.get(status, 0) + 1
            if aligned is None:
                continue
            try:
                vec = ada.get_feature_vector(aligned).astype(np.float32)
            except Exception:
                status_counts["adaface_failed"] = status_counts.get("adaface_failed", 0) + 1
                continue
            valid.append({"name": os.path.basename(p), "vector": vec})
            status_counts["used"] = status_counts.get("used", 0) + 1

        out[uid] = valid
        if idx % 10 == 0 or idx == len(folders):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{idx:>4}/{len(folders)}] user {uid}: kept {len(valid)} imgs"
                f"  | elapsed {elapsed:.1f}s"
            )
    return out, status_counts


def save_cache(path: str, embeddings: Dict[str, List[Dict[str, Any]]]) -> None:
    user_ids: List[str] = []
    names: List[str] = []
    vectors: List[np.ndarray] = []
    counts: List[int] = []
    for uid, items in embeddings.items():
        counts.append(len(items))
        user_ids.append(uid)
        for it in items:
            names.append(it["name"])
            vectors.append(it["vector"])
    if vectors:
        vec_arr = np.stack(vectors).astype(np.float32)
    else:
        vec_arr = np.zeros((0, 0), dtype=np.float32)
    np.savez(
        path,
        user_ids=np.array(user_ids, dtype=object),
        counts=np.array(counts, dtype=np.int64),
        names=np.array(names, dtype=object),
        vectors=vec_arr,
    )
    print(f"[cache] saved {sum(counts)} embeddings for {len(user_ids)} users to {path}")


def load_cache(path: str) -> Dict[str, List[Dict[str, Any]]]:
    data = np.load(path, allow_pickle=True)
    user_ids = list(data["user_ids"])
    counts = list(data["counts"])
    names = list(data["names"])
    vectors = data["vectors"]
    out: Dict[str, List[Dict[str, Any]]] = {}
    pos = 0
    name_pos = 0
    for uid, c in zip(user_ids, counts):
        items = []
        for _ in range(int(c)):
            items.append({"name": str(names[name_pos]), "vector": vectors[pos]})
            pos += 1
            name_pos += 1
        out[str(uid)] = items
    print(f"[cache] loaded {pos} embeddings for {len(user_ids)} users from {path}")
    return out


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q * 100.0))


def time_summary(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"),
                "p95_ms": float("nan"), "count": 0}
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50.0)),
        "p95_ms": float(np.percentile(arr, 95.0)),
        "count": int(arr.size),
    }


def build_handler(name: str):
    if name == "neural_ms":
        from wifakey_module.wifakey_handler import WiFaKeyHandler
        return WiFaKeyHandler()
    if name == "bch":
        from wifakey_module.wifakey_handler_bch import WiFaKeyBCHHandler
        return WiFaKeyBCHHandler()
    raise ValueError(f"Unknown handler: {name}")


def teardown_handler(handler) -> None:
    try:
        if hasattr(handler, "sess") and handler.sess is not None:
            handler.sess.close()
    except Exception:
        pass
    try:
        del handler
    except Exception:
        pass
    gc.collect()
    # Reset TF graph to free memory between handlers (Neural-MS uses TF1).
    try:
        import tensorflow.compat.v1 as tf  # noqa
        tf.reset_default_graph()
    except Exception:
        pass


def evaluate_handler(
    name: str,
    handler,
    embeddings: Dict[str, List[Dict[str, Any]]],
    inter_pairs: int,
    rng: random.Random,
) -> Dict[str, Any]:
    print(f"\n{'='*70}\nEvaluating handler: {name}\n{'='*70}")

    intra_pairs = 0
    intra_success = 0
    intra_fail = 0
    enroll_ms: List[float] = []
    verify_intra_ms: List[float] = []
    fail_examples: List[str] = []

    user_ids = sorted(embeddings.keys())
    user_ids_with_data = [u for u in user_ids if len(embeddings[u]) >= 1]

    # Warm-up to avoid penalising the first measurement.
    if user_ids_with_data:
        first_emb = embeddings[user_ids_with_data[0]][0]["vector"]
        try:
            hd, mr, kh = handler.enroll(first_emb)
            handler.verify(first_emb, hd, mr, kh)
        except Exception as exc:
            print(f"[warn] warm-up failed for {name}: {exc}")

    # ---- intra pairs (TAR) ----
    for uid in user_ids_with_data:
        items = embeddings[uid]
        if len(items) < 2:
            continue
        for i in range(len(items)):
            try:
                t0 = time.perf_counter()
                helper, mask_r, key_hash = handler.enroll(items[i]["vector"])
                enroll_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                fail_examples.append(f"enroll user={uid} img={items[i]['name']} err={exc}")
                continue
            for j in range(i + 1, len(items)):
                intra_pairs += 1
                try:
                    t0 = time.perf_counter()
                    ok = handler.verify(items[j]["vector"], helper, mask_r, key_hash)
                    verify_intra_ms.append((time.perf_counter() - t0) * 1000.0)
                except Exception as exc:
                    intra_fail += 1
                    fail_examples.append(
                        f"verify_intra user={uid} {items[i]['name']}->{items[j]['name']} err={exc}"
                    )
                    continue
                if ok:
                    intra_success += 1
                else:
                    intra_fail += 1
                    if len(fail_examples) < 30:
                        fail_examples.append(
                            f"intra_reject user={uid} {items[i]['name']}->{items[j]['name']}"
                        )

    tar = (intra_success / intra_pairs) if intra_pairs else float("nan")
    print(f"  intra pairs={intra_pairs}  TAR={tar*100:.2f}%")

    # ---- inter pairs (FAR) ----
    inter_attempts = 0
    inter_false_accept = 0
    verify_inter_ms: List[float] = []
    eligible_users = [u for u in user_ids_with_data if len(embeddings[u]) >= 1]
    if len(eligible_users) >= 2 and inter_pairs > 0:
        seen: set = set()
        target = inter_pairs
        attempts_cap = max(target * 40, 200)
        attempts = 0
        while inter_attempts < target and attempts < attempts_cap:
            attempts += 1
            victim, impostor = rng.sample(eligible_users, 2)
            i_v = 0  # enroll always uses the first valid image of the victim
            j = rng.randrange(len(embeddings[impostor]))
            key = (victim, i_v, impostor, j)
            if key in seen:
                continue
            seen.add(key)
            try:
                helper, mask_r, key_hash = handler.enroll(embeddings[victim][i_v]["vector"])
            except Exception:
                continue
            try:
                t0 = time.perf_counter()
                ok = handler.verify(
                    embeddings[impostor][j]["vector"], helper, mask_r, key_hash
                )
                verify_inter_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                continue
            inter_attempts += 1
            if ok:
                inter_false_accept += 1
    far = (inter_false_accept / inter_attempts) if inter_attempts else float("nan")
    print(
        f"  inter pairs={inter_attempts}  FAR={far*100:.4f}%  (FA={inter_false_accept})"
    )

    enroll_summary = time_summary(enroll_ms)
    verify_intra_summary = time_summary(verify_intra_ms)
    verify_inter_summary = time_summary(verify_inter_ms)
    print(
        f"  enroll  mean={enroll_summary['mean_ms']:.2f} ms"
        f"  p95={enroll_summary['p95_ms']:.2f} ms"
    )
    print(
        f"  verify (intra) mean={verify_intra_summary['mean_ms']:.2f} ms"
        f"  p95={verify_intra_summary['p95_ms']:.2f} ms"
    )

    return {
        "handler": name,
        "n": int(getattr(handler, "feature_length", -1)),
        "k": int(getattr(handler, "key_length", -1)),
        "rate": float(getattr(handler, "rate", float("nan")))
        if hasattr(handler, "rate") else None,
        "kappa": float(getattr(handler, "kappa", float("nan"))),
        "intra": {
            "pairs": intra_pairs,
            "success": intra_success,
            "fail": intra_fail,
            "tar": tar,
        },
        "inter": {
            "pairs": inter_attempts,
            "false_accept": inter_false_accept,
            "far": far,
        },
        "time": {
            "enroll_ms": enroll_summary,
            "verify_intra_ms": verify_intra_summary,
            "verify_inter_ms": verify_inter_summary,
        },
        "fail_examples": fail_examples[:50],
    }


def print_compare_table(runs: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("END-TO-END COMPARISON")
    print("=" * 92)
    header = (
        f"{'handler':<12} {'n':>5} {'k':>5} {'rate':>7} "
        f"{'TAR%':>7} {'FAR%':>8} "
        f"{'enroll ms':>10} {'verify ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        rate = r.get("rate")
        rate_str = f"{rate:.4f}" if isinstance(rate, float) else "  --  "
        tar = r["intra"].get("tar", float("nan")) * 100
        far = r["inter"].get("far", float("nan")) * 100
        em = r["time"]["enroll_ms"].get("mean_ms", float("nan"))
        vm = r["time"]["verify_intra_ms"].get("mean_ms", float("nan"))
        print(
            f"{r['handler']:<12} {r['n']:>5} {r['k']:>5} {rate_str:>7} "
            f"{tar:>7.2f} {far:>8.4f} "
            f"{em:>10.2f} {vm:>10.2f}"
        )


def render_plot(out_path: str, runs: List[Dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    handlers = [r["handler"] for r in runs]
    tar = [r["intra"].get("tar", float("nan")) * 100 for r in runs]
    far = [r["inter"].get("far", float("nan")) * 100 for r in runs]
    enroll_ms = [r["time"]["enroll_ms"].get("mean_ms", float("nan")) for r in runs]
    verify_ms = [r["time"]["verify_intra_ms"].get("mean_ms", float("nan")) for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(handlers))
    width = 0.35

    ax0 = axes[0]
    ax0.bar(x - width / 2, tar, width, label="TAR%", color="tab:green")
    ax0.bar(x + width / 2, far, width, label="FAR%", color="tab:red")
    ax0.set_xticks(x)
    ax0.set_xticklabels(handlers)
    ax0.set_ylabel("rate (%)")
    ax0.set_title("Genuine / impostor outcomes")
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend()

    ax1 = axes[1]
    ax1.bar(x - width / 2, enroll_ms, width, label="enroll mean", color="tab:blue")
    ax1.bar(x + width / 2, verify_ms, width, label="verify mean", color="tab:purple")
    ax1.set_xticks(x)
    ax1.set_xticklabels(handlers)
    ax1.set_ylabel("ms")
    ax1.set_title("Enroll / Verify mean latency")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    fig.suptitle("WiFaKey ECC end-to-end comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def maybe_quiet_handler_prints():
    """Optionally suppress the print() inside WiFaKeyHandler.verify."""
    import builtins
    real_print = builtins.print

    def filtered(*args, **kwargs):
        if args and isinstance(args[0], str) and (
            args[0].startswith("[WiFaKey] Verify ")
        ):
            return
        return real_print(*args, **kwargs)
    builtins.print = filtered


def main() -> int:
    args = parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    handler_names = [h.strip() for h in args.handlers.split(",") if h.strip()]
    for h in handler_names:
        if h not in ALLOWED_HANDLERS:
            raise SystemExit(
                f"Unknown handler: {h}. Choose from {sorted(ALLOWED_HANDLERS)}."
            )

    cache_path = (
        args.cache if os.path.isabs(args.cache) else os.path.join(here, args.cache)
    )

    embeddings: Dict[str, List[Dict[str, Any]]]
    status_counts: Dict[str, int] = {}
    if args.use_cache and not args.rebuild_cache and os.path.exists(cache_path):
        embeddings = load_cache(cache_path)
    else:
        folders = list_user_folders(args.dataset_root, args.max_users)
        print(f"[info] Dataset root : {args.dataset_root}")
        print(f"[info] User folders : {len(folders)}")

        print("[init] FaceProcessor ...")
        from vision_module.face_processor import FaceProcessor
        fp = FaceProcessor(
            ctx_id=args.ctx_id,
            confidence_threshold=args.confidence_threshold,
        )
        print("[init] AdaFaceExtractor ...")
        from feature_extractor.adaface_handler import AdaFaceExtractor
        ada = AdaFaceExtractor(device=args.adaface_device)

        embeddings, status_counts = extract_embeddings(
            folders, args.image_glob, args.max_images, fp, ada
        )
        try:
            save_cache(cache_path, embeddings)
        except Exception as exc:
            print(f"[warn] could not save cache: {exc}")

        # Free face/AdaFace before loading WiFaKey handlers.
        try:
            del fp, ada
        except Exception:
            pass
        gc.collect()

    n_users_with_bits = sum(1 for v in embeddings.values() if len(v) > 0)
    n_total_imgs = sum(len(v) for v in embeddings.values())
    print(
        f"[info] users with embeddings: {n_users_with_bits}, total images: {n_total_imgs}"
    )

    if args.quiet_handler_prints:
        maybe_quiet_handler_prints()

    runs: List[Dict[str, Any]] = []
    for name in handler_names:
        try:
            handler = build_handler(name)
        except Exception as exc:
            print(f"[error] failed to build handler {name}: {exc}")
            continue
        try:
            run = evaluate_handler(
                name, handler, embeddings, args.inter_pairs, rng
            )
            runs.append(run)
        finally:
            teardown_handler(handler)

    print_compare_table(runs)

    payload = {
        "config": {
            "dataset_root": args.dataset_root,
            "image_glob": args.image_glob,
            "max_users": args.max_users,
            "max_images": args.max_images,
            "inter_pairs": args.inter_pairs,
            "seed": args.seed,
            "handlers": handler_names,
        },
        "stats": {
            "users_with_embeddings": n_users_with_bits,
            "images_total": n_total_imgs,
            "extract_status_counts": status_counts,
        },
        "runs": runs,
    }

    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o)
                  if isinstance(o, (np.floating,))
                  else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"\n[info] JSON written to {out_path}")

    if args.plot:
        plot_path = args.plot if os.path.isabs(args.plot) else os.path.join(here, args.plot)
        try:
            render_plot(plot_path, runs)
            print(f"[info] Plot saved to {plot_path}")
        except Exception as exc:
            print(f"[warn] failed to plot: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
