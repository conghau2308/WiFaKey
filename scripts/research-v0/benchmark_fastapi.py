import argparse
import base64
import glob
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize_ms(values_ms: list[float]) -> dict[str, Any]:
    if not values_ms:
        return {
            "count": 0,
            "mean_ms": None,
            "std_ms": None,
            "min_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    vals = sorted(values_ms)
    std_ms = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "count": len(vals),
        "mean_ms": round(sum(vals) / len(vals), 3),
        "std_ms": round(std_ms, 3),
        "min_ms": round(vals[0], 3),
        "p50_ms": round(percentile(vals, 0.50), 3),
        "p95_ms": round(percentile(vals, 0.95), 3),
        "p99_ms": round(percentile(vals, 0.99), 3),
        "max_ms": round(vals[-1], 3),
    }


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        parsed = {}
        if body:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"raw": body}
        return e.code, parsed


def image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_benchmark(
    base_url: str,
    images_dir: str,
    image_pattern: str,
    max_images: int,
    warmup: int,
    timeout_s: float,
    username_prefix: str,
) -> int:
    image_paths = sorted(glob.glob(os.path.join(images_dir, image_pattern)))
    if not image_paths:
        print(f"[ERROR] No images found in '{images_dir}' with pattern '{image_pattern}'")
        return 1
    if max_images > 0:
        image_paths = image_paths[:max_images]

    print("=" * 70)
    print("WiFaKey FastAPI Benchmark")
    print("=" * 70)
    print(f"FastAPI base URL      : {base_url}")
    print(f"Images dir            : {images_dir}")
    print(f"Pattern               : {image_pattern}")
    print(f"Number of images      : {len(image_paths)}")
    print(f"Warm-up images        : {warmup}")
    print(f"HTTP timeout          : {timeout_s}s")
    print("-" * 70)

    enroll_lat_ms: list[float] = []
    verify_lat_ms: list[float] = []
    enroll_ok = 0
    enroll_fail = 0
    verify_ok = 0
    verify_fail = 0

    for idx, image_path in enumerate(image_paths):
        image_b64 = image_to_b64(image_path)
        username = f"{username_prefix}_{idx:05d}"

        enroll_url = f"{base_url.rstrip('/')}/enroll/{username}"
        t0 = time.perf_counter()
        enroll_status, enroll_body = post_json(enroll_url, {"image": image_b64}, timeout_s)
        enroll_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if idx >= warmup:
            enroll_lat_ms.append(enroll_elapsed_ms)

        if enroll_status != 200:
            enroll_fail += 1
            detail = enroll_body.get("detail", enroll_body)
            print(f"[ENROLL FAIL] idx={idx} status={enroll_status} detail={detail}")
            continue
        enroll_ok += 1

        helper_data_b64 = enroll_body.get("helper_data_b64")
        mask_b64 = enroll_body.get("mask_b64")
        key_hash_b64 = enroll_body.get("key_hash_b64")
        if not (helper_data_b64 and mask_b64 and key_hash_b64):
            enroll_fail += 1
            print(f"[ENROLL FAIL] idx={idx} missing output fields")
            continue

        verify_url = f"{base_url.rstrip('/')}/verify/{username}"
        verify_payload = {
            "image": image_b64,
            "helper_data_b64": helper_data_b64,
            "mask_b64": mask_b64,
            "key_hash_b64": key_hash_b64,
        }
        t1 = time.perf_counter()
        verify_status, verify_body = post_json(verify_url, verify_payload, timeout_s)
        verify_elapsed_ms = (time.perf_counter() - t1) * 1000.0
        if idx >= warmup:
            verify_lat_ms.append(verify_elapsed_ms)

        if verify_status != 200:
            verify_fail += 1
            detail = verify_body.get("detail", verify_body)
            print(f"[VERIFY FAIL] idx={idx} status={verify_status} detail={detail}")
            continue
        if verify_body.get("success") is True:
            verify_ok += 1
        else:
            verify_fail += 1

    print("Request success summary:")
    print(f"  - enroll_ok     : {enroll_ok}")
    print(f"  - enroll_fail   : {enroll_fail}")
    print(f"  - verify_ok     : {verify_ok}")
    print(f"  - verify_fail   : {verify_fail}")
    print("-" * 70)

    print("FastAPI endpoint latency stats (ms):")
    for name, values in [("POST /enroll/{username}", enroll_lat_ms), ("POST /verify/{username}", verify_lat_ms)]:
        s = summarize_ms(values)
        print(
            f"{name}\n"
            f"  count={s['count']} mean={s['mean_ms']} std={s['std_ms']} "
            f"min={s['min_ms']} p50={s['p50_ms']} p95={s['p95_ms']} p99={s['p99_ms']} max={s['max_ms']}"
        )
    print("-" * 70)
    print("Done.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WiFaKey FastAPI endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL.")
    parser.add_argument("--images-dir", required=True, help="Directory containing input images.")
    parser.add_argument("--image-pattern", default="*.jpg", help="Glob pattern for images. Default: *.jpg")
    parser.add_argument("--max-images", type=int, default=0, help="Max images to process. 0 means all (full images).")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up samples ignored per endpoint.")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout (seconds).")
    parser.add_argument("--username-prefix", default="bench", help="Prefix used to form username path parameter.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_benchmark(
            base_url=args.base_url,
            images_dir=args.images_dir,
            image_pattern=args.image_pattern,
            max_images=args.max_images,
            warmup=args.warmup,
            timeout_s=args.timeout_s,
            username_prefix=args.username_prefix,
        )
    )
