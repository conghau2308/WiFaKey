import argparse
import glob
import os
import statistics
import time
from typing import Any

import cv2
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


def benchmark_modules(
    images_dir: str,
    image_pattern: str,
    max_images: int,
    warmup: int,
    ctx_id: int,
    confidence_threshold: float,
    adaface_device: str,
) -> int:
    # Import heavy modules lazily so --help stays fast.
    from vision_module.face_processor import FaceProcessor
    from feature_extractor.adaface_handler import AdaFaceExtractor
    from wifakey_module.wifakey_handler import WiFaKeyHandler

    image_paths = sorted(glob.glob(os.path.join(images_dir, image_pattern)))
    if not image_paths:
        print(f"[ERROR] No images found in '{images_dir}' with pattern '{image_pattern}'")
        return 1

    if max_images > 0:
        image_paths = image_paths[:max_images]

    print("=" * 70)
    print("WiFaKey Module Performance Benchmark")
    print("=" * 70)
    print(f"Images dir           : {images_dir}")
    print(f"Pattern              : {image_pattern}")
    print(f"Number of images     : {len(image_paths)}")
    print(f"Warm-up images       : {warmup}")
    print(f"FaceProcessor ctx_id : {ctx_id}")
    print(f"AdaFace device       : {adaface_device}")
    print("-" * 70)

    # Measure initialization time for each module.
    t0 = time.perf_counter()
    face_processor = FaceProcessor(
        det_model="buffalo_l",
        ctx_id=ctx_id,
        confidence_threshold=confidence_threshold,
    )
    face_init_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    adaface = AdaFaceExtractor(device=adaface_device)
    adaface_init_ms = (time.perf_counter() - t1) * 1000.0

    wifakey_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wifakey_module", "data")
    t2 = time.perf_counter()
    wifakey = WiFaKeyHandler(
        data_path=wifakey_data,
        weights_path=os.path.join(wifakey_data, "Weights_Var_MS"),
        biases_path=os.path.join(wifakey_data, "Biases_Var_MS"),
    )
    wifakey_init_ms = (time.perf_counter() - t2) * 1000.0

    print("Initialization times:")
    print(f"  - FaceProcessor : {face_init_ms:.3f} ms")
    print(f"  - AdaFace       : {adaface_init_ms:.3f} ms")
    print(f"  - WiFaKey       : {wifakey_init_ms:.3f} ms")
    print("-" * 70)

    face_lat_ms: list[float] = []
    adaface_lat_ms: list[float] = []
    enroll_lat_ms: list[float] = []
    verify_lat_ms: list[float] = []
    status_counts: dict[str, int] = {}

    aligned_faces: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []

    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            status_counts["image_read_failed"] = status_counts.get("image_read_failed", 0) + 1
            continue

        # Face module timing
        t_face = time.perf_counter()
        aligned_face, status = face_processor.process(img)
        face_elapsed_ms = (time.perf_counter() - t_face) * 1000.0
        if idx >= warmup:
            face_lat_ms.append(face_elapsed_ms)

        status_counts[status] = status_counts.get(status, 0) + 1
        if aligned_face is None:
            continue
        aligned_faces.append(aligned_face)

    for idx, aligned_face in enumerate(aligned_faces):
        t_ada = time.perf_counter()
        emb = adaface.get_feature_vector(aligned_face)
        ada_elapsed_ms = (time.perf_counter() - t_ada) * 1000.0
        if idx >= warmup:
            adaface_lat_ms.append(ada_elapsed_ms)
        embeddings.append(emb.astype(np.float32))

    for idx, emb in enumerate(embeddings):
        t_enroll = time.perf_counter()
        helper_data, mask, key_hash = wifakey.enroll(emb)
        enroll_elapsed_ms = (time.perf_counter() - t_enroll) * 1000.0
        if idx >= warmup:
            enroll_lat_ms.append(enroll_elapsed_ms)

        t_verify = time.perf_counter()
        _ = wifakey.verify(emb, helper_data, mask, key_hash)
        verify_elapsed_ms = (time.perf_counter() - t_verify) * 1000.0
        if idx >= warmup:
            verify_lat_ms.append(verify_elapsed_ms)

    print("FaceProcessor status distribution:")
    for k in sorted(status_counts.keys()):
        print(f"  - {k}: {status_counts[k]}")
    print("-" * 70)

    print("Module latency stats (ms):")
    modules = [
        ("FaceProcessor.process", face_lat_ms),
        ("AdaFaceExtractor.get_feature_vector", adaface_lat_ms),
        ("WiFaKeyHandler.enroll", enroll_lat_ms),
        ("WiFaKeyHandler.verify", verify_lat_ms),
    ]

    for name, values in modules:
        s = summarize_ms(values)
        print(
            f"{name}\n"
            f"  count={s['count']} mean={s['mean_ms']} std={s['std_ms']} "
            f"min={s['min_ms']} p50={s['p50_ms']} p95={s['p95_ms']} p99={s['p99_ms']} max={s['max_ms']}"
        )
    print("-" * 70)
    print(f"Aligned faces used      : {len(aligned_faces)}")
    print(f"Embeddings extracted    : {len(embeddings)}")
    print("Done.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WiFaKey modules independently.")
    parser.add_argument("--images-dir", required=True, help="Directory containing input images.")
    parser.add_argument("--image-pattern", default="*.jpg", help="Glob pattern for images. Default: *.jpg")
    parser.add_argument("--max-images", type=int, default=0, help="Max images to process. 0 means all (full images).")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up samples ignored per module.")
    parser.add_argument("--ctx-id", type=int, default=0, help="FaceProcessor ctx_id. 0=GPU, -1=CPU.")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Face detection confidence threshold.")
    parser.add_argument("--adaface-device", default="cuda", help="AdaFace device: cuda or cpu.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        benchmark_modules(
            images_dir=args.images_dir,
            image_pattern=args.image_pattern,
            max_images=args.max_images,
            warmup=args.warmup,
            ctx_id=args.ctx_id,
            confidence_threshold=args.confidence_threshold,
            adaface_device=args.adaface_device,
        )
    )
