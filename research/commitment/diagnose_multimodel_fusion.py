"""
diagnose_multimodel_fusion.py

Fusion đa thuật toán: dùng embedding ArcFace để điều chỉnh LLR của Empirical LLR.
"""

import argparse, csv, hashlib, os, sys, cv2, numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from vision_module.face_processor import FaceProcessor
from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR

N, m, Z = 52, 42, 16
RAW_IMG_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "labeled_faces_in_the_wild", "lfw-deepfunneled"
)
CACHE_DIR_ARC = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache_arcface",
)
os.makedirs(CACHE_DIR_ARC, exist_ok=True)


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def load_embedding_adaface(name, imagenum):
    path = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "embeddings_cache",
        f"{name}_{int(imagenum):04d}.npy",
    )
    return np.load(path)


def load_or_compute_arcface(face_processor, name, imagenum):
    """Lấy embedding ArcFace, dùng cache nếu có."""
    cache_path = os.path.join(CACHE_DIR_ARC, f"{name}_{int(imagenum):04d}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    img_path = os.path.join(RAW_IMG_DIR, name, f"{name}_{int(imagenum):04d}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = face_processor.app.get(img)
    if not faces:
        return None
    best_face = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    if hasattr(best_face, "embedding"):
        emb = best_face.embedding
        np.save(cache_path, emb)
        return emb
    return None


def load_pairs(pairs_csv, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, handler.N, handler.Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def run_diagnostic(
    handler, face_processor, pairs_csv, max_pairs, lookup_path, alpha, sim_threshold
):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    rows = load_pairs(pairs_csv, max_pairs)
    n_total = 0
    pass_baseline = 0
    pass_fusion = 0

    for row in rows:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        name_enroll, img_enroll = row["name_enroll"], int(row["imagenum_enroll"])
        name_verify, img_verify = row["name_verify"], int(row["imagenum_verify"])

        emb_enroll_ada = load_embedding_adaface(name_enroll, img_enroll)
        helper, sel_mask, key_hash = handler.enroll(emb_enroll_ada)

        # Baseline Empirical LLR
        emb_verify_ada = load_embedding_adaface(name_verify, img_verify)
        proj = np.dot(emb_verify_ada, handler.M_matrix)
        bits, margin = binarize_with_perbit_confidence(proj, handler.intervals)
        idx = np.where(sel_mask == 1)[0]
        noisy = np.logical_xor(bits[idx], helper).astype(np.uint8)
        margin_sel = margin[idx]
        llr_baseline = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        ok_baseline = _try_decode(handler, llr_baseline, key_hash)

        # Fusion
        emb_verify_arc = load_or_compute_arcface(
            face_processor, name_verify, img_verify
        )
        if emb_verify_arc is not None:
            # Chuẩn hóa cả hai embedding trước khi tính cosine similarity
            ada_norm = emb_verify_ada / np.linalg.norm(emb_verify_ada)
            arc_norm = emb_verify_arc / np.linalg.norm(emb_verify_arc)
            sim = np.dot(ada_norm, arc_norm)
            if sim > sim_threshold:
                scale = 1.0 + alpha * (sim - sim_threshold)
                llr_fusion = llr_baseline * scale
            else:
                llr_fusion = llr_baseline
        else:
            llr_fusion = llr_baseline

        ok_fusion = _try_decode(handler, llr_fusion, key_hash)

        if ok_baseline:
            pass_baseline += 1
        if ok_fusion:
            pass_fusion += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"Empirical LLR gốc     : {pass_baseline}/{n_total} ({100*pass_baseline/n_total:.2f}%)"
    )
    print(
        f"Empirical LLR + Fusion (alpha={alpha}, sim_thresh={sim_threshold}): {pass_fusion}/{n_total} ({100*pass_fusion/n_total:.2f}%)"
    )
    handler.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument(
        "--alpha", type=float, default=0.5, help="Mức độ tăng LLR khi đồng thuận"
    )
    ap.add_argument(
        "--sim-threshold",
        type=float,
        default=0.5,
        help="Ngưỡng cosine similarity để áp dụng fusion",
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = os.path.join(root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    pairs_dir = os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    pairs_csv = os.path.join(pairs_dir, "tune_genuine.csv")

    print("Khởi tạo FaceProcessor...")
    face_processor = FaceProcessor(
        det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7
    )

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    run_diagnostic(
        handler,
        face_processor,
        pairs_csv,
        args.max_pairs,
        args.lookup,
        args.alpha,
        args.sim_threshold,
    )


if __name__ == "__main__":
    main()
