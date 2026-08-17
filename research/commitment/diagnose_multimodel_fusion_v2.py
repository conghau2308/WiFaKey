"""
diagnose_multimodel_fusion_v2.py (BẢN QUÉT NHIỀU DELTA)

Quét qua danh sách delta_disagree_values và so sánh GMR.
"""

import argparse, csv, gc, hashlib, os, sys, cv2, numpy as np
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


def load_pairs(pairs_csv, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def cache_arcface_embeddings(pairs_csv, max_pairs):
    print("Pre-caching ArcFace embeddings...")
    fp = FaceProcessor(det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7)
    rows = load_pairs(pairs_csv, max_pairs)
    all_images = set()
    for row in rows:
        all_images.add((row["name_verify"], int(row["imagenum_verify"])))
    for name, imagenum in all_images:
        cache_path = os.path.join(CACHE_DIR_ARC, f"{name}_{int(imagenum):04d}.npy")
        if os.path.exists(cache_path):
            continue
        img_path = os.path.join(RAW_IMG_DIR, name, f"{name}_{int(imagenum):04d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = fp.app.get(img)
        if faces:
            best = max(
                faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            if hasattr(best, "embedding"):
                np.save(cache_path, best.embedding)
    return fp


def load_arcface_cached(name, imagenum):
    cache_path = os.path.join(CACHE_DIR_ARC, f"{name}_{int(imagenum):04d}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None


def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, handler.N, handler.Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def run_diagnostic(
    handler, pairs_csv, max_pairs, lookup_path, delta_agree, delta_disagree_list
):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    rows = load_pairs(pairs_csv, max_pairs)
    n_total = len(rows)

    # Lưu trữ kết quả baseline và các biến cần cho fusion
    baselines = []
    fusion_data = []  # mỗi phần tử: (agreement_array, llr_baseline)

    print("Đánh giá baseline...")
    for row in rows:
        name_verify, img_verify = row["name_verify"], int(row["imagenum_verify"])
        emb_enroll_ada = load_embedding_adaface(
            row["name_enroll"], int(row["imagenum_enroll"])
        )
        helper, sel_mask, key_hash = handler.enroll(emb_enroll_ada)
        emb_verify_ada = load_embedding_adaface(name_verify, img_verify)
        proj_ada = np.dot(emb_verify_ada, handler.M_matrix)
        bits_ada, margin_ada = binarize_with_perbit_confidence(
            proj_ada, handler.intervals
        )
        idx = np.where(sel_mask == 1)[0]
        noisy = np.logical_xor(bits_ada[idx], helper).astype(np.uint8)
        llr_baseline = emp_mod.modulate(
            noisy, context={"margin": margin_ada[idx]}
        ).flatten()
        ok = _try_decode(handler, llr_baseline, key_hash)

        baselines.append(ok)

        # Chuẩn bị dữ liệu fusion
        emb_verify_arc = load_arcface_cached(name_verify, img_verify)
        if emb_verify_arc is not None:
            proj_arc = np.dot(emb_verify_arc, handler.M_matrix)
            bits_arc, _ = binarize_with_perbit_confidence(proj_arc, handler.intervals)
            bits_arc_sel = bits_arc[idx]
            agreement = (bits_ada[idx] == bits_arc_sel).astype(np.float32)
        else:
            agreement = np.ones_like(
                llr_baseline
            )  # nếu không có arcface, coi như luôn đồng thuận
        fusion_data.append((agreement, llr_baseline, key_hash))

    baseline_gmr = sum(baselines) / n_total * 100
    print(f"Baseline GMR: {baseline_gmr:.2f}% ({sum(baselines)}/{n_total})")

    print("\nQuét các giá trị delta_disagree:")
    print(f"{'Delta_disagree':>15s} {'GMR':>10s} {'Cải thiện':>12s}")
    print("-" * 40)
    for delta_disagree in delta_disagree_list:
        pass_fusion = 0
        for (agreement, llr_baseline, key_hash), ok_base in zip(fusion_data, baselines):
            scale = np.ones_like(llr_baseline)
            scale -= delta_disagree * (1 - agreement)  # giảm LLR cho bit bất đồng
            llr_fusion = llr_baseline * scale
            if _try_decode(handler, llr_fusion, key_hash):
                pass_fusion += 1
        gmr = pass_fusion / n_total * 100
        improvement = gmr - baseline_gmr
        print(f"{delta_disagree:15.2f} {gmr:10.2f}% {improvement:+12.2f} điểm %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--delta-agree", type=float, default=0.0)
    ap.add_argument(
        "--delta-disagree-list", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2]
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

    # Cache ArcFace
    fp = cache_arcface_embeddings(pairs_csv, args.max_pairs)
    del fp
    gc.collect()

    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    run_diagnostic(
        handler,
        pairs_csv,
        args.max_pairs,
        args.lookup,
        args.delta_agree,
        args.delta_disagree_list,
    )


if __name__ == "__main__":
    main()
