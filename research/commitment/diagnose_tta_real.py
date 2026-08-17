"""
diagnose_tta_real.py (ĐÃ SỬA: xử lý khi TTA không tạo được embedding)

Test-Time Augmentation (TTA) thực sự trên ảnh gốc.
- Tạo K biến thể ảnh (flip, xoay nhẹ, brightness jitter).
- Trích xuất embedding cho từng biến thể, lấy trung bình (hypersphere mean).
- Cache embedding TTA để dùng lại.
- Nếu không có biến thể nào hợp lệ, fallback về embedding gốc (không TTA).
- So sánh GMR: Empirical LLR gốc vs Empirical LLR + TTA.

Cách dùng:
    python research/commitment/diagnose_tta_real.py \
        --lookup experiments/out_step3/reliability_lookup.npz \
        --K 5 --max-pairs 50
"""

import argparse, csv, hashlib, os, sys, cv2, numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from vision_module.face_processor import FaceProcessor
from feature_extractor.adaface_handler import AdaFaceExtractor
from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR

N, m, Z = 52, 42, 16
RAW_IMG_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "labeled_faces_in_the_wild", "lfw-deepfunneled"
)
CACHE_DIR_TTA = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache_tta",
)
os.makedirs(CACHE_DIR_TTA, exist_ok=True)


# --- Các hàm hỗ trợ ---
def load_embedding(name, imagenum):
    path = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "embeddings_cache",
        f"{name}_{int(imagenum):04d}.npy",
    )
    return np.load(path)


def get_tta_embedding(face_processor, adaface, name, imagenum, K):
    """
    Trả về embedding TTA (512,) cho ảnh.
    Nếu cache tồn tại, dùng cache.
    Nếu không, tạo K biến thể, trung bình các embedding hợp lệ.
    Nếu không có embedding nào hợp lệ, fallback về embedding gốc.
    """
    cache_file = os.path.join(CACHE_DIR_TTA, f"{name}_{int(imagenum):04d}_tta{K}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)

    # Đọc ảnh gốc
    img_path = os.path.join(RAW_IMG_DIR, name, f"{name}_{int(imagenum):04d}.jpg")
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

    valid_embeddings = []
    for k in range(K):
        # Tạo biến thể
        aug = raw_img.copy()
        # Flip ngang với xác suất 0.5
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)
        # Xoay nhẹ (-5 đến 5 độ)
        angle = np.random.uniform(-5, 5)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # Điều chỉnh độ sáng (0.9 - 1.1)
        brightness = np.random.uniform(0.9, 1.1)
        aug = np.clip(aug * brightness, 0, 255).astype(np.uint8)

        # Qua pipeline
        aligned_rgb, status = face_processor.process(aug)
        if aligned_rgb is not None:
            emb = adaface.get_feature_vector(aligned_rgb)
            valid_embeddings.append(emb)

    # Nếu có ít nhất 1 embedding hợp lệ, dùng chúng
    if len(valid_embeddings) > 0:
        emb_stack = np.stack(valid_embeddings, axis=0)
        norms = np.linalg.norm(emb_stack, axis=1, keepdims=True)
        emb_normed = emb_stack / norms
        mean_emb = np.mean(emb_normed, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        np.save(cache_file, mean_emb)
        return mean_emb
    else:
        # Fallback: dùng embedding gốc (cache gốc)
        print(
            f"  [WARN] Không có biến thể nào hợp lệ cho {name}_{imagenum}, dùng embedding gốc."
        )
        orig_emb = load_embedding(name, imagenum)
        # Cũng cache lại để lần sau không phải fallback nữa
        np.save(cache_file, orig_emb)
        return orig_emb


def load_pairs(pairs_csv, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


# --- Hàm giải mã ---
def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, handler.N, handler.Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def run_diagnostic(
    handler, face_processor, adaface, pairs_csv, max_pairs, lookup_path, K
):
    emp_mod = EmpiricalLLR(lookup_path=lookup_path)
    rows = load_pairs(pairs_csv, max_pairs)
    n_total = 0
    pass_baseline = 0
    pass_tta = 0

    for row in rows:
        if max_pairs is not None and n_total >= max_pairs:
            break
        n_total += 1

        name_enroll, img_enroll = row["name_enroll"], int(row["imagenum_enroll"])
        name_verify, img_verify = row["name_verify"], int(row["imagenum_verify"])

        emb_enroll = load_embedding(name_enroll, img_enroll)
        helper, sel_mask, key_hash = handler.enroll(emb_enroll)

        # --- Baseline: Empirical LLR với embedding gốc ---
        emb_verify_orig = load_embedding(name_verify, img_verify)
        proj_orig = np.dot(emb_verify_orig, handler.M_matrix)
        bits_orig, margin_orig = binarize_with_perbit_confidence(
            proj_orig, handler.intervals
        )
        idx = np.where(sel_mask == 1)[0]
        noisy_orig = np.logical_xor(bits_orig[idx], helper).astype(np.uint8)
        margin_sel_orig = margin_orig[idx]
        llr_orig = emp_mod.modulate(
            noisy_orig, context={"margin": margin_sel_orig}
        ).flatten()
        ok_baseline = _try_decode(handler, llr_orig, key_hash)

        # --- TTA ---
        emb_verify_tta = get_tta_embedding(
            face_processor, adaface, name_verify, img_verify, K
        )
        proj_tta = np.dot(emb_verify_tta, handler.M_matrix)
        bits_tta, margin_tta = binarize_with_perbit_confidence(
            proj_tta, handler.intervals
        )
        noisy_tta = np.logical_xor(bits_tta[idx], helper).astype(np.uint8)
        margin_sel_tta = margin_tta[idx]
        llr_tta = emp_mod.modulate(
            noisy_tta, context={"margin": margin_sel_tta}
        ).flatten()
        ok_tta = _try_decode(handler, llr_tta, key_hash)

        if ok_baseline:
            pass_baseline += 1
        if ok_tta:
            pass_tta += 1

    print(f"Tổng cặp test: {n_total}")
    print(
        f"Empirical LLR (gốc)     : {pass_baseline}/{n_total} ({100*pass_baseline/n_total:.2f}%)"
    )
    print(
        f"Empirical LLR + TTA (K={K}): {pass_tta}/{n_total} ({100*pass_tta/n_total:.2f}%)"
    )
    handler.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--K", type=int, default=5, help="Số biến thể TTA cho mỗi ảnh")
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

    # Khởi tạo face processor và model AdaFace
    print("Khởi tạo FaceProcessor + AdaFaceExtractor...")
    face_processor = FaceProcessor(
        det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7
    )
    adaface = AdaFaceExtractor(device="cuda")

    # Khởi tạo handler WiFaKey
    handler = SecureWiFaKeyHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    run_diagnostic(
        handler, face_processor, adaface, pairs_csv, args.max_pairs, args.lookup, args.K
    )


if __name__ == "__main__":
    main()
