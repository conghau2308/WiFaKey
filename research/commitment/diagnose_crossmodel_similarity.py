"""
diagnose_crossmodel_similarity.py

Đo phân phối cosine similarity giữa embedding AdaFace và ArcFace trên tập LFW.
"""

import os, sys, cv2, numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from vision_module.face_processor import FaceProcessor

RAW_IMG_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "raw", "labeled_faces_in_the_wild", "lfw-deepfunneled"
)
CACHE_ADA = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)


def load_adaface(name, imagenum):
    return np.load(os.path.join(CACHE_ADA, f"{name}_{int(imagenum):04d}.npy"))


def get_arcface(fp, name, imagenum):
    img_path = os.path.join(RAW_IMG_DIR, name, f"{name}_{int(imagenum):04d}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = fp.app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return best.embedding if hasattr(best, "embedding") else None


# Lấy danh sách ảnh từ tune_genuine.csv
import csv

pairs_csv = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "pairs",
    "tune_genuine.csv",
)
all_images = set()
with open(pairs_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_images.add((row["name_enroll"], int(row["imagenum_enroll"])))
        all_images.add((row["name_verify"], int(row["imagenum_verify"])))

print(f"Tổng số ảnh duy nhất: {len(all_images)}")

fp = FaceProcessor(det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7)
similarities = []
for name, imagenum in all_images:
    ada = load_adaface(name, imagenum)
    arc = get_arcface(fp, name, imagenum)
    if arc is not None:
        ada_n = ada / np.linalg.norm(ada)
        arc_n = arc / np.linalg.norm(arc)
        sim = np.dot(ada_n, arc_n)
        similarities.append(sim)

similarities = np.array(similarities)
print(f"Số ảnh tính được similarity: {len(similarities)}")
print(f"Min: {similarities.min():.4f}, Max: {similarities.max():.4f}")
print(f"Mean: {similarities.mean():.4f}, Median: {np.median(similarities):.4f}")
print(
    f"Phân vị 1%: {np.percentile(similarities, 1):.4f}, 5%: {np.percentile(similarities, 5):.4f}"
)
print(
    f"Phân vị 10%: {np.percentile(similarities, 10):.4f}, 25%: {np.percentile(similarities, 25):.4f}"
)
