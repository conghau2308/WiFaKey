"""
check_arcface_embedding.py

Kiểm tra xem InsightFace buffalo_l có trả về embedding không.
"""

import cv2
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from vision_module.face_processor import FaceProcessor

# Khởi tạo FaceProcessor (đã có app)
fp = FaceProcessor(det_model="buffalo_l", ctx_id=0, confidence_threshold=0.7)

# Lấy một ảnh bất kỳ từ dataset
img_path = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "raw",
    "labeled_faces_in_the_wild",
    "lfw-deepfunneled",
    "Aaron_Peirsol",
    "Aaron_Peirsol_0001.jpg",
)
img = cv2.imread(img_path)

# Phân tích khuôn mặt
faces = fp.app.get(img)
if faces:
    print(f"Số khuôn mặt tìm thấy: {len(faces)}")
    face = faces[0]
    print("Các thuộc tính có sẵn:", dir(face))
    if hasattr(face, "embedding"):
        emb = face.embedding
        print(f"Shape embedding: {emb.shape}")
        print(f"Norm embedding: {np.linalg.norm(emb):.4f}")
    else:
        print("❌ Không có thuộc tính embedding!")
else:
    print("Không tìm thấy khuôn mặt.")
