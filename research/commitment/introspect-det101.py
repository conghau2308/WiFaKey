import os
import json
import numpy as np
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


# ---- Bước 1: Tìm + introspect det_10g.onnx ----
def find_insightface_models():
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".insightface", "models", "buffalo_l"),
        os.path.join(home, "AppData", "Roaming", ".insightface", "models", "buffalo_l"),
        os.path.join(home, ".cache", "insightface", "models", "buffalo_l"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


buffalo_dir = find_insightface_models()
det_path = os.path.join(buffalo_dir, "det_10g.onnx")
print(f"det_10g.onnx tại: {det_path}\n")

session = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
print("=== det_10g.onnx INPUTS ===")
for i in session.get_inputs():
    print(f"  name={i.name}, shape={i.shape}, type={i.type}")
print("=== det_10g.onnx OUTPUTS (thứ tự quan trọng — giữ nguyên index) ===")
for idx, o in enumerate(session.get_outputs()):
    print(f"  [{idx}] name={o.name}, shape={o.shape}, type={o.type}")

# ---- Bước 2: Chạy FaceAnalysis THẬT trên 1 ảnh test, dump kết quả cuối ----
IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "labeled_faces_in_the_wild",
    "lfw-deepfunneled",
    "Aaron_Guiel",
    "Aaron_Guiel_0001.jpg",
)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {IMAGE_PATH}")

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

faces = app.get(img)
if not faces:
    raise RuntimeError("Không phát hiện được khuôn mặt trong ảnh này, chọn ảnh khác")

face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

result = {
    "image_path": IMAGE_PATH,
    "image_shape": list(img.shape),  # [H, W, C], BGR
    "bbox": face.bbox.tolist(),
    "det_score": float(face.det_score),
    "kps": face.kps.tolist(),
}

with open("face_detection_test_vector.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n=== KẾT QUẢ (đã lưu vào face_detection_test_vector.json) ===")
print(json.dumps(result, indent=2))

# ---- Bước 3: lưu ảnh raw dạng .bin để tôi đọc thẳng vào Rust được ----
img_bgr = np.ascontiguousarray(img)
img_bgr.tofile("face_detection_test_image.bin")
print(
    f"\nĐã lưu ảnh raw: face_detection_test_image.bin ({img_bgr.nbytes} bytes, shape={img_bgr.shape})"
)
