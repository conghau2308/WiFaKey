import numpy as np
import cv2
import onnxruntime as ort
import os
import sys

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

DET_SIZE = (640, 640)
session = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
in_name = session.get_inputs()[0].name
out_names = [o.name for o in session.get_outputs()]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "labeled_faces_in_the_wild",
    "lfw-deepfunneled",
    "Aaron_Guiel",
    "Aaron_Guiel_0001.jpg",
)

img = cv2.imread(IMAGE_PATH)  # dùng ĐÚNG ảnh đã test lần trước (Aaron_Guiel_0001.jpg)
ih, iw = img.shape[:2]
dw, dh = DET_SIZE
scale = min(dw / iw, dh / ih)
nw, nh = int(iw * scale), int(ih * scale)
resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
canvas = np.zeros((dh, dw, 3), dtype=np.uint8)
canvas[:nh, :nw] = resized

blob = (canvas.astype(np.float32) - 127.5) / 128.0
blob = np.expand_dims(blob.transpose(2, 0, 1), 0)

outputs = session.run(out_names, {in_name: blob})

# Lưu cả 9 tensor + thông tin scale để tôi dựng lại đúng bước decode
np.savez("det10g_raw_outputs.npz",
         scale=scale, nw=nw, nh=nh, iw=iw, ih=ih,
         out0=outputs[0], out1=outputs[1], out2=outputs[2],
         out3=outputs[3], out4=outputs[4], out5=outputs[5],
         out6=outputs[6], out7=outputs[7], out8=outputs[8])
print("Đã lưu det10g_raw_outputs.npz")
for i, o in enumerate(outputs):
    print(f"  out{i}: shape={o.shape}, min={o.min():.4f}, max={o.max():.4f}")