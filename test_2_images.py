import os
import sys
import numpy as np
import cv2

# ================= IMPORT MODULE NỘI BỘ =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
sys.path.append(PROJECT_ROOT)

from feature_extractor.adaface_handler import AdaFaceExtractor
from vision_module.face_processor import FaceProcessor
from wifakey_module.wifakey_handler import WiFaKeyHandler

# ================= CẤU HÌNH =================
WIFAKEY_DATA = os.path.join(PROJECT_ROOT, "wifakey_module", "data")

def read_image(path):
    if not os.path.exists(path):
        return None
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return img if img is not None else None
    except:
        return None


def test_two_images(img1_path, img2_path):
    print("=" * 60)
    print("      🔍 TEST WIFAKEY – 2 ẢNH TÙY CHỌN")
    print("=" * 60)

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("❌ Lỗi: File ảnh không tồn tại.")
        return

    print("⏳ Đang load model & chức năng xử lý...")
    extractor = AdaFaceExtractor()
    face_processor = FaceProcessor()
    wifakey = WiFaKeyHandler(
        data_path=WIFAKEY_DATA,
        weights_path=os.path.join(WIFAKEY_DATA, "Weights_Var_MS"),
        biases_path=os.path.join(WIFAKEY_DATA, "Biases_Var_MS"),
    )
    print("✅ Module đã sẵn sàng")

    img1 = read_image(img1_path)
    img2 = read_image(img2_path)

    if img1 is None or img2 is None:
        print("❌ Không đọc được ảnh.")
        return

    print("⏳ Đang ALIGN khuôn mặt...")
    a1 = face_processor.process(img1)
    a2 = face_processor.process(img2)

    if a1 is None or a2 is None:
        print("❌ Không detect được khuôn mặt.")
        return

    print("⏳ Đang sinh embedding...")
    try:
        f1 = extractor.get_feature_vector(a1)
        f2 = extractor.get_feature_vector(a2)
        f1 = np.array(f1).astype(np.float32)
        f2 = np.array(f2).astype(np.float32)
    except Exception as e:
        print("❌ Lỗi trích xuất đặc trưng:", e)
        return

    print("✅ Embedding OK")
    print(" - Vector 1:", f1.shape)
    print(" - Vector 2:", f2.shape)

    print("\n🔐 ENROLL bằng ảnh 1...")
    helper_data, key_hash = wifakey.enroll(f1)

    print("\n🔑 VERIFY ảnh 2...")
    success = wifakey.verify(f2, helper_data, key_hash)

    print("\n" + "=" * 60)
    if success:
        print("🎉 MỞ KHÓA THÀNH CÔNG → 2 ẢNH CÙNG NGƯỜI!")
    else:
        print("🚫 MỞ KHÓA THẤT BẠI → 2 ẢNH KHÁC NGƯỜI!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách chạy:")
        print(" python test2.py <image1> <image2>")
        sys.exit(1)

    test_two_images(sys.argv[1], sys.argv[2])
