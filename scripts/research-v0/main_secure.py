"""
main_secure.py

Bản service song song với main.py gốc — KHÔNG sửa main.py. Chỉ khác ở chỗ
dùng SecureWiFaKeyHandler (selection/puncturing) thay vì WiFaKeyHandler (AND
mask cũ đã xác nhận có lỗ hổng lộ key qua Gaussian elimination).

Field "mask_b64" trong API cũ được đổi tên thành "selection_mask_b64" để
tránh nhầm lẫn ý nghĩa (chọn vị trí, không phải AND-che-về-0). Toàn bộ phần
FaceProcessor / AdaFaceExtractor / cấu trúc endpoint giữ nguyên logic như
main.py gốc.

Đặt file này ở project_root, cùng cấp với main.py gốc (không phải trong
research/ hay wifakey_module/, vì đây là service runnable, không phải thư
viện/module nghiên cứu):

    project_root/
        main.py            <- gốc, không đổi, port 8002
        main_secure.py      <- file này, port 8003
        wifakey_module/
        research/
            commitment/
                v1_selection_puncturing.py

Chạy độc lập với main.py (port khác) để so sánh/migrate dần:

    python main_secure.py   # mặc định port 8003, xem __main__ bên dưới
"""

import base64
import os
import sys
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from vision_module.face_processor import FaceProcessor
    from feature_extractor.adaface_handler import AdaFaceExtractor
    from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
except ImportError as e:
    print(f"[CRITICAL] Import error: {e}. Check sys.path or project structure.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("WiFaKeySecureAPI")

face_processor: FaceProcessor | None = None
adaface_extractor: AdaFaceExtractor | None = None
wifakey_handler: SecureWiFaKeyHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_processor, adaface_extractor, wifakey_handler

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info("🚀 Initializing WiFaKey SECURE System on GPU...")

    try:
        face_processor = FaceProcessor(
            det_model="buffalo_l",
            ctx_id=0,
            confidence_threshold=0.7,
        )
        adaface_extractor = AdaFaceExtractor(device="cuda")

        wifakey_data_path = os.path.join(base_dir, "wifakey_module", "data")
        wifakey_handler = SecureWiFaKeyHandler(
            data_path=wifakey_data_path,
            weights_path=os.path.join(wifakey_data_path, "Weights_Var_MS"),
            biases_path=os.path.join(wifakey_data_path, "Biases_Var_MS"),
        )

        logger.info("✅ All models loaded successfully on GPU (secure handler).")

        logger.info("🔥 Performing GPU warm-up...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = face_processor.process(dummy_img)
        dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
        _ = adaface_extractor.get_feature_vector(dummy_face)
        logger.info("✅ Warm-up complete.")

    except Exception as e:
        logger.error(f"❌ Initialization Failed: {e}", exc_info=True)
        raise e

    yield
    logger.info("🛑 Secure server shutting down.")


app = FastAPI(
    title="WiFaKey Secure Biometric Cryptosystem Service (selection/puncturing patch)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_exports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
if os.path.isdir(_exports_dir):
    app.mount("/models", StaticFiles(directory=_exports_dir), name="models")


class EnrollRequest(BaseModel):
    image: str


class EnrollResponse(BaseModel):
    helper_data_b64: str
    selection_mask_b64: str  # đổi tên từ mask_b64 để rõ ngữ nghĩa mới
    key_hash_b64: str


class VerifyRequest(BaseModel):
    image: str
    helper_data_b64: str
    selection_mask_b64: str
    key_hash_b64: str


class VerifyResponse(BaseModel):
    success: bool
    message: str = ""


PIPELINE_ERROR_MESSAGES = {
    "invalid_image_base64": "Invalid image data (base64 decode failed).",
    "no_image": "Input image is empty or unreadable.",
    "no_face": "No face detected in image.",
    "low_confidence": "Face detected with low confidence. Please retry with clearer face.",
    "spoof_detected": "Spoofing detected. Please use a live face in front of camera.",
    "no_landmarks": "Face landmarks not found. Please retry.",
    "feature_extraction_failed": "Feature extraction failed. Please retry with another image.",
}


def base64_to_image(base64_string: str) -> np.ndarray | None:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Base64 decoding failed: {e}")
        return None


def process_image_pipeline(image_b64: str):
    raw_image = base64_to_image(image_b64)
    if raw_image is None:
        return None, "invalid_image_base64"

    aligned_face, status = face_processor.process(raw_image)
    if aligned_face is None:
        logger.warning(f"Face pipeline failed at vision stage: {status}")
        return None, status

    try:
        embedding = adaface_extractor.get_feature_vector(aligned_face)
        return embedding, "ok"
    except Exception as e:
        logger.warning(f"AdaFace extraction failed: {e}")
        return None, "feature_extraction_failed"


@app.post("/enroll/{username}", response_model=EnrollResponse)
async def enroll_user(username: str, request: EnrollRequest):
    logger.info(f"REQ: Enroll user '{username}' (secure handler)")

    embedding, pipeline_status = process_image_pipeline(request.image)
    if embedding is None:
        message = PIPELINE_ERROR_MESSAGES.get(
            pipeline_status, "Face detection or extraction failed."
        )
        raise HTTPException(status_code=400, detail=message)

    try:
        helper_data, selection_mask, key_hash = wifakey_handler.enroll(embedding)

        return EnrollResponse(
            helper_data_b64=base64.b64encode(helper_data.tobytes()).decode("utf-8"),
            selection_mask_b64=base64.b64encode(selection_mask.tobytes()).decode(
                "utf-8"
            ),
            key_hash_b64=base64.b64encode(key_hash).decode("utf-8"),
        )
    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        raise HTTPException(status_code=500, detail="Internal cryptographic error.")


@app.post("/verify/{username}", response_model=VerifyResponse)
async def verify_user(username: str, request: VerifyRequest):
    logger.info(f"REQ: Verify user '{username}' (secure handler)")

    try:
        helper_data = np.frombuffer(
            base64.b64decode(request.helper_data_b64), dtype=np.uint8
        )
        selection_mask = np.frombuffer(
            base64.b64decode(request.selection_mask_b64), dtype=np.uint8
        )
        key_hash = base64.b64decode(request.key_hash_b64)

        if selection_mask.shape[0] != wifakey_handler.full_binary_length:
            return VerifyResponse(
                success=False,
                message=(
                    f"Invalid selection_mask length: {selection_mask.shape[0]}, "
                    f"expected {wifakey_handler.full_binary_length}."
                ),
            )

    except Exception as e:
        return VerifyResponse(success=False, message=f"Data decoding error: {e}")

    embedding, pipeline_status = process_image_pipeline(request.image)
    if embedding is None:
        message = PIPELINE_ERROR_MESSAGES.get(
            pipeline_status, "Face detection or extraction failed."
        )
        return VerifyResponse(success=False, message=message)

    try:
        success = wifakey_handler.verify(
            embedding, helper_data, selection_mask, key_hash
        )
        status_msg = (
            "Verification successful."
            if success
            else "Verification failed (face does not match enrolled template)."
        )
        return VerifyResponse(success=success, message=status_msg)

    except Exception as e:
        logger.error(f"Verification error: {e}")
        return VerifyResponse(success=False, message="Decryption process failed.")


if __name__ == "__main__":
    logger.info("🚀 Launching WiFaKey SECURE Server...")
    # Port khác 8002 (main.py gốc) để chạy song song, tiện so sánh/migrate.
    uvicorn.run(app, host="0.0.0.0", port=8003)
