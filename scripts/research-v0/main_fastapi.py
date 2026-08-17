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

# =======================
# Internal Module Imports
# =======================
try:
    from vision_module.face_processor import FaceProcessor
    from feature_extractor.adaface_handler import AdaFaceExtractor
    from wifakey_module.wifakey_handler import WiFaKeyHandler
except ImportError as e:
    print(f"[CRITICAL] Import error: {e}. Check sys.path or project structure.")
    sys.exit(1)

# =======================
# Logging Setup
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WiFaKeyAPI")

# =======================
# Global Model Instances
# =======================
face_processor: FaceProcessor | None = None
adaface_extractor: AdaFaceExtractor | None = None
wifakey_handler: WiFaKeyHandler | None = None

# =======================
# FastAPI Lifespan
# =======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_processor, adaface_extractor, wifakey_handler

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info("🚀 Initializing WiFaKey System on GPU...")

    try:
        # 1. InsightFace - Detect + Align
        face_processor = FaceProcessor(
            det_model="buffalo_l",
            ctx_id=0,
            confidence_threshold=0.7
        )

        # 2. AdaFace - Feature Extraction
        adaface_extractor = AdaFaceExtractor(device="cuda")

        # 3. WiFaKey Handler (adaMTrans)
        wifakey_data_path = os.path.join(base_dir, "wifakey_module", "data")
        wifakey_handler = WiFaKeyHandler(
            data_path=wifakey_data_path,
            weights_path=os.path.join(wifakey_data_path, "Weights_Var_MS"),
            biases_path=os.path.join(wifakey_data_path, "Biases_Var_MS"),
        )

        logger.info("✅ All models loaded successfully on GPU.")

        # CUDA Warm-up
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
    logger.info("🛑 Server shutting down.")

# =======================
# FastAPI App
# =======================
app = FastAPI(
    title="WiFaKey Biometric Cryptosystem Service",
    lifespan=lifespan
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

# =======================
# Data Models
# =======================
class EnrollRequest(BaseModel):
    image: str  # Base64 encoded image

class EnrollResponse(BaseModel):
    helper_data_b64: str
    mask_b64: str        # New: Randomized mask indices
    key_hash_b64: str

class VerifyRequest(BaseModel):
    image: str
    helper_data_b64: str
    mask_b64: str        # New: Required for adaMTrans bit filtering
    key_hash_b64: str

class VerifyResponse(BaseModel):
    success: bool
    message: str = ""

# =======================
# Utility Functions
# =======================
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

def process_image_pipeline(image_b64: str) -> tuple[np.ndarray | None, str]:
    """Standard Pipeline: Decode -> Align -> Extract Embedding"""
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

# =======================
# API Endpoints
# =======================
@app.post("/enroll/{username}", response_model=EnrollResponse)
async def enroll_user(username: str, request: EnrollRequest):
    logger.info(f"REQ: Enroll user '{username}'")

    embedding, pipeline_status = process_image_pipeline(request.image)
    if embedding is None:
        message = PIPELINE_ERROR_MESSAGES.get(
            pipeline_status, "Face detection or extraction failed."
        )
        raise HTTPException(status_code=400, detail=message)

    try:
        # adaMTrans enrollment returns 3 components
        helper_data, mask, key_hash = wifakey_handler.enroll(embedding)

        return EnrollResponse(
            helper_data_b64=base64.b64encode(helper_data.tobytes()).decode("utf-8"),
            mask_b64=base64.b64encode(mask.tobytes()).decode("utf-8"),
            key_hash_b64=base64.b64encode(key_hash).decode("utf-8"),
        )
    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        raise HTTPException(status_code=500, detail="Internal cryptographic error.")

@app.post("/verify/{username}", response_model=VerifyResponse)
async def verify_user(username: str, request: VerifyRequest):
    logger.info(f"REQ: Verify user '{username}'")

    try:
        # Decoding components
        helper_data = np.frombuffer(base64.b64decode(request.helper_data_b64), dtype=np.uint8)
        mask_r = np.frombuffer(base64.b64decode(request.mask_b64), dtype=np.uint8)
        key_hash = base64.b64decode(request.key_hash_b64)

        if mask_r.shape[0] != wifakey_handler.full_binary_length:
            return VerifyResponse(success=False, message=f"Invalid mask length: {mask_r.shape[0]}, expected {wifakey_handler.full_binary_length}.")
            
    except Exception as e:
        return VerifyResponse(success=False, message=f"Data decoding error: {e}")

    embedding, pipeline_status = process_image_pipeline(request.image)
    if embedding is None:
        message = PIPELINE_ERROR_MESSAGES.get(
            pipeline_status, "Face detection or extraction failed."
        )
        return VerifyResponse(success=False, message=message)

    try:
        success = wifakey_handler.verify(embedding, helper_data, mask_r, key_hash)
        status_msg = (
            "Verification successful."
            if success
            else "Verification failed (face does not match enrolled template)."
        )
        return VerifyResponse(success=success, message=status_msg)

    except Exception as e:
        logger.error(f"Verification error: {e}")
        return VerifyResponse(success=False, message="Decryption process failed.")

# =======================
# Main Entry Point
# =======================
if __name__ == "__main__":
    logger.info("🚀 Launching WiFaKey Secure Server...")
    uvicorn.run(app, host="0.0.0.0", port=8002)