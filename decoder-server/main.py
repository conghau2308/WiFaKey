"""
WiFaKey Decode Service

Nhận LLR (832 float) từ server Java, chạy Neural‑MS decoder (ONNX),
so sánh SHA256(k_prime) với key_hash đã lưu, trả về success.
Server này không bao giờ thấy dữ liệu sinh trắc thô, chỉ thấy LLR đã biến đổi.
"""

import hashlib
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="WiFaKey Decode Service")

# ── Cấu hình model ──────────────────────────────────────────────
MODEL_PATH = os.environ.get("WIFAIKEY_ONNX_MODEL", "neural_ms.onnx")
PROVIDERS = ["CPUExecutionProvider"]  # có thể thêm CUDA nếu server có GPU

# Kiểm tra file model tồn tại
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Không tìm thấy model ONNX tại {MODEL_PATH}")

# Load ONNX session một lần duy nhất
session = ort.InferenceSession(MODEL_PATH, providers=PROVIDERS)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ── Models cho request/response ─────────────────────────────────
class VerifyRequest(BaseModel):
    llr: list[float]        # 832 phần tử
    key_hash: str           # hex string 64 ký tự

class VerifyResponse(BaseModel):
    success: bool

class DecodeRequest(BaseModel):
    llr: list[float]

class DecodeResponse(BaseModel):
    key_hex: str
    key_hash: str

# ── Helper functions ────────────────────────────────────────────
def _decode(llr_list: list[float]) -> bytes:
    """Chạy ONNX decoder, trả về 20 bytes key."""
    if len(llr_list) != 832:
        raise HTTPException(status_code=400, detail="llr phải có đúng 832 phần tử")

    llr = np.array(llr_list, dtype=np.float32).reshape(1, 52, 16)
    output = session.run([output_name], {input_name: llr})[0]  # shape (1, 832)
    bits = (output.flatten() > 0).astype(np.uint8)[:160]
    key_bytes = np.packbits(bits).tobytes()
    return key_bytes

# ── API endpoints ───────────────────────────────────────────────
@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest):
    """Nhận LLR và key_hash, tự decode và so sánh."""
    key_bytes = _decode(request.llr)
    computed_hash = hashlib.sha256(key_bytes).hexdigest()
    expected_hash = request.key_hash.lower()
    if computed_hash == expected_hash:
        return {"success": True}
    return {"success": False}

@app.post("/decode", response_model=DecodeResponse)
async def decode(request: DecodeRequest):
    """Chỉ decode, trả về key dạng hex. Dành cho server Java nếu muốn tự so sánh."""
    key_bytes = _decode(request.llr)
    return {
        "key_hex": key_bytes.hex(),
        "key_hash": hashlib.sha256(key_bytes).hexdigest()
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}