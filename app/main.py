import logging
from fastapi import FastAPI, HTTPException
from .models import DecodeRequest, DecodeResponse
from .decoder import DecoderService
import hashlib
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo decoder service (chỉ một lần)
decoder_service = DecoderService()

app = FastAPI(title="WiFaKey Decode Service", version="1.0")

@app.post("/decode", response_model=DecodeResponse)
async def decode(request: DecodeRequest):
    try:
        # 1. Giải mã LLR thành key bits
        key_bits = decoder_service.decode(request.llr)  # numpy array (160,) uint8

        # 2. Tính SHA256 của key bits (dạng bytes)
        key_bytes = key_bits.tobytes()  # 20 bytes (160 bits)
        computed_hash = hashlib.sha256(key_bytes).hexdigest()

        # 3. So sánh với key_hash gửi lên
        success = computed_hash == request.key_hash.lower()
        logger.info(f"Decode result: {success} (computed: {computed_hash[:8]}..., provided: {request.key_hash[:8]}...)")

        return DecodeResponse(success=success, message="OK" if success else "Hash mismatch")
    except Exception as e:
        logger.exception("Decode error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

# Shutdown hook để giải phóng session
@app.on_event("shutdown")
def shutdown_event():
    decoder_service.close()