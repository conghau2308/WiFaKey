from pydantic import BaseModel, Field
from typing import List

class DecodeRequest(BaseModel):
    llr: List[float] = Field(..., description="832 giá trị LLR (số thực)")
    key_hash: str = Field(..., description="SHA256 hash của key, dạng hex string (64 ký tự)")

class DecodeResponse(BaseModel):
    success: bool
    message: str = ""