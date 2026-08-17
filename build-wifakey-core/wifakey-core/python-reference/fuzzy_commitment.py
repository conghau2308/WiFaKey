import numpy as np
import hashlib


def generate_random_key() -> bytes:
    """Sinh khóa ngẫu nhiên 160-bit (20 bytes)."""
    return np.random.bytes(20)  # 20 bytes = 160 bits


def key_hash(k: bytes) -> bytes:
    """SHA256 của khóa."""
    return hashlib.sha256(k).digest()


def xor_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR hai mảng bit (cùng độ dài)."""
    return np.logical_xor(a, b).astype(np.uint8)
