import numpy as np
import hashlib
import hmac


def _hmac_permutation(salt_bytes: bytes, n_bits: int) -> np.ndarray:
    """Tạo hoán vị bằng cách gán mỗi vị trí một giá trị ngẫu nhiên từ HMAC, rồi sắp xếp."""
    values = np.zeros(n_bits)
    for i in range(n_bits):
        msg = i.to_bytes(4, "big")
        digest = hmac.new(salt_bytes, msg, hashlib.sha256).digest()
        values[i] = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return np.argsort(values)


def generate_permutation(salt_bytes: bytes, n_bits: int = 1536) -> np.ndarray:
    return _hmac_permutation(salt_bytes, n_bits)


def apply_permutation(bits: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return bits[perm]
