import numpy as np
import hashlib
import hmac


def _hmac_gaussian(secret: bytes, row: int, col: int) -> float:
    """Tạo một số Gaussian từ HMAC(secret, row || col)."""
    msg = row.to_bytes(4, "big") + col.to_bytes(4, "big")
    digest = hmac.new(secret, msg, hashlib.sha256).digest()
    # Chuyển 4 byte đầu thành số nguyên, rồi dùng Box‑Muller
    u = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    v = int.from_bytes(digest[4:8], "big") / 0xFFFFFFFF
    r = (-2.0 * np.log(u + 1e-15)) ** 0.5
    theta = 2.0 * np.pi * v
    return r * np.cos(theta)  # dùng 1 trong 2 giá trị độc lập


def generate_projection_matrix(user_secret: bytes, dim: int = 512) -> np.ndarray:
    M = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            # Mỗi cặp (i,j) dùng một index để tạo giá trị Gaussian
            idx = i * dim + j
            # Để tạo 2 giá trị độc lập từ một lần gọi, ta tách chẵn/lẻ
            val = _hmac_gaussian(user_secret, idx, 0)
            M[i, j] = val
        # Chuẩn hoá hàng
        norm = np.linalg.norm(M[i, :])
        if norm > 0:
            M[i, :] /= norm
    return M.astype(np.float32)


def biohash_project(embedding: np.ndarray, user_secret: bytes) -> np.ndarray:
    M = generate_projection_matrix(user_secret, len(embedding))
    v_proj = np.dot(embedding, M.T)
    norm = np.linalg.norm(v_proj)
    if norm > 0:
        v_proj /= norm
    return v_proj.astype(np.float32)
