import numpy as np


def select_top_margin_indices(margin: np.ndarray, k: int = 832) -> np.ndarray:
    """Chọn k vị trí có margin lớn nhất."""
    idx = np.argpartition(-margin, k)[:k]
    idx.sort()
    return idx


def build_mask(indices: np.ndarray, total_len: int = 1536) -> np.ndarray:
    """Tạo mặt nạ nhị phân từ danh sách chỉ số."""
    mask = np.zeros(total_len, dtype=np.uint8)
    mask[indices] = 1
    return mask
