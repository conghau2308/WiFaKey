import numpy as np
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)


def binarize(feature_vector: np.ndarray, M_matrix: np.ndarray, intervals: np.ndarray):
    """
    feature_vector: (512,) float32 – embedding đã qua BioHashing
    M_matrix: (512, 512) – ma trận gốc WiFaKey
    intervals: (3,) – các ngưỡng thermometer code
    Trả về:
        bits: (1536,) uint8
        margin: (1536,) float32
    """
    projected = np.dot(feature_vector, M_matrix)
    bits, margin = binarize_with_perbit_confidence(projected, intervals)
    return bits.astype(np.uint8), margin.astype(np.float32)
