"""
extract_generator.py

Trích xuất ma trận sinh G (160 x 832) từ LDPC encoder.
"""

import numpy as np
import os, sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_lib import Encode

N, m, Z = 52, 42, 16
encoder = Encode.Proto_LDPC(N, m, Z)

# Lấy ma trận sinh
# Proto_LDPC thường lưu ma trận sinh dưới dạng systematic: [I | P]
# Ta sẽ sinh ra tất cả các vector đơn vị, encode từng cái, rồi ghép lại thành G
k = (N - m) * Z  # 160
n = N * Z  # 832

G = np.zeros((k, n), dtype=np.uint8)
for i in range(k):
    key = np.zeros((1, k), dtype=np.uint8)
    key[0, i] = 1
    codeword = encoder.encode_LDPC(key).flatten()
    G[i] = codeword

print(f"Kích thước ma trận sinh G: {G.shape}")

# Lưu ra file
output_path = os.path.join(os.path.dirname(__file__), "generator_matrix_G.npy")
np.save(output_path, G)
print(f"Đã lưu G vào {output_path}")

# In vài dòng đầu để kiểm tra
print("5 dòng đầu của G:")
print(G[:5, :10])
