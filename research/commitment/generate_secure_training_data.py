"""
generate_secure_training_data.py

Tạo bộ dữ liệu huấn luyện cho Neural-MS decoder từ pipeline AN TOÀN.
Sử dụng SecureWiFaKeyHandler và EmpiricalLLR.
Lưu các cặp (LLR đầu vào, Target Codeword) dưới dạng file .npy.
"""

import os
import sys
import csv
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR
from wifakey_module.wifakey_lib.utils import lssc_binary
from wifakey_module.wifakey_lib import Encode

N, m, Z = 52, 42, 16
code_k = N - m
KEY_LENGTH = code_k * Z

DATA_DIR = os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)
PAIRS_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
)
LOOKUP_PATH = os.path.join(
    _PROJECT_ROOT, "experiments", "out_step3", "reliability_lookup.npz"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "secure_training_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_embedding(name, imagenum):
    return np.load(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def main():
    handler = SecureWiFaKeyHandler(data_path=DATA_DIR)
    encoder = Encode.Proto_LDPC(N, m, Z)
    emp_llr = EmpiricalLLR(lookup_path=LOOKUP_PATH, masked_mag=1.25)

    pairs = []
    with open(os.path.join(PAIRS_DIR, "tune_genuine.csv"), newline="") as f:
        for row in csv.DictReader(f):
            try:
                emb1 = load_embedding(row["name_enroll"], row["imagenum_enroll"])
                emb2 = load_embedding(row["name_verify"], row["imagenum_verify"])
                pairs.append((emb1, emb2))
            except FileNotFoundError:
                pass

    print(f"Tìm thấy {len(pairs)} cặp. Bắt đầu sinh dữ liệu...")

    all_llr = []
    all_target = []
    counter = 0
    rng = np.random.default_rng(0)  # Cố định để tái lập

    for emb_enroll, emb_verify in pairs:
        # 1. Enrollment an toàn
        # Để lấy được target codeword, ta cần tự thực hiện logic enroll mà không qua hàm enroll có sẵn,
        # hoặc sửa tạm handler. Ở đây ta sẽ mô phỏng chính xác logic trong SecureWiFaKeyHandler.
        b_full_e = handler._binarize_full(emb_enroll).astype(np.uint8)
        # Chọn ngẫu nhiên 832 vị trí
        selection_indices = rng.choice(
            len(b_full_e), size=handler.feature_length, replace=False
        )
        selection_indices.sort()
        selection_mask = np.zeros(len(b_full_e), dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected_enroll = b_full_e[selection_indices]

        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected_enroll, codeword).astype(np.uint8)

        # 2. Tính LLR từ ảnh verify
        # Cần lấy margin cho các bit ĐÃ CHỌN
        projected = np.dot(emb_verify, handler.M_matrix)
        b_full_v_bits, margin_v = binarize_with_perbit_confidence(
            projected, handler.intervals
        )
        b_selected_verify = b_full_v_bits[selection_indices]
        margin_selected = margin_v[selection_indices]

        noisy_bits = np.logical_xor(b_selected_verify, helper_data).astype(np.uint8)
        # Empirical LLR
        llr = emp_llr.modulate(noisy_bits, context={"margin": margin_selected}).reshape(
            N, Z
        )
        # Target
        target_bipolar = codeword.astype(np.float32) * 2.0 - 1.0

        all_llr.append(llr)
        all_target.append(target_bipolar)

        counter += 1
        if counter % 200 == 0:
            print(f"Đã xử lý {counter}/{len(pairs)} cặp")

    # Lưu dữ liệu
    np.save(
        os.path.join(OUTPUT_DIR, "train_llr.npy"), np.array(all_llr, dtype=np.float32)
    )
    np.save(
        os.path.join(OUTPUT_DIR, "train_target.npy"),
        np.array(all_target, dtype=np.float32),
    )
    print(f"Hoàn tất. Dữ liệu đã được lưu vào {OUTPUT_DIR}")
    handler.sess.close()


if __name__ == "__main__":
    main()
