"""
generate_margin_training_data.py

Tạo bộ dữ liệu huấn luyện cho Neural-MS decoder từ pipeline Margin Selection.
Dùng MarginSelectionHandler và EmpiricalLLR.
Lưu các cặp (LLR đầu vào, Target Codeword) dưới dạng file .npy.
"""

import os, sys, csv, numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.diagnose_margin_selection_v2 import MarginSelectionHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "margin_training_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_embedding(name, imagenum):
    return np.load(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def load_pairs(csv_path, max_pairs=None):
    pairs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                emb1 = load_embedding(row["name_enroll"], int(row["imagenum_enroll"]))
                emb2 = load_embedding(row["name_verify"], int(row["imagenum_verify"]))
                pairs.append((emb1, emb2))
            except FileNotFoundError:
                pass
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def main():
    handler = MarginSelectionHandler(data_path=DATA_DIR)
    encoder = Encode.Proto_LDPC(N, m, Z)
    emp_llr = EmpiricalLLR(lookup_path=LOOKUP_PATH, masked_mag=1.25)

    # Load dữ liệu từ tune + select (nếu có)
    pairs = []
    for tier in ["tune", "select"]:
        csv_path = os.path.join(PAIRS_DIR, f"{tier}_genuine.csv")
        if os.path.exists(csv_path):
            tier_pairs = load_pairs(csv_path)
            pairs.extend(tier_pairs)
            print(f"Đã load {len(tier_pairs)} cặp từ {tier}")

    print(f"Tổng số cặp huấn luyện: {len(pairs)}")

    all_llr = []
    all_target = []
    rng = np.random.default_rng(0)

    for i, (emb_enroll, emb_verify) in enumerate(pairs):
        # Enrollment với margin selection
        b_full_e = handler._binarize_full(emb_enroll).astype(np.uint8)
        projected_e = np.dot(emb_enroll, handler.M_matrix)
        _, margin_e = binarize_with_perbit_confidence(projected_e, handler.intervals)
        selection_indices = np.argpartition(-margin_e, handler.feature_length)[
            : handler.feature_length
        ]
        selection_indices.sort()
        b_selected_e = b_full_e[selection_indices]

        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected_e, codeword).astype(np.uint8)

        # Verify
        b_full_v = handler._binarize_full(emb_verify).astype(np.uint8)
        b_selected_v = b_full_v[selection_indices]
        noisy = np.logical_xor(b_selected_v, helper_data).astype(np.uint8)

        projected_v = np.dot(emb_verify, handler.M_matrix)
        _, margin_v = binarize_with_perbit_confidence(projected_v, handler.intervals)
        margin_sel = margin_v[selection_indices]

        llr = emp_llr.modulate(noisy, context={"margin": margin_sel}).reshape(N, Z)
        target_bipolar = codeword.astype(np.float32) * 2.0 - 1.0

        all_llr.append(llr)
        all_target.append(target_bipolar)

        if (i + 1) % 200 == 0:
            print(f"Đã xử lý {i+1}/{len(pairs)} cặp")

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
