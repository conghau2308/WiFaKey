"""
07_selfmatch_and_genuine_real_embeddings.py

Test này KHÔNG dùng bất kỳ file nào trong research/ (verify_variant.py,
NeuralMSOriginal, binarize_with_confidence...) - chỉ gọi THẲNG
handler.enroll() và handler.verify() 100% NGUYÊN BẢN từ
wifakey_module/wifakey_handler.py, trên embedding THẬT đã cache từ LFW.

Mục đích: tách bạch dứt điểm 2 khả năng còn lại sau khi đã sửa lỗi tràn
số uint8:
  (a) Vẫn còn bug trong research/pipeline/verify_variant.py (harness của
      tôi viết) - nếu vậy, test này (bỏ qua hoàn toàn harness) sẽ PASS.
  (b) BER thật giữa các cặp ảnh LFW (do κ/M_matrix chưa phù hợp với
      AdaFace+LFW) vượt quá khả năng sửa lỗi của LDPC - nếu vậy, ngay cả
      test này (dùng code gốc 100%) cũng sẽ FAIL.

Gồm 2 phần:
  TEST SELF-MATCH: verify bằng CHÍNH embedding vừa enroll (không nhiễu
      biometric, chỉ có nhiễu do mask) - PHẢI thành công gần 100%.
  TEST GENUINE THẬT: verify bằng ảnh KHÁC của cùng người (từ
      data/processed/pairs/select_genuine.csv) - đo tỷ lệ thành công thật.

Cách chạy:
    python scripts/research/07_selfmatch_and_genuine_real_embeddings.py
"""

import os
import sys
import csv
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler

CACHE_DIR = os.path.join(
    _PROJECT_ROOT,
    "datasets",
    "processed",
    "labeled_faces_in_the_wild",
    "embeddings_cache",
)
PAIRS_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "labeled_faces_in_the_wild", "processed", "pairs"
)


def load_embedding(name, imagenum):
    path = os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy")
    return np.load(path)


def load_some_embeddings(n=20):
    """Lấy ngẫu nhiên n embedding bất kỳ từ cache để test self-match."""
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    if len(files) == 0:
        raise FileNotFoundError(
            f"Không có embedding nào trong {CACHE_DIR}. "
            f"Chạy scripts/01_extract_embeddings.py trước."
        )
    np.random.shuffle(files)
    return [np.load(os.path.join(CACHE_DIR, f)) for f in files[:n]]


def load_genuine_pairs(n=None):
    path = os.path.join(PAIRS_DIR, "select_genuine.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Chạy scripts/02_build_pairs_dataset.py trước."
        )
    pairs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emb1 = load_embedding(row["name_enroll"], row["imagenum_enroll"])
            emb2 = load_embedding(row["name_verify"], row["imagenum_verify"])
            pairs.append((emb1, emb2))
    if n is not None:
        pairs = pairs[:n]
    return pairs


def test_self_match(handler, embeddings, label="SELF-MATCH"):
    print(
        f"=== TEST {label} (verify bằng CHÍNH embedding vừa enroll, code 100% gốc) ==="
    )
    n_ok = 0
    for emb in embeddings:
        helper_data, mask_r, key_hash = handler.enroll(emb)
        success = handler.verify(emb, helper_data, mask_r, key_hash)
        n_ok += int(success)
    rate = n_ok / len(embeddings)
    print(f"  Thành công: {n_ok}/{len(embeddings)} ({rate:.1%})")
    return rate


def test_genuine_real(handler, pairs):
    print(
        "\n=== TEST GENUINE THẬT (ảnh KHÁC của cùng người, từ LFW, code 100% gốc) ==="
    )
    n_ok = 0
    ber_list = []
    for emb_enroll, emb_verify in pairs:
        helper_data, mask_r, key_hash = handler.enroll(emb_enroll)
        success = handler.verify(emb_verify, helper_data, mask_r, key_hash)
        n_ok += int(success)

        # đo thêm BER thô (không qua decode) để so sánh
        b_full_e = handler._binarize_full(emb_enroll).astype(np.uint8)
        b_full_v = handler._binarize_full(emb_verify).astype(np.uint8)
        b_masked_e = (b_full_e & mask_r)[: handler.feature_length]
        b_masked_v = (b_full_v & mask_r)[: handler.feature_length]
        ber_list.append(np.mean(b_masked_e != b_masked_v))

    rate = n_ok / len(pairs)
    ber_arr = np.array(ber_list)
    print(f"  Thành công (decode đúng key): {n_ok}/{len(pairs)} ({rate:.1%})")
    print(f"  BER trung bình (trước decode): {ber_arr.mean():.4f}")
    print(f"  BER median: {np.median(ber_arr):.4f}")
    print(
        f"  % cặp có BER <= 0.176 (ngưỡng tham khảo): {np.mean(ber_arr <= 0.176):.1%}"
    )
    return rate, ber_arr


def main():
    handler = WiFaKeyHandler()

    embeddings_sample = load_some_embeddings(n=20)
    rate_self = test_self_match(handler, embeddings_sample)

    if rate_self < 0.9:
        print(
            "\n❌ Self-match vẫn fail dù dùng code 100% gốc, không qua research/ harness."
        )
        print("   => Vẫn còn vấn đề trong CHÍNH wifakey_handler.py hoặc dữ liệu")
        print("      (M_matrix.npy/binarization_intervals.npy) - cần điều tra tiếp,")
        print("      không phải do research/pipeline/verify_variant.py nữa.")
        return

    print("\n✅ Self-match PASS -> pipeline gốc hoạt động đúng trên embedding thật.")
    print("   Nếu run_ab_soft_llr.py (dùng research/ harness) vẫn FRR=1.0, bug nằm")
    print(
        "   trong research/pipeline/verify_variant.py hoặc research/quantizer/, KHÔNG"
    )
    print("   phải trong wifakey_handler.py hay trong dữ liệu.\n")

    genuine_pairs = load_genuine_pairs(n=200)
    rate_genuine, bers = test_genuine_real(handler, genuine_pairs)

    if rate_genuine < 0.5:
        print("\n⚠️  Tỷ lệ thành công trên cặp genuine THẬT (ảnh khác) vẫn thấp dù")
        print("   self-match pass -> đây là vấn đề THẬT về BER giữa 2 ảnh khác nhau")
        print("   của cùng người (κ/M_matrix chưa phù hợp với AdaFace+LFW), KHÔNG")
        print("   phải bug code. Cần chạy look4noncerate()/look4noncerate_joint() từ")
        print("   wifakey_lib/utils.py trên chính embedding LFW này để tìm κ mới.")
    else:
        print("\n✅ Genuine pairs thật cũng cho tỷ lệ hợp lý -> pipeline gốc ổn.")
        print("   => Bug chắc chắn nằm trong research/pipeline/verify_variant.py hoặc")
        print(
            "      research/quantizer/v0_lssc_with_confidence.py khi dùng cho A/B test."
        )


if __name__ == "__main__":
    main()
