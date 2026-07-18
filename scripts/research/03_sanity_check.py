"""
03_sanity_check.py

FRR=1.0 tuyệt đối ở CẢ baseline lẫn soft_llr là dấu hiệu bất thường -
cần cô lập nguyên nhân trước khi kết luận bất cứ điều gì về soft-LLR
hay về hệ thống. Chạy 3 test độc lập, theo thứ tự tăng dần độ phức tạp:

TEST 1 - Decoder thuần túy (không qua biometric):
    Sinh random_key -> encode LDPC -> decode NGAY LẬP TỨC với LLR không nhiễu
    (chính codeword đó, dấu đúng, biên độ lớn). Nếu decoder hoạt động đúng,
    đây PHẢI thành công 100% - đây là trường hợp dễ nhất có thể có.
    Fail ở đây => lỗi nằm ở decoder/session TF (trọng số sai, graph sai,
    hoặc bug trong NeuralMSOriginal wrapper).

TEST 2 - Self-match (verify bằng CHÍNH ảnh vừa enroll):
    Không có nhiễu biometric (cùng 1 embedding), chỉ có nhiễu do mask
    (nhưng mask áp dụng như nhau cho cả 2 lần vì y hệt embedding).
    y_noisy_bits phải là vector toàn số 0. PHẢI thành công gần 100%.
    Fail ở đây (mà Test 1 pass) => lỗi nằm ở verify_variant.py (cách
    ghép quantizer/mask/XOR), KHÔNG phải ở decoder hay biometric.

TEST 3 - Raw Hamming BER giữa cặp genuine thật (KHÔNG qua decoder):
    Đo trực tiếp tỷ lệ bit lỗi giữa b_selected(enroll) và b_selected(verify)
    của cùng người, TRƯỚC khi XOR với helper_data / trước khi decode.
    So với ngưỡng chịu lỗi lý thuyết của code (khoảng 0.15-0.18 theo comment
    gốc trong utils.py, dù cần xác nhận lại cho đúng Z=16 chứ không phải
    Z=10 như ghi chú gốc đề cập). Nếu Test 1+2 pass nhưng BER thật > ngưỡng
    này => đây là giới hạn THẬT của hệ thống trên dataset này (κ/M_matrix
    chưa phù hợp với AdaFace+LFW), không phải bug code.

Cách chạy:
    python scripts/03_sanity_check.py
"""

import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation as orig_modulation
from research.decoder.v0_neural_ms_original import NeuralMSOriginal
from research.quantizer.v0_lssc_with_confidence import binarize_with_confidence


def test1_decoder_only(handler, decoder, n_trials=20):
    print("=== TEST 1: Decoder thuần túy (LLR không nhiễu) ===")
    n_ok = 0
    for _ in range(n_trials):
        random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
        codeword = (handler.encoder.encode_LDPC(random_key).flatten() % 2).astype(bool)

        # LLR "hoàn hảo": chính codeword đó, biên độ lớn (rất tin cậy)
        llr = orig_modulation.BPSK(codeword).astype(np.float32)

        decoded_key = decoder.decode(llr)
        if np.array_equal(decoded_key, random_key.flatten()):
            n_ok += 1

    rate = n_ok / n_trials
    print(f"  Thành công: {n_ok}/{n_trials} ({rate:.1%})")
    if rate < 0.95:
        print("  ❌ NGHI VẤN: decoder lỗi ngay cả với input không nhiễu.")
        print(
            "     -> Kiểm tra: trọng số Weights_Var*/Biases_Var* có đúng file, đúng shape không?"
        )
        print(
            "     -> Kiểm tra: graph decoder build đúng với N/m/Z hiện tại (52/42/16) không?"
        )
    else:
        print("  ✅ Decoder hoạt động đúng ở điều kiện lý tưởng.")
    return rate


def test2_self_match(handler, decoder, embeddings_sample, n_trials=20):
    print("\n=== TEST 2: Self-match (verify bằng chính ảnh vừa enroll) ===")
    n_ok = 0
    tested = 0
    for emb in embeddings_sample[:n_trials]:
        helper_data, mask_r, key_hash = handler.enroll(emb)
        # verify bằng CHÍNH embedding vừa dùng để enroll -> phải là trường hợp dễ nhất
        success = handler.verify(emb, helper_data, mask_r, key_hash)
        n_ok += int(success)
        tested += 1

    rate = n_ok / max(tested, 1)
    print(f"  Thành công: {n_ok}/{tested} ({rate:.1%})")
    if rate < 0.95:
        print("  ❌ NGHI VẤN: self-match cũng fail dù không có nhiễu biometric.")
        print(
            "     -> Nếu Test 1 PASS nhưng Test 2 FAIL: lỗi nằm ở logic verify_variant.py"
        )
        print("        (mask/XOR/feature_length), KHÔNG phải decoder.")
        print(
            "     -> Lưu ý: test này gọi THẲNG handler.verify() gốc (không qua harness"
        )
        print(
            "        research/), nên nếu vẫn fail ở đây thì lỗi nằm trong chính pipeline gốc."
        )
    else:
        print(
            "  ✅ Self-match hoạt động đúng -> lỗi 100% FRR trước đó KHÔNG phải do bug"
        )
        print(
            "     trong verify_variant.py, mà do BER thật giữa 2 ảnh khác nhau quá cao (xem Test 3)."
        )
    return rate


def test3_raw_ber(handler, genuine_pairs_sample):
    print("\n=== TEST 3: Raw Hamming BER giữa cặp genuine thật (trước decode) ===")
    bers = []
    for emb_enroll, emb_verify in genuine_pairs_sample:
        helper_data, mask_r, key_hash = handler.enroll(emb_enroll)

        projected_v = np.dot(emb_verify, handler.M_matrix)
        b_full_v, _ = binarize_with_confidence(projected_v, handler.intervals)
        b_masked_v = (b_full_v.astype(np.uint8) & mask_r)[: handler.feature_length]

        # b_selected phía enroll suy ra lại từ helper_data + random_key đã biết
        # (ở đây đo gián tiếp qua y_noisy so với codeword thay vì encode lại,
        #  đơn giản hơn: đo trực tiếp bit lỗi so với b_selected lúc enroll)
        projected_e = np.dot(emb_enroll, handler.M_matrix)
        b_full_e, _ = binarize_with_confidence(projected_e, handler.intervals)
        b_masked_e = (b_full_e.astype(np.uint8) & mask_r)[: handler.feature_length]

        ber = np.mean(b_masked_e != b_masked_v)
        bers.append(ber)

    bers = np.array(bers)
    print(f"  BER trung bình: {bers.mean():.4f}")
    print(f"  BER median:     {np.median(bers):.4f}")
    print(
        f"  % cặp có BER <= 0.176 (ngưỡng tham khảo, cần xác nhận lại cho Z=16): "
        f"{np.mean(bers <= 0.176):.1%}"
    )
    if bers.mean() > 0.20:
        print(
            "  ❌ BER trung bình quá cao so với khả năng sửa lỗi thực tế của LDPC rate~0.19."
        )
        print(
            "     -> κ=0.3125 hard-code có thể KHÔNG phù hợp với AdaFace+LFW hiện tại,"
        )
        print(
            "        cần chạy lại look4noncerate()/look4noncerate_joint() (đã có sẵn trong"
        )
        print("        wifakey_lib/utils.py) trên chính dataset LFW này để tìm κ mới.")
    return bers


def main():
    handler = WiFaKeyHandler()
    decoder = NeuralMSOriginal(handler)

    test1_decoder_only(handler, decoder)

    # TODO: nạp embedding thật từ cache (data/processed/embeddings_cache/*.npy)
    # embeddings_sample = [...]
    # genuine_pairs_sample = [...]
    # test2_self_match(handler, decoder, embeddings_sample)
    # test3_raw_ber(handler, genuine_pairs_sample)

    print("\n⚠️  Cắm dữ liệu embedding thật vào main() (đọc từ")
    print("   data/processed/embeddings_cache/) rồi chạy lại Test 2 và Test 3.")


if __name__ == "__main__":
    main()
