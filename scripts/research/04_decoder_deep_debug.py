"""
04_decoder_deep_debug.py

Sau khi Test 0 (parity check) đã xác nhận G/H khớp nhau về đại số, và
Test 1 (decoder thuần) vẫn fail 100%, lỗi CHẮC CHẮN nằm trong TF graph
hoặc trọng số của decoder. Script này thu hẹp tiếp bằng 3 test độc lập:

TEST A - Kiểm tra shape trọng số đã load từ file .txt:
    So khớp shape thực tế của Weights_Var{i}/Biases_Var{i} với shape lý
    thuyết (dựa trên sum_edge = tổng số cạnh trong Tanner graph).
    Lệch shape => sai file, sai định dạng, hoặc numpy tự broadcast âm thầm
    (không báo lỗi nhưng kết quả vô nghĩa).

TEST B - All-zero codeword (trường hợp dễ nhất có thể có):
    Codeword toàn số 0 luôn thỏa H*c=0 với MỌI ma trận H (không phụ thuộc
    gì vào cấu trúc mã). Nếu decoder vẫn fail ở đây, gần như chắc chắn là
    lỗi nằm ở graph/trọng số, không còn nghi ngờ gì về mã nữa.

TEST C - Đảo quy ước dấu quyết định:
    Thử cả '> 0' lẫn '< 0' khi ra quyết định bit từ y_pred_llr. Nếu đảo
    dấu làm decode đúng trở lại => lỗi là quy ước dấu bị ngược ở đâu đó
    (rất dễ sửa, không phải lỗi trọng số).

Cách chạy:
    python scripts/research/04_decoder_deep_debug.py
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation as orig_modulation


def test_a_weight_shapes(handler):
    print("=== TEST A: Shape trọng số đã load ===")
    base_pcm = handler.code_PCM_base.copy()
    pcm01 = np.where(base_pcm == -1, 0, 1)
    sum_edge_v = np.sum(pcm01, axis=0)
    sum_edge = int(np.sum(sum_edge_v))
    print(f"  sum_edge (tổng số cạnh Tanner graph, lý thuyết): {sum_edge}")

    for i in [0, 1, handler.iters_max - 1]:
        w_path = os.path.join(handler.weights_path, f"Weights_Var{i}.txt")
        b_path = os.path.join(handler.biases_path, f"Biases_Var{i}.txt")
        w = np.loadtxt(w_path, delimiter=",", dtype=np.float32)
        b = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        print(f"  iter {i}: Weights shape={w.shape}, Biases shape={b.shape}, "
              f"Weights mean={w.mean():.4f}, std={w.std():.4f}, "
              f"Biases mean={b.mean():.4f}")
        if w.size == 0 or np.allclose(w, 0):
            print(f"    ❌ Weights_Var{i} rỗng hoặc toàn 0 - file trọng số có vấn đề!")
        if w.shape[0] != sum_edge and w.ndim >= 1:
            print(f"    ⚠️  Shape {w.shape} không khớp sum_edge={sum_edge} theo chiều 0 - "
                  f"kiểm tra lại file trọng số có đúng khớp với N/m/Z hiện tại không.")


def test_b_all_zero_codeword(handler):
    print("\n=== TEST B: All-zero codeword (trường hợp dễ nhất tuyệt đối) ===")
    codeword = np.zeros(handler.feature_length, dtype=bool)  # luôn thỏa H*c=0
    llr = orig_modulation.BPSK(codeword).astype(np.float32)  # toàn -1

    y_llr = llr.reshape((1, handler.N, handler.Z))
    y_pred_llr = handler.sess.run(handler.decoder_output, feed_dict={handler.xa: y_llr})

    decoded_gt0 = (y_pred_llr > 0).astype(int).flatten()
    decoded_lt0 = (y_pred_llr < 0).astype(int).flatten()

    print(f"  Input: toàn bit 0 (LLR toàn -{1.0})")
    print(f"  Output y_pred_llr: min={y_pred_llr.min():.3f}, max={y_pred_llr.max():.3f}, "
          f"mean={y_pred_llr.mean():.3f}")
    print(f"  Decode với '>0': số bit=1 trong kết quả: {decoded_gt0.sum()}/{len(decoded_gt0)} "
          f"(kỳ vọng: 0 nếu decode đúng)")
    print(f"  Decode với '<0': số bit=1 trong kết quả: {decoded_lt0.sum()}/{len(decoded_lt0)} "
          f"(kỳ vọng: {len(decoded_lt0)} nếu đây là quy ước đúng thay thế)")

    if decoded_gt0.sum() == 0:
        print("  ✅ Decode all-zero ĐÚNG với quy ước '>0' hiện tại.")
    elif decoded_lt0.sum() == len(decoded_lt0):
        print("  ⚠️  Quy ước dấu bị NGƯỢC: nên dùng '<0' thay vì '>0' khi ra quyết định.")
    else:
        print("  ❌ Cả 2 quy ước đều sai với input dễ nhất có thể có "
              "-> lỗi nằm ở graph/trọng số, không phải quy ước dấu.")
    return y_pred_llr


def test_c_random_codeword_both_conventions(handler, n_trials=20):
    print("\n=== TEST C: Random codeword, thử cả 2 quy ước dấu ===")
    n_ok_gt0, n_ok_lt0 = 0, 0
    for _ in range(n_trials):
        random_key = np.random.randint(0, 2, size=(1, handler.key_length), dtype=int)
        codeword = (handler.encoder.encode_LDPC(random_key).flatten() % 2).astype(bool)
        llr = orig_modulation.BPSK(codeword).astype(np.float32)

        y_llr = llr.reshape((1, handler.N, handler.Z))
        y_pred_llr = handler.sess.run(handler.decoder_output, feed_dict={handler.xa: y_llr})

        decoded_gt0 = (y_pred_llr > 0).astype(int).flatten()[: handler.key_length]
        decoded_lt0 = (y_pred_llr < 0).astype(int).flatten()[: handler.key_length]

        n_ok_gt0 += int(np.array_equal(decoded_gt0, random_key.flatten()))
        n_ok_lt0 += int(np.array_equal(decoded_lt0, random_key.flatten()))

    print(f"  Thành công với '>0': {n_ok_gt0}/{n_trials}")
    print(f"  Thành công với '<0': {n_ok_lt0}/{n_trials}")
    if n_ok_lt0 > n_ok_gt0:
        print("  ⚠️  XÁC NHẬN: quy ước dấu bị ngược - sửa '(y_pred_llr > 0)' "
              "thành '(y_pred_llr < 0)' trong wifakey_handler.verify().")
    elif n_ok_gt0 == n_trials:
        print("  ✅ Quy ước hiện tại đúng, decoder hoạt động tốt với random codeword.")
    else:
        print("  ❌ Cả 2 quy ước đều không đạt 100% dù Test 0 xác nhận G/H khớp nhau "
              "-> nghi vấn cao nhất chuyển về TEST A (trọng số) hoặc lỗi wiring graph "
              "(W_odd2even/W_even2odd/W_output/Lift_M1/Lift_M2 trong _build_decoder()).")


def main():
    handler = WiFaKeyHandler()
    test_a_weight_shapes(handler)
    test_b_all_zero_codeword(handler)
    test_c_random_codeword_both_conventions(handler)


if __name__ == "__main__":
    main()