"""
06_intermediate_tensor_debug.py

Cấu trúc 3 ma trận (W_even2odd, W_odd2even, W_output) đã xác nhận ĐÚNG.
Trọng số đã xác nhận BÌNH THƯỜNG (không outlier). Nhưng biên độ ở vòng
lặp 1 (~500-1500) vượt xa giới hạn lý thuyết (~16-20, tính từ bậc
variable-node cao nhất và trọng số trung bình).

=> Kết luận: lỗi nằm ở MỘT PHÉP TÍNH TRUNG GIAN cụ thể bên trong vòng lặp
(rất có thể là chỗ reshape/transpose/tile giữa 2 chiều Z và chỉ số cạnh),
không thể phát hiện bằng kiểm tra thống kê tổng - cần in giá trị TỪNG BƯỚC.

Script này dựng lại CHÍNH XÁC các phép tính của MỘT vòng lặp đầu tiên
(i=0) trong _build_decoder(), nhưng thêm điểm kiểm tra (fetch) sau MỖI
bước tính toán trung gian, để xác định chính xác bước nào giá trị bắt
đầu vượt quá phạm vi hợp lý (kỳ vọng: mọi giá trị nên nằm trong khoảng
xấp xỉ [-20, 20] ở vòng lặp 1, dựa trên cấu trúc/trọng số đã xác nhận).

Cách chạy:
    python scripts/research/06_intermediate_tensor_debug.py
"""

import os
import sys  # noqa: E402

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from wifakey_module.wifakey_lib import Modulation as orig_modulation  # noqa: E402

N, m, Z = 52, 42, 16
DATA_DIR = os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
WEIGHTS_PATH = os.path.join(DATA_DIR, "Weights_Var_MS")
BIASES_PATH = os.path.join(DATA_DIR, "Biases_Var_MS")


def stat(name, arr):
    a = np.asarray(arr)
    print(
        f"  {name:<14} shape={str(a.shape):<18} min={a.min():>10.4f} "
        f"max={a.max():>10.4f} mean={a.mean():>10.4f}"
    )


def main():
    pcm_path = os.path.join(DATA_DIR, "BaseGraph", "BaseGraph2_Set0.txt")
    code_PCM_base = np.loadtxt(pcm_path, int, delimiter=None)
    code_PCM = (code_PCM_base != -1).astype(np.int32)

    sum_edge_c = np.sum(code_PCM, axis=1)
    sum_edge_v = np.sum(code_PCM, axis=0)
    sum_edge = int(np.sum(sum_edge_v))
    neurons_per_odd_layer = sum_edge

    W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    W_skipconn2even = np.zeros((N, sum_edge), dtype=np.float32)
    W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    W_output = np.zeros((sum_edge, N), dtype=np.float32)

    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM[i, j] == 1:
                num_of_conn = int(np.sum(code_PCM[:, j]))
                idx = np.argwhere(code_PCM[:, j] == 1)
                for l in range(num_of_conn):
                    vec_tmp = np.zeros(sum_edge, dtype=np.float32)
                    for r in range(code_PCM.shape[0]):
                        if code_PCM[r, j] == 1 and idx[l][0] != r:
                            idx_row = np.cumsum(code_PCM[r, 0 : j + 1])[-1] - 1
                            cnt = 0
                            if r > 0:
                                cnt = np.cumsum(sum_edge_c[0:r])[-1]
                            vec_tmp[idx_row + cnt] = 1
                    W_odd2even[:, k] = vec_tmp.transpose()
                    k += 1
                break
    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM[i, j] == 1:
                idx_row = np.cumsum(code_PCM[i, 0 : j + 1])[-1] - 1
                c1, c2 = 0, np.cumsum(sum_edge_c[0 : i + 1])[-1]
                if i > 0:
                    c1 = np.cumsum(sum_edge_c[0:i])[-1]
                W_even2odd[k, c1:c2] = 1.0
                W_even2odd[k, c1 + idx_row] = 0.0
                k += 1
    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM[i, j] == 1:
                idx_row = np.cumsum(code_PCM[i, 0 : j + 1])[-1] - 1
                cnt = 0
                if i > 0:
                    cnt = np.cumsum(sum_edge_c[0:i])[-1]
                W_output[cnt + idx_row, k] = 1.0
        k += 1
    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM[i, j] == 1:
                W_skipconn2even[j, k] = 1.0
                k += 1

    Lift_M1 = np.zeros(
        (neurons_per_odd_layer * Z, neurons_per_odd_layer * Z), np.float32
    )
    Lift_M2 = np.zeros(
        (neurons_per_odd_layer * Z, neurons_per_odd_layer * Z), np.float32
    )
    k = 0
    for j in range(code_PCM.shape[1]):
        for i in range(code_PCM.shape[0]):
            if code_PCM_base[i, j] != -1:
                Lift_num = code_PCM_base[i, j] % Z
                for h in range(Z):
                    Lift_M1[k * Z + h, k * Z + (h + Lift_num) % Z] = 1
                k += 1
    k = 0
    for i in range(code_PCM.shape[0]):
        for j in range(code_PCM.shape[1]):
            if code_PCM_base[i, j] != -1:
                Lift_num = code_PCM_base[i, j] % Z
                for h in range(Z):
                    Lift_M2[k * Z + h, k * Z + (h + Lift_num) % Z] = 1
                k += 1

    # Kiểm tra Lift_M1/Lift_M2 có phải hoán vị hợp lệ không (mỗi hàng/cột đúng 1 giá trị 1)
    print(
        "Kiểm tra Lift_M1 là hoán vị hợp lệ:",
        (
            "✅"
            if np.all(Lift_M1.sum(axis=0) == 1) and np.all(Lift_M1.sum(axis=1) == 1)
            else "❌ SAI"
        ),
    )
    print(
        "Kiểm tra Lift_M2 là hoán vị hợp lệ:",
        (
            "✅"
            if np.all(Lift_M2.sum(axis=0) == 1) and np.all(Lift_M2.sum(axis=1) == 1)
            else "❌ SAI"
        ),
    )
    print()

    graph = tf.Graph()
    with graph.as_default():
        w0 = np.loadtxt(
            os.path.join(WEIGHTS_PATH, "Weights_Var0.txt"),
            delimiter=",",
            dtype=np.float32,
        )
        b0 = np.loadtxt(
            os.path.join(BIASES_PATH, "Biases_Var0.txt"),
            delimiter=",",
            dtype=np.float32,
        )
        Weights_Var0 = tf.constant(w0)
        Biases_Var0 = tf.constant(b0)

        xa = tf.placeholder(tf.float32, shape=[1, N, Z], name="xa")
        xa_input = tf.transpose(xa, [0, 2, 1])
        LLRa0 = tf.zeros((1, Z, sum_edge), dtype=tf.float32)

        x0 = tf.matmul(xa_input, W_skipconn2even)
        x1 = tf.matmul(LLRa0, W_odd2even)
        x2 = tf.add(x0, x1)
        x2_before_lift = x2

        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [1, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_M1.transpose())
        x2 = tf.reshape(x2, [1, neurons_per_odd_layer, Z])
        x2_after_lift = tf.transpose(x2, [0, 2, 1])

        x_tile = tf.tile(x2_after_lift, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(
            x_tile_mul, [1, Z, neurons_per_odd_layer, neurons_per_odd_layer]
        )

        x2_abs = tf.add(
            tf.abs(x2_1), 10000 * (1 - tf.cast(tf.abs(x2_1) > 0, tf.float32))
        )
        x3 = tf.reduce_min(x2_abs, axis=3)

        x2_2 = -x2_1
        x4 = tf.add(
            tf.zeros((1, Z, neurons_per_odd_layer, neurons_per_odd_layer)),
            1 - 2 * tf.cast(x2_2 < 0, tf.float32),
        )
        x4_prod = -tf.reduce_prod(x4, axis=3)
        x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
        x_output_0_before_lift = x_output_0

        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [1, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_M2)
        x_output_0 = tf.reshape(x_output_0, [1, neurons_per_odd_layer, Z])
        x_output_0_after_lift = tf.transpose(x_output_0, [0, 2, 1])

        x_output_1 = tf.add(
            tf.multiply(tf.abs(x_output_0_after_lift), Weights_Var0), Biases_Var0
        )
        x_output_1_relu = tf.multiply(x_output_1, tf.cast(x_output_1 > 0, tf.float32))
        LLRa1 = tf.multiply(x_output_1_relu, tf.sign(x_output_0_after_lift))

        y_output_2 = tf.matmul(LLRa1, W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xa, y_output_3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        codeword = np.zeros(N * Z, dtype=bool)
        llr_input = orig_modulation.BPSK(codeword).astype(np.float32).reshape(1, N, Z)

        (
            v_x2_before,
            v_x2_after,
            v_x2_1,
            v_x3,
            v_x_out0_before,
            v_x_out0_after,
            v_x_out1,
            v_llra1,
            v_y4,
        ) = sess.run(
            [
                x2_before_lift,
                x2_after_lift,
                x2_1,
                x3,
                x_output_0_before_lift,
                x_output_0_after_lift,
                x_output_1,
                LLRa1,
                y_output_4,
            ],
            feed_dict={xa: llr_input},
        )

        print(
            "=== Giá trị từng bước trung gian ở vòng lặp 1 (kỳ vọng: hầu hết trong khoảng [-20,20]) ==="
        )
        stat("x2 (before lift)", v_x2_before)
        stat("x2 (after lift)", v_x2_after)
        stat("x2_1 (tile*mask)", v_x2_1)
        stat("x3 (min-sum)", v_x3)
        stat("x_out0 (trước lift)", v_x_out0_before)
        stat("x_out0 (sau lift)", v_x_out0_after)
        stat("x_output_1", v_x_out1)
        stat("LLRa1", v_llra1)
        stat("y_output_4 (final)", v_y4)


if __name__ == "__main__":
    main()
