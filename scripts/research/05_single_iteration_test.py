"""
02_single_iteration_test.py

Test B/C cho thấy output "nổ" biên độ (lên tới ±390) sau 25 vòng lặp với
input dễ nhất có thể có (all-zero codeword). Câu hỏi cần trả lời: lỗi đã
xuất hiện ngay ở VÒNG LẶP ĐẦU TIÊN (bug wiring cơ bản), hay chỉ khuếch đại
dần qua nhiều vòng (có thể là vấn đề ổn định số học/learning rate của
riêng bộ trọng số, ít nghiêm trọng hơn)?

Script này dựng lại CHÍNH XÁC logic trong _build_decoder(), nhưng dừng ở
từng mốc iters=1, 2, 5, 10, 25 để in ra biên độ output tại mỗi mốc, dùng
CÙNG all-zero codeword như Test B.

Nếu ngay ở iters=1 đã thấy output méo mó/lệch hướng rõ rệt (dù chưa "nổ"
to như ở 25 vòng) -> xác nhận bug nằm ở 1 vòng lặp cơ bản (rất có thể ở
Lift_M1/Lift_M2 hoặc W_even2odd/W_odd2even), không phải do tích lũy số học.

Cách chạy:
    python scripts/research/02_single_iteration_test.py
"""

import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_lib import Modulation as orig_modulation

N, m, Z = 52, 42, 16
DATA_DIR = os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
WEIGHTS_PATH = os.path.join(DATA_DIR, "Weights_Var_MS")
BIASES_PATH = os.path.join(DATA_DIR, "Biases_Var_MS")
CHECKPOINT_ITERS = [1, 2, 5, 10, 25]


def build_decoder_with_checkpoints(max_iters, checkpoint_iters):
    """Copy gần như nguyên vẹn _build_decoder() gốc, chỉ thêm việc lưu lại
    output tại các mốc iteration trung gian để quan sát."""
    graph = tf.Graph()
    outputs_at_checkpoint = {}

    with graph.as_default():
        pcm_path = os.path.join(DATA_DIR, "BaseGraph", "BaseGraph2_Set0.txt")
        code_PCM = np.loadtxt(pcm_path, int, delimiter=None).copy()
        for i in range(code_PCM.shape[0]):
            for j in range(code_PCM.shape[1]):
                code_PCM[i, j] = 0 if code_PCM[i, j] == -1 else 1
        sum_edge_c = np.sum(code_PCM, axis=1)
        sum_edge_v = np.sum(code_PCM, axis=0)
        sum_edge = np.sum(sum_edge_v)

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

        code_PCM1 = np.loadtxt(pcm_path, int, delimiter=None)
        neurons_per_odd_layer = int(np.sum(sum_edge_v))
        Lift_M1 = np.zeros(
            (neurons_per_odd_layer * Z, neurons_per_odd_layer * Z), np.float32
        )
        Lift_M2 = np.zeros(
            (neurons_per_odd_layer * Z, neurons_per_odd_layer * Z), np.float32
        )
        k = 0
        for j in range(code_PCM1.shape[1]):
            for i in range(code_PCM1.shape[0]):
                if code_PCM1[i, j] != -1:
                    Lift_num = code_PCM1[i, j] % Z
                    for h in range(Z):
                        Lift_M1[k * Z + h, k * Z + (h + Lift_num) % Z] = 1
                    k += 1
        k = 0
        for i in range(code_PCM1.shape[0]):
            for j in range(code_PCM1.shape[1]):
                if code_PCM1[i, j] != -1:
                    Lift_num = code_PCM1[i, j] % Z
                    for h in range(Z):
                        Lift_M2[k * Z + h, k * Z + (h + Lift_num) % Z] = 1
                    k += 1

        net_dict = {}
        for i in range(max_iters):
            w = np.loadtxt(
                os.path.join(WEIGHTS_PATH, f"Weights_Var{i}.txt"),
                delimiter=",",
                dtype=np.float32,
            )
            b = np.loadtxt(
                os.path.join(BIASES_PATH, f"Biases_Var{i}.txt"),
                delimiter=",",
                dtype=np.float32,
            )
            net_dict[f"Weights_Var{i}"] = tf.Variable(w.copy(), name=f"Weights_Var{i}")
            net_dict[f"Biases_Var{i}"] = tf.Variable(b.copy(), name=f"Biases_Var{i}")

        xa = tf.placeholder(tf.float32, shape=[1, N, Z], name="xa")
        xa_input = tf.transpose(xa, [0, 2, 1])
        net_dict["LLRa0"] = tf.zeros((1, Z, sum_edge), dtype=tf.float32)

        for i in range(max_iters):
            x0 = tf.matmul(xa_input, W_skipconn2even)
            x1 = tf.matmul(net_dict[f"LLRa{i}"], W_odd2even)
            x2 = tf.add(x0, x1)
            x2 = tf.transpose(x2, [0, 2, 1])
            x2 = tf.reshape(x2, [1, neurons_per_odd_layer * Z])
            x2 = tf.matmul(x2, Lift_M1.transpose())
            x2 = tf.reshape(x2, [1, neurons_per_odd_layer, Z])
            x2 = tf.transpose(x2, [0, 2, 1])
            x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
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
            x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
            x_output_0 = tf.reshape(x_output_0, [1, Z * neurons_per_odd_layer])
            x_output_0 = tf.matmul(x_output_0, Lift_M2)
            x_output_0 = tf.reshape(x_output_0, [1, neurons_per_odd_layer, Z])
            x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
            x_output_1 = tf.add(
                tf.multiply(tf.abs(x_output_0), net_dict[f"Weights_Var{i}"]),
                net_dict[f"Biases_Var{i}"],
            )
            x_output_1 = tf.multiply(x_output_1, tf.cast(x_output_1 > 0, tf.float32))
            net_dict[f"LLRa{i+1}"] = tf.multiply(x_output_1, tf.sign(x_output_0))
            y_output_2 = tf.matmul(net_dict[f"LLRa{i+1}"], W_output)
            y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
            y_output_4 = tf.add(xa, y_output_3)
            out_flat = tf.reshape(y_output_4, [1, N * Z])
            if (i + 1) in checkpoint_iters:
                outputs_at_checkpoint[i + 1] = out_flat

        init_op = tf.global_variables_initializer()

    return graph, xa, outputs_at_checkpoint, init_op


def main():
    max_iters = max(CHECKPOINT_ITERS)
    print(f"Dựng decoder graph với checkpoint tại iters={CHECKPOINT_ITERS} ...")
    graph, xa, outputs_at_checkpoint, init_op = build_decoder_with_checkpoints(
        max_iters, CHECKPOINT_ITERS
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)

        codeword = np.zeros(N * Z, dtype=bool)  # all-zero, luôn hợp lệ
        llr_input = orig_modulation.BPSK(codeword).astype(np.float32).reshape(1, N, Z)

        print(
            f"\n{'Iter':>6} | {'min':>10} | {'max':>10} | {'mean':>10} | {'% bit sai (>0)':>16}"
        )
        print("-" * 65)
        for it in CHECKPOINT_ITERS:
            out = sess.run(outputs_at_checkpoint[it], feed_dict={xa: llr_input})
            wrong_pct = np.mean(out.flatten() > 0) * 100  # kỳ vọng: 0% (input toàn 0)
            print(
                f"{it:>6} | {out.min():>10.3f} | {out.max():>10.3f} | "
                f"{out.mean():>10.3f} | {wrong_pct:>15.1f}%"
            )


if __name__ == "__main__":
    main()
