"""
train.py - Fine-tune Neural-MS decoder trên phân phối nhiễu THẬT.

Ý tưởng: trọng số gốc Weights_Var_MS/Biases_Var_MS được huấn luyện ngầm
định cho biên độ LLR CỐ ĐỊNH (~1, khớp hard BPSK). Khi dùng soft-LLR (biên
độ biến thiên theo độ tin cậy thật), decoder không tận dụng được thông
tin đó vì chưa từng "thấy" phân phối này lúc train. Fine-tune ở đây nghĩa
là tiếp tục train (KHÔNG train lại từ đầu) trên chính phân phối LLR biến
thiên mà hệ thống thực tế sẽ gặp.

QUAN TRỌNG:
- Khởi tạo từ trọng số GỐC (không random init) - chỉ tinh chỉnh.
- Dữ liệu train lấy từ tập 'tune' (Tầng 1) - KHÔNG dùng tập 'select'/'final'
  để tránh rò rỉ dữ liệu đánh giá vào quá trình huấn luyện.
- Lưu trọng số mới vào thư mục RIÊNG (Weights_Var_MS_finetuned/), không
  ghi đè bản gốc - baseline luôn phải còn nguyên để so sánh.

Cách chạy:
    python research/decoder/v1_neural_ms_finetuned/train.py
"""

import os
import sys
import csv
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from wifakey_module.wifakey_lib import Encode
from research.quantizer.v0_lssc_with_confidence import binarize_with_confidence
from research.modulation.v1_soft_distance_llr import SoftDistanceLLR

# ==================== Cấu hình ====================
N, m, Z = 52, 42, 16
ITERS_MAX = 25
ORIGINAL_WEIGHTS_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "Weights_Var_MS"
)
ORIGINAL_BIASES_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "Biases_Var_MS"
)
OUTPUT_WEIGHTS_PATH = "/content/drive/MyDrive/WiFaKey_finetuned/Weights_Var_MS_finetuned"
OUTPUT_BIASES_PATH = "/content/drive/MyDrive/WiFaKey_finetuned/Biases_Var_MS_finetuned"

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
BASEGRAPH_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "BaseGraph", "BaseGraph2_Set0.txt"
)

BATCH_SIZE = 16
N_EPOCHS = 2
LEARNING_RATE = 1e-4
SCALE = 60.0
MIN_MAG = 0.1
MAX_MAG = 5.0
MASKED_MAG = 1.0


def load_embedding(name, imagenum):
    return np.load(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def load_tune_genuine_pairs():
    path = os.path.join(PAIRS_DIR, "tune_genuine.csv")
    pairs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emb1 = load_embedding(row["name_enroll"], row["imagenum_enroll"])
            emb2 = load_embedding(row["name_verify"], row["imagenum_verify"])
            pairs.append((emb1, emb2))
    return pairs


# ==================== Dựng graph decoder (hỗ trợ batch, trainable) ====================
def build_trainable_decoder(batch_size, init_weights_path, init_biases_path):
    """Dựng lại graph giống _build_decoder() gốc, nhưng cho phép batch_size > 1
    và thêm loss + optimizer để fine-tune. Khởi tạo trọng số từ file gốc."""
    code_PCM_base = np.loadtxt(BASEGRAPH_PATH, int, delimiter=None)
    code_PCM = (code_PCM_base != -1).astype(np.int32)

    sum_edge_c = np.sum(code_PCM, axis=1)
    sum_edge_v = np.sum(code_PCM, axis=0)
    sum_edge = int(np.sum(sum_edge_v))
    neurons_per_odd_layer = sum_edge
    
    print(f"sum_edge = {sum_edge}", flush=True)
    bytes_per_tensor = BATCH_SIZE * Z * sum_edge * sum_edge * 4  # float32
    print(f"Mỗi tensor 4D dạng [batch,Z,sum_edge,sum_edge] tốn ~{bytes_per_tensor/1e9:.2f} GB", flush=True)
    print(f"x{ITERS_MAX} vòng x ~4 tensor loại này/vòng (x2_1, x2_abs, x4, ...) => ước lượng thô: "
        f"~{bytes_per_tensor*4*ITERS_MAX/1e9:.1f} GB chỉ riêng forward activations", flush=True)

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

    graph = tf.Graph()
    with graph.as_default():
        net_dict = {}
        for i in range(ITERS_MAX):
            w = np.loadtxt(
                os.path.join(init_weights_path, f"Weights_Var{i}.txt"),
                delimiter=",",
                dtype=np.float32,
            )
            b = np.loadtxt(
                os.path.join(init_biases_path, f"Biases_Var{i}.txt"),
                delimiter=",",
                dtype=np.float32,
            )
            net_dict[f"Weights_Var{i}"] = tf.Variable(
                w.copy(), name=f"Weights_Var{i}", trainable=True
            )
            net_dict[f"Biases_Var{i}"] = tf.Variable(
                b.copy(), name=f"Biases_Var{i}", trainable=True
            )

        xa = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name="xa")
        target = tf.placeholder(
            tf.float32, shape=[batch_size, N, Z], name="target"
        )  # codeword bipolar

        xa_input = tf.transpose(xa, [0, 2, 1])
        net_dict["LLRa0"] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)

        for i in range(ITERS_MAX):
            x0 = tf.matmul(xa_input, W_skipconn2even)
            x1 = tf.matmul(net_dict[f"LLRa{i}"], W_odd2even)
            x2 = tf.add(x0, x1)
            x2 = tf.transpose(x2, [0, 2, 1])
            x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
            x2 = tf.matmul(x2, Lift_M1.transpose())
            x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
            x2 = tf.transpose(x2, [0, 2, 1])
            x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
            W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
            x_tile_mul = tf.multiply(x_tile, W_input_reshape)
            x2_1 = tf.reshape(
                x_tile_mul,
                [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer],
            )
            x2_abs = tf.add(
                tf.abs(x2_1), 10000 * (1 - tf.cast(tf.abs(x2_1) > 0, tf.float32))
            )
            x3 = tf.reduce_min(x2_abs, axis=3)
            x2_2 = -x2_1
            x4 = tf.add(
                tf.zeros((batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer)),
                1 - 2 * tf.cast(x2_2 < 0, tf.float32),
            )
            x4_prod = -tf.reduce_prod(x4, axis=3)
            x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
            x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
            x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
            x_output_0 = tf.matmul(x_output_0, Lift_M2)
            x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
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
            net_dict[f"ya_output{i}"] = tf.reshape(y_output_4, [batch_size, N * Z])

        decoder_output = net_dict[f"ya_output{ITERS_MAX-1}"]
        target_flat = tf.reshape(target, [batch_size, N * Z])

        # Soft-BER loss: khuyến khích decoder_output cùng dấu với target,
        # biên độ càng lớn theo đúng hướng càng tốt (logistic/softplus loss
        # kiểu Nachmani et al. cho neural LDPC decoder).
        loss = tf.reduce_mean(tf.nn.softplus(-target_flat * decoder_output))

        bit_correct = tf.reduce_mean(
            tf.cast(tf.equal(tf.sign(decoder_output), tf.sign(target_flat)), tf.float32)
        )

        trainable_vars = [net_dict[f"Weights_Var{i}"] for i in range(ITERS_MAX)] + [
            net_dict[f"Biases_Var{i}"] for i in range(ITERS_MAX)
        ]
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss, var_list=trainable_vars)

        init_op = tf.global_variables_initializer()

    return {
        "graph": graph,
        "xa": xa,
        "target": target,
        "loss": loss,
        "bit_correct": bit_correct,
        "train_op": train_op,
        "init_op": init_op,
        "net_dict": net_dict,
    }


# ==================== Sinh batch dữ liệu train từ embedding thật ====================
def make_training_batch(handler, encoder, modulation, pairs, batch_size, rng):
    llr_batch = np.zeros((batch_size, N, Z), dtype=np.float32)
    target_batch = np.zeros((batch_size, N, Z), dtype=np.float32)

    idxs = rng.choice(len(pairs), size=batch_size, replace=True)
    for b, idx in enumerate(idxs):
        emb_enroll, emb_verify = pairs[idx]

        b_full_e = handler._binarize_full(emb_enroll).astype(np.uint8)
        u = rng.uniform(0.0, 1.0, size=len(b_full_e))
        mask_r = (u >= handler.kappa).astype(np.uint8)
        b_selected_e = (b_full_e & mask_r)[: handler.feature_length]

        random_key = rng.integers(0, 2, size=(1, handler.key_length))
        codeword = encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected_e, codeword).astype(np.uint8)

        projected_v = np.dot(emb_verify, handler.M_matrix)
        bits_v, confidence_v = binarize_with_confidence(projected_v, handler.intervals)
        b_selected_v = (bits_v.astype(np.uint8) & mask_r)[: handler.feature_length]
        conf_selected = confidence_v[: handler.feature_length]
        mask_selected = mask_r[: handler.feature_length]

        y_noisy_bits = np.logical_xor(b_selected_v, helper_data)
        llr = modulation(
            y_noisy_bits, context={"distance": conf_selected, "mask": mask_selected}
        )

        target_bipolar = codeword.astype(np.float32) * 2 - 1

        llr_batch[b] = llr.reshape(N, Z)
        target_batch[b] = target_bipolar.reshape(N, Z)

    return llr_batch, target_batch


def main():
    os.makedirs(OUTPUT_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_BIASES_PATH, exist_ok=True)

    print(
        "Nạp handler gốc (chỉ để dùng M_matrix/intervals/kappa, KHÔNG dùng session TF của nó)..."
    )
    handler = WiFaKeyHandler()
    encoder = Encode.Proto_LDPC(N, m, Z)
    modulation = SoftDistanceLLR(
        scale=SCALE, min_mag=MIN_MAG, max_mag=MAX_MAG, masked_mag=MASKED_MAG
    )

    print("Nạp cặp genuine từ tập TUNE (Tầng 1)...")
    pairs = load_tune_genuine_pairs()
    print(f"-> {len(pairs)} cặp genuine dùng để fine-tune.\n")

    print("Dựng graph trainable, khởi tạo từ trọng số GỐC...", flush=True)
    tg = build_trainable_decoder(
        BATCH_SIZE, ORIGINAL_WEIGHTS_PATH, ORIGINAL_BIASES_PATH
    )
    print("Graph đã build xong.", flush=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    rng = np.random.default_rng(0)

    with tf.Session(graph=tg["graph"], config=config) as sess:
        sess.run(tg["init_op"])
        print("Session init xong, chuẩn bị chạy step đầu tiên...", flush=True)

        steps_per_epoch = max(len(pairs) // BATCH_SIZE, 1)
        for epoch in range(N_EPOCHS):
            epoch_loss, epoch_acc = [], []
            for step in range(steps_per_epoch):
                llr_batch, target_batch = make_training_batch(
                    handler, encoder, modulation, pairs, BATCH_SIZE, rng
                )
                print("Đã tạo batch đầu tiên, gọi sess.run...", flush=True)
                _, loss_val, acc_val = sess.run(
                    [tg["train_op"], tg["loss"], tg["bit_correct"]],
                    feed_dict={tg["xa"]: llr_batch, tg["target"]: target_batch},
                )
                print(f"Step đầu tiên xong: loss={loss_val}, acc={acc_val}", flush=True)
                epoch_loss.append(loss_val)
                epoch_acc.append(acc_val)

            print(
                f"Epoch {epoch+1}/{N_EPOCHS}: loss={np.mean(epoch_loss):.4f}  "
                f"bit_acc={np.mean(epoch_acc):.4f}"
            )

        print("\nLưu trọng số đã fine-tune...")
        for i in range(ITERS_MAX):
            w_val = sess.run(tg["net_dict"][f"Weights_Var{i}"])
            b_val = sess.run(tg["net_dict"][f"Biases_Var{i}"])
            np.savetxt(
                os.path.join(OUTPUT_WEIGHTS_PATH, f"Weights_Var{i}.txt"),
                w_val,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(OUTPUT_BIASES_PATH, f"Biases_Var{i}.txt"),
                b_val,
                delimiter=",",
            )

    print(f"\n✅ Hoàn tất. Trọng số mới lưu tại:")
    print(f"   {OUTPUT_WEIGHTS_PATH}")
    print(f"   {OUTPUT_BIASES_PATH}")
    print(
        f"   (Trọng số GỐC không hề bị đụng vào - vẫn nguyên tại "
        f"wifakey_module/data/Weights_Var_MS/)"
    )


if __name__ == "__main__":
    main()
