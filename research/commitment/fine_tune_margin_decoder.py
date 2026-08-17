"""
fine_tune_margin_decoder.py

Fine‑tune Neural‑MS decoder (20 vòng cuối) trên dữ liệu từ Margin Selection.
"""

import os, sys, numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

N, m, Z = 52, 42, 16
ITERS_MAX = 25
TRAINABLE_ITERS = 20
FREEZE_UNTIL = ITERS_MAX - TRAINABLE_ITERS

BASEGRAPH_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "BaseGraph", "BaseGraph2_Set0.txt"
)
ORIGINAL_WEIGHTS_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "Weights_Var_MS"
)
ORIGINAL_BIASES_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "Biases_Var_MS"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "margin_training_data")
TRAIN_LLR_PATH = os.path.join(DATA_DIR, "train_llr.npy")
TRAIN_TARGET_PATH = os.path.join(DATA_DIR, "train_target.npy")

OUTPUT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(__file__), "weights", "Weights_Var_MS_finetuned_margin"
)
OUTPUT_BIASES_DIR = os.path.join(
    os.path.dirname(__file__), "weights", "Biases_Var_MS_finetuned_margin"
)
os.makedirs(OUTPUT_WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_BIASES_DIR, exist_ok=True)

BATCH_SIZE = 1
ACCUM_STEPS = 8
VALID_FRAC = 0.15
N_EPOCHS = 500
LEARNING_RATE = 1e-4
PATIENCE = 30


def build_finetune_graph():
    # Đọc base graph
    code_PCM_base = np.loadtxt(BASEGRAPH_PATH, int, delimiter=None)
    code_PCM_bin = (code_PCM_base != -1).astype(np.int32)

    sum_edge_c = np.sum(code_PCM_bin, axis=1)
    sum_edge_v = np.sum(code_PCM_bin, axis=0)
    sum_edge = int(np.sum(sum_edge_v))
    neurons_per_odd_layer = sum_edge

    # ---- Xây dựng các ma trận cố định ----
    W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    W_skipconn2even = np.zeros((N, sum_edge), dtype=np.float32)
    W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    W_output = np.zeros((sum_edge, N), dtype=np.float32)

    k = 0
    for j in range(code_PCM_bin.shape[1]):
        for i in range(code_PCM_bin.shape[0]):
            if code_PCM_bin[i, j] == 1:
                num_conn = int(np.sum(code_PCM_bin[:, j]))
                idx_list = np.argwhere(code_PCM_bin[:, j] == 1)
                for l in range(num_conn):
                    vec = np.zeros(sum_edge, dtype=np.float32)
                    for r in range(code_PCM_bin.shape[0]):
                        if code_PCM_bin[r, j] == 1 and idx_list[l][0] != r:
                            idx_row = np.cumsum(code_PCM_bin[r, 0 : j + 1])[-1] - 1
                            offset = 0 if r == 0 else np.cumsum(sum_edge_c[0:r])[-1]
                            vec[idx_row + offset] = 1
                    W_odd2even[:, k] = vec
                    k += 1
                break
    k = 0
    for j in range(code_PCM_bin.shape[1]):
        for i in range(code_PCM_bin.shape[0]):
            if code_PCM_bin[i, j] == 1:
                idx_row = np.cumsum(code_PCM_bin[i, 0 : j + 1])[-1] - 1
                start = 0 if i == 0 else np.cumsum(sum_edge_c[0:i])[-1]
                end = np.cumsum(sum_edge_c[0 : i + 1])[-1]
                W_even2odd[k, start:end] = 1.0
                W_even2odd[k, start + idx_row] = 0.0
                k += 1
    k = 0
    for j in range(code_PCM_bin.shape[1]):
        for i in range(code_PCM_bin.shape[0]):
            if code_PCM_bin[i, j] == 1:
                idx_row = np.cumsum(code_PCM_bin[i, 0 : j + 1])[-1] - 1
                offset = 0 if i == 0 else np.cumsum(sum_edge_c[0:i])[-1]
                W_output[offset + idx_row, k] = 1.0
        k += 1
    k = 0
    for j in range(code_PCM_bin.shape[1]):
        for i in range(code_PCM_bin.shape[0]):
            if code_PCM_bin[i, j] == 1:
                W_skipconn2even[j, k] = 1.0
                k += 1

    # Ma trận lifting
    Lift_M1 = np.zeros(
        (neurons_per_odd_layer * Z, neurons_per_odd_layer * Z), np.float32
    )
    Lift_M2 = np.zeros(
        (neurons_per_odd_layer * Z, neurons_per_odd_layer * Z), np.float32
    )
    k = 0
    for j in range(code_PCM_base.shape[1]):
        for i in range(code_PCM_base.shape[0]):
            if code_PCM_base[i, j] != -1:
                shift = code_PCM_base[i, j] % Z
                for h in range(Z):
                    Lift_M1[k * Z + h, k * Z + (h + shift) % Z] = 1
                k += 1
    k = 0
    for i in range(code_PCM_base.shape[0]):
        for j in range(code_PCM_base.shape[1]):
            if code_PCM_base[i, j] != -1:
                shift = code_PCM_base[i, j] % Z
                for h in range(Z):
                    Lift_M2[k * Z + h, k * Z + (h + shift) % Z] = 1
                k += 1

    # ---- Xây dựng đồ thị TF ----
    graph = tf.Graph()
    with graph.as_default():
        xa = tf.placeholder(tf.float32, [None, N, Z], name="xa")
        target = tf.placeholder(tf.float32, [None, N, Z], name="target")

        # Load trọng số gốc và đóng băng/cho phép huấn luyện
        net = {}
        for i in range(ITERS_MAX):
            w_orig = np.loadtxt(
                os.path.join(ORIGINAL_WEIGHTS_PATH, f"Weights_Var{i}.txt"),
                delimiter=",",
                dtype=np.float32,
            )
            b_orig = np.loadtxt(
                os.path.join(ORIGINAL_BIASES_PATH, f"Biases_Var{i}.txt"),
                delimiter=",",
                dtype=np.float32,
            )
            is_trainable = i >= FREEZE_UNTIL
            net[f"Weights_Var{i}"] = tf.Variable(
                w_orig.copy(), name=f"Weights_Var{i}", trainable=is_trainable
            )
            net[f"Biases_Var{i}"] = tf.Variable(
                b_orig.copy(), name=f"Biases_Var{i}", trainable=is_trainable
            )

        # Các constant
        W_skipconn2even_c = tf.constant(W_skipconn2even, dtype=tf.float32)
        W_odd2even_c = tf.constant(W_odd2even, dtype=tf.float32)
        Lift_M1_T_c = tf.constant(Lift_M1.transpose(), dtype=tf.float32)
        Lift_M2_c = tf.constant(Lift_M2, dtype=tf.float32)
        W_even2odd_flat = tf.constant(W_even2odd.T.reshape(-1), dtype=tf.float32)
        W_output_c = tf.constant(W_output, dtype=tf.float32)

        xa_input = tf.transpose(xa, [0, 2, 1])
        net["LLRa0"] = tf.zeros_like(tf.matmul(xa_input, W_skipconn2even_c))

        for i in range(ITERS_MAX):
            x0 = tf.matmul(xa_input, W_skipconn2even_c)
            x1 = tf.matmul(net[f"LLRa{i}"], W_odd2even_c)
            x2 = tf.add(x0, x1)
            x2 = tf.transpose(x2, [0, 2, 1])
            x2 = tf.reshape(x2, [-1, neurons_per_odd_layer * Z])
            x2 = tf.matmul(x2, Lift_M1_T_c)
            x2 = tf.reshape(x2, [-1, neurons_per_odd_layer, Z])
            x2 = tf.transpose(x2, [0, 2, 1])
            x_tile = tf.tile(x2, [1, 1, neurons_per_odd_layer])
            x_tile_mul = tf.multiply(x_tile, W_even2odd_flat)
            x2_1 = tf.reshape(
                x_tile_mul, [-1, Z, neurons_per_odd_layer, neurons_per_odd_layer]
            )
            x2_abs = tf.add(
                tf.abs(x2_1), 10000.0 * (1.0 - tf.cast(tf.abs(x2_1) > 0, tf.float32))
            )
            x3 = tf.reduce_min(x2_abs, axis=3)
            x2_2 = -x2_1
            x4 = tf.add(tf.zeros_like(x2_1), 1.0 - 2.0 * tf.cast(x2_2 < 0, tf.float32))
            x4_prod = -tf.reduce_prod(x4, axis=3)
            x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
            x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
            x_output_0 = tf.reshape(x_output_0, [-1, Z * neurons_per_odd_layer])
            x_output_0 = tf.matmul(x_output_0, Lift_M2_c)
            x_output_0 = tf.reshape(x_output_0, [-1, neurons_per_odd_layer, Z])
            x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
            x_output_1 = tf.add(
                tf.multiply(tf.abs(x_output_0), net[f"Weights_Var{i}"]),
                net[f"Biases_Var{i}"],
            )
            x_output_1 = tf.multiply(x_output_1, tf.cast(x_output_1 > 0, tf.float32))
            llr_next = tf.multiply(x_output_1, tf.sign(x_output_0))

            if i == FREEZE_UNTIL - 1:
                llr_next = tf.stop_gradient(llr_next)

            net[f"LLRa{i+1}"] = llr_next
            y_output_2 = tf.matmul(net[f"LLRa{i+1}"], W_output_c)
            y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
            y_output_4 = tf.add(xa, y_output_3)
            net[f"ya_output{i}"] = tf.reshape(y_output_4, [-1, N * Z])

        decoder_output = net[f"ya_output{ITERS_MAX-1}"]

        # Loss: tập trung vào 160 bit info
        code_k = N - m
        key_len = code_k * Z
        loss_weights = np.ones(N * Z, dtype=np.float32)
        loss_weights[:key_len] = 8.0
        loss_weights_c = tf.constant(loss_weights, dtype=tf.float32)

        target_flat = tf.reshape(target, [-1, N * Z])
        per_bit_loss = tf.nn.softplus(-target_flat * decoder_output)
        loss = tf.reduce_mean(per_bit_loss * loss_weights_c)

        # Metrics
        bit_correct = tf.reduce_mean(
            tf.cast(tf.equal(tf.sign(decoder_output), tf.sign(target_flat)), tf.float32)
        )
        dec_key = decoder_output[:, :key_len]
        targ_key = target_flat[:, :key_len]
        all_correct = tf.reduce_all(
            tf.equal(tf.sign(dec_key), tf.sign(targ_key)), axis=1
        )
        exact_match = tf.reduce_mean(tf.cast(all_correct, tf.float32))

        # Optimizer với gradient accumulation
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        trainable_vars = [
            v
            for v in [net[f"Weights_Var{i}"] for i in range(ITERS_MAX)]
            + [net[f"Biases_Var{i}"] for i in range(ITERS_MAX)]
            if v.trainable
        ]
        grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable_vars)
        grad_placeholders = []
        grad_fetch_tensors = []
        apply_pairs = []
        for g, v in grads_and_vars:
            grad_fetch_tensors.append(g)
            ph = tf.placeholder(tf.float32, shape=v.shape)
            grad_placeholders.append(ph)
            apply_pairs.append((ph, v))
        apply_op = optimizer.apply_gradients(apply_pairs)

        init_op = tf.global_variables_initializer()

    return {
        "graph": graph,
        "xa": xa,
        "target": target,
        "loss": loss,
        "bit_correct": bit_correct,
        "exact_match": exact_match,
        "grad_fetch_tensors": grad_fetch_tensors,
        "grad_placeholders": grad_placeholders,
        "apply_op": apply_op,
        "init_op": init_op,
        "net": net,
    }


def main():
    # Load dữ liệu
    print("Đang load dữ liệu huấn luyện...")
    train_llr = np.load(TRAIN_LLR_PATH).astype(np.float32)
    train_target_raw = np.load(TRAIN_TARGET_PATH).astype(np.float32)
    train_target = train_target_raw.reshape((-1, N, Z))
    print(f"Kích thước dữ liệu: LLR {train_llr.shape}, Target {train_target.shape}")

    # Chia train / validation
    n_total = train_llr.shape[0]
    n_val = int(n_total * VALID_FRAC)
    indices = np.random.permutation(n_total)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    val_llr = train_llr[val_idx]
    val_target = train_target[val_idx]
    train_llr = train_llr[train_idx]
    train_target = train_target[train_idx]
    print(f"Train: {len(train_idx)} mẫu, Validation: {len(val_idx)} mẫu")

    # Xây dựng đồ thị
    print("Xây dựng đồ thị fine‑tune (đóng băng 5 vòng đầu)...")
    tg = build_finetune_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    best_val_exact = -1.0
    best_epoch = -1
    best_weights = None
    patience_counter = 0

    with tf.Session(graph=tg["graph"], config=config) as sess:
        sess.run(tg["init_op"])

        # Đánh giá ban đầu
        eval_batch_size = 4
        val_loss, val_bit, val_exact = 0.0, 0.0, 0.0
        for start in range(0, len(val_idx), eval_batch_size):
            end = min(start + eval_batch_size, len(val_idx))
            feed_val = {
                tg["xa"]: val_llr[start:end],
                tg["target"]: val_target[start:end],
            }
            l, b, e = sess.run(
                [tg["loss"], tg["bit_correct"], tg["exact_match"]], feed_dict=feed_val
            )
            n = end - start
            val_loss += l * n
            val_bit += b * n
            val_exact += e * n
        val_loss /= len(val_idx)
        val_bit /= len(val_idx)
        val_exact /= len(val_idx)
        print(
            f"Epoch 0 (trước fine‑tune): val_loss={val_loss:.4f}, val_bit_acc={val_bit:.4f}, val_exact_match={val_exact:.4f}"
        )

        for epoch in range(1, N_EPOCHS + 1):
            # Huấn luyện với gradient accumulation
            perm = np.random.permutation(len(train_idx))
            accumulated_grads = [
                np.zeros(ph.shape.as_list(), dtype=np.float32)
                for ph in tg["grad_placeholders"]
            ]
            accum_count = 0
            for start in range(0, len(train_idx), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(train_idx))
                batch_idx = perm[start:end]
                feed = {
                    tg["xa"]: train_llr[batch_idx],
                    tg["target"]: train_target[batch_idx],
                }
                grads = sess.run(tg["grad_fetch_tensors"], feed_dict=feed)
                for k in range(len(accumulated_grads)):
                    accumulated_grads[k] += grads[k]
                accum_count += 1
                if accum_count == ACCUM_STEPS:
                    feed_apply = {
                        ph: accumulated_grads[k] / ACCUM_STEPS
                        for k, ph in enumerate(tg["grad_placeholders"])
                    }
                    sess.run(tg["apply_op"], feed_dict=feed_apply)
                    accumulated_grads = [
                        np.zeros(ph.shape.as_list(), dtype=np.float32)
                        for ph in tg["grad_placeholders"]
                    ]
                    accum_count = 0

            # Đánh giá validation
            val_loss, val_bit, val_exact = 0.0, 0.0, 0.0
            for start in range(0, len(val_idx), eval_batch_size):
                end = min(start + eval_batch_size, len(val_idx))
                feed_val = {
                    tg["xa"]: val_llr[start:end],
                    tg["target"]: val_target[start:end],
                }
                l, b, e = sess.run(
                    [tg["loss"], tg["bit_correct"], tg["exact_match"]],
                    feed_dict=feed_val,
                )
                n = end - start
                val_loss += l * n
                val_bit += b * n
                val_exact += e * n
            val_loss /= len(val_idx)
            val_bit /= len(val_idx)
            val_exact /= len(val_idx)
            print(
                f"Epoch {epoch:3d}: val_loss={val_loss:.4f}, val_bit_acc={val_bit:.4f}, val_exact_match={val_exact:.4f}"
            )

            if val_exact > best_val_exact:
                best_val_exact = val_exact
                best_epoch = epoch
                patience_counter = 0
                best_weights = {}
                for i in range(ITERS_MAX):
                    w_val, b_val = sess.run(
                        [tg["net"][f"Weights_Var{i}"], tg["net"][f"Biases_Var{i}"]]
                    )
                    best_weights[i] = (w_val, b_val)
                print(f"  -> Cập nhật tốt nhất (epoch {epoch})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping sau {epoch} epochs.")
                    break

        if best_weights is not None:
            for i in range(ITERS_MAX):
                w, b = best_weights[i]
                np.savetxt(
                    os.path.join(OUTPUT_WEIGHTS_DIR, f"Weights_Var{i}.txt"),
                    w,
                    delimiter=",",
                )
                np.savetxt(
                    os.path.join(OUTPUT_BIASES_DIR, f"Biases_Var{i}.txt"),
                    b,
                    delimiter=",",
                )
            print(
                f"Đã lưu bộ trọng số tốt nhất (epoch {best_epoch}) vào:\n  {OUTPUT_WEIGHTS_DIR}\n  {OUTPUT_BIASES_DIR}"
            )
        else:
            print("Không có cải thiện nào được ghi nhận.")


if __name__ == "__main__":
    main()
