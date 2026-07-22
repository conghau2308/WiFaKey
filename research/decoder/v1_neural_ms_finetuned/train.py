"""
train.py - Fine-tune Neural-MS decoder tren phan phoi nhieu THAT.

QUAN TRONG:
- Khoi tao tu trong so GOC (khong random init) - chi tinh chinh.
- Du lieu train lay tu tap 'tune' (Tang 1) - KHONG dung tap 'select'/'final'.
- Luu trong so moi vao thu muc RIENG (Weights_Var_MS_finetuned/), khong
  ghi de ban goc.
- TRUNCATED BACKPROP: chi backprop qua `trainable_iters` vong lap CUOI
  (trainable=False + tf.stop_gradient tai ranh gioi dong bang).
- HOISTED CONSTANTS: cac ma tran cau truc duoc tao tf.constant MOT LAN
  DUY NHAT truoc vong lap 25 lan, tranh nhan ban 25 lan trong graph.
- BATCH-AGNOSTIC GRAPH: placeholder [None, N, Z], moi reshape dung -1.
- TRAIN/VAL SPLIT + EARLY STOPPING + CHUNKED EVAL: tune_genuine.csv tach
  train/val noi bo. Val danh gia TOAN BO moi epoch, chia lo nho de khong
  tran bo nho. Baseline (epoch=0, truoc fine-tune) duoc ghi lai de so
  sanh truc tiep.
- EXACT_MATCH (quan trong nhat): ngoai bit_correct (trung binh dung/sai
  tung bit tren CA 832 bit codeword, gom ca phan parity khong ai quan
  tam), con theo doi `exact_match` - ty le mau co DU 160 bit message
  (code_k*Z bit DAU cua codeword) dung TOAN BO. Day moi la con so PHAN
  ANH DUNG nhung gi handler.verify() that kiem tra (hash ca block message,
  tat-ca-hoac-khong) - bit_correct cao khong dam bao exact_match cao neu
  loi rai rac moi mau vai bit khac nhau. Viec chon best_epoch/so sanh
  trainable_iters PHAI dua tren exact_match, khong phai bit_correct.

Cach chay:
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

N, m, Z = 52, 42, 16
ITERS_MAX = 25
TRAINABLE_ITERS = 20

ORIGINAL_WEIGHTS_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "Weights_Var_MS"
)
ORIGINAL_BIASES_PATH = os.path.join(
    _PROJECT_ROOT, "wifakey_module", "data", "Biases_Var_MS"
)
OUTPUT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "Weights_Var_MS_finetuned"
)
OUTPUT_BIASES_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "Biases_Var_MS_finetuned"
)

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
N_EPOCHS = 100
PATIENCE = 8
LEARNING_RATE = 1e-4
SCALE = 60.0
MIN_MAG = 0.1
MAX_MAG = 5.0
MASKED_MAG = 1.0


def load_embedding(name, imagenum):
    return np.load(os.path.join(CACHE_DIR, f"{name}_{int(imagenum):04d}.npy"))


def load_tune_genuine_pairs(val_fraction=0.15, seed=123):
    path = os.path.join(PAIRS_DIR, "tune_genuine.csv")
    pairs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emb1 = load_embedding(row["name_enroll"], row["imagenum_enroll"])
            emb2 = load_embedding(row["name_verify"], row["imagenum_verify"])
            pairs.append((emb1, emb2))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(int(len(pairs) * val_fraction), 1)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def build_trainable_decoder(
    init_weights_path,
    init_biases_path,
    trainable_iters=TRAINABLE_ITERS,
    learning_rate=LEARNING_RATE,
):
    assert 0 < trainable_iters <= ITERS_MAX
    freeze_until = ITERS_MAX - trainable_iters

    code_PCM_base = np.loadtxt(BASEGRAPH_PATH, int, delimiter=None)
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
            is_trainable = i >= freeze_until
            net_dict[f"Weights_Var{i}"] = tf.Variable(
                w.copy(), name=f"Weights_Var{i}", trainable=is_trainable
            )
            net_dict[f"Biases_Var{i}"] = tf.Variable(
                b.copy(), name=f"Biases_Var{i}", trainable=is_trainable
            )

        xa = tf.placeholder(tf.float32, shape=[None, N, Z], name="xa")
        target = tf.placeholder(tf.float32, shape=[None, N, Z], name="target")

        xa_input = tf.transpose(xa, [0, 2, 1])

        W_skipconn2even_c = tf.constant(W_skipconn2even, dtype=tf.float32)
        W_odd2even_c = tf.constant(W_odd2even, dtype=tf.float32)
        Lift_M1_T_c = tf.constant(Lift_M1.transpose(), dtype=tf.float32)
        Lift_M2_c = tf.constant(Lift_M2, dtype=tf.float32)
        W_even2odd_input_reshape_c = tf.constant(
            np.reshape(W_even2odd.transpose(), [-1]), dtype=tf.float32
        )
        W_output_c = tf.constant(W_output, dtype=tf.float32)

        net_dict["LLRa0"] = tf.zeros_like(tf.matmul(xa_input, W_skipconn2even_c))

        for i in range(ITERS_MAX):
            x0 = tf.matmul(xa_input, W_skipconn2even_c)
            x1 = tf.matmul(net_dict[f"LLRa{i}"], W_odd2even_c)
            x2 = tf.add(x0, x1)
            x2 = tf.transpose(x2, [0, 2, 1])
            x2 = tf.reshape(x2, [-1, neurons_per_odd_layer * Z])
            x2 = tf.matmul(x2, Lift_M1_T_c)
            x2 = tf.reshape(x2, [-1, neurons_per_odd_layer, Z])
            x2 = tf.transpose(x2, [0, 2, 1])
            x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
            x_tile_mul = tf.multiply(x_tile, W_even2odd_input_reshape_c)
            x2_1 = tf.reshape(
                x_tile_mul, [-1, Z, neurons_per_odd_layer, neurons_per_odd_layer]
            )
            x2_abs = tf.add(
                tf.abs(x2_1), 10000 * (1 - tf.cast(tf.abs(x2_1) > 0, tf.float32))
            )
            x3 = tf.reduce_min(x2_abs, axis=3)
            x2_2 = -x2_1
            x4 = tf.add(tf.zeros_like(x2_1), 1 - 2 * tf.cast(x2_2 < 0, tf.float32))
            x4_prod = -tf.reduce_prod(x4, axis=3)
            x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
            x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
            x_output_0 = tf.reshape(x_output_0, [-1, Z * neurons_per_odd_layer])
            x_output_0 = tf.matmul(x_output_0, Lift_M2_c)
            x_output_0 = tf.reshape(x_output_0, [-1, neurons_per_odd_layer, Z])
            x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
            x_output_1 = tf.add(
                tf.multiply(tf.abs(x_output_0), net_dict[f"Weights_Var{i}"]),
                net_dict[f"Biases_Var{i}"],
            )
            x_output_1 = tf.multiply(x_output_1, tf.cast(x_output_1 > 0, tf.float32))
            llr_next = tf.multiply(x_output_1, tf.sign(x_output_0))

            if i == freeze_until - 1:
                llr_next = tf.stop_gradient(llr_next)

            net_dict[f"LLRa{i+1}"] = llr_next
            y_output_2 = tf.matmul(net_dict[f"LLRa{i+1}"], W_output_c)
            y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
            y_output_4 = tf.add(xa, y_output_3)
            net_dict[f"ya_output{i}"] = tf.reshape(y_output_4, [-1, N * Z])

        decoder_output = net_dict[f"ya_output{ITERS_MAX-1}"]
        target_flat = tf.reshape(target, [-1, N * Z])

        # QUAN TRỌNG: loss gốc chia đều cho cả 832 bit (672 bit parity +
        # 160 bit message) - vì parity chiếm 81% số bit, gradient chủ yếu
        # tối ưu cho ĐÚNG PHẦN KHÔNG QUYẾT ĐỊNH exact_match (verify() chỉ
        # hash 160 bit message, bỏ qua parity hoàn toàn). Thực nghiệm xác
        # nhận: val_exact_match ĐỨNG YÊN qua nhiều epoch dù bit_acc dao
        # động - dấu hiệu rõ ràng của việc loss "lệch trọng tâm" này.
        # Sửa: nhân trọng số MESSAGE_LOSS_WEIGHT cho 160 bit message, giữ
        # nguyên trọng số 1.0 cho phần parity (vẫn cần ít nhiều để decoder
        # hội tụ đúng cấu trúc mã, nhưng không được lấn át phần message).
        code_k = N - m
        key_length = code_k * Z
        MESSAGE_LOSS_WEIGHT = 8.0
        loss_weight_np = np.ones(N * Z, dtype=np.float32)
        loss_weight_np[:key_length] = MESSAGE_LOSS_WEIGHT
        loss_weight_c = tf.constant(loss_weight_np, dtype=tf.float32)

        per_bit_loss = tf.nn.softplus(-target_flat * decoder_output)
        loss = tf.reduce_mean(per_bit_loss * loss_weight_c)
        bit_correct = tf.reduce_mean(
            tf.cast(tf.equal(tf.sign(decoder_output), tf.sign(target_flat)), tf.float32)
        )

        # exact_match: tinh TREN DUNG key_length bit DAU (phan message,
        # code_k*Z = 160 bit) - khop CHINH XAC voi handler.verify() that
        # (chi hash decoded_codeword[:key_length], khong quan tam 672 bit
        # parity con lai). Day la thuoc do QUYET DINH genuine_success that,
        # bit_correct chi la proxy yeu hon (tinh tren ca 832 bit).
        decoder_key_bits = decoder_output[:, :key_length]
        target_key_bits = target_flat[:, :key_length]
        sample_all_correct = tf.reduce_all(
            tf.equal(tf.sign(decoder_key_bits), tf.sign(target_key_bits)), axis=1
        )
        exact_match = tf.reduce_mean(tf.cast(sample_all_correct, tf.float32))

        trainable_vars = [
            net_dict[f"Weights_Var{i}"] for i in range(freeze_until, ITERS_MAX)
        ] + [net_dict[f"Biases_Var{i}"] for i in range(freeze_until, ITERS_MAX)]
        assert len(trainable_vars) == 2 * trainable_iters

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, var_list=trainable_vars)

        init_op = tf.global_variables_initializer()

    return {
        "graph": graph,
        "xa": xa,
        "target": target,
        "loss": loss,
        "bit_correct": bit_correct,
        "exact_match": exact_match,
        "train_op": train_op,
        "init_op": init_op,
        "net_dict": net_dict,
        "freeze_until": freeze_until,
        "trainable_iters": trainable_iters,
    }


def _encode_pair(handler, encoder, modulation, emb_enroll, emb_verify, rng):
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
    return llr.reshape(N, Z), target_bipolar.reshape(N, Z)


def make_training_batch(handler, encoder, modulation, pairs, batch_size, rng):
    llr_batch = np.zeros((batch_size, N, Z), dtype=np.float32)
    target_batch = np.zeros((batch_size, N, Z), dtype=np.float32)

    idxs = rng.choice(len(pairs), size=batch_size, replace=True)
    for b, idx in enumerate(idxs):
        emb_enroll, emb_verify = pairs[idx]
        llr, target_bipolar = _encode_pair(
            handler, encoder, modulation, emb_enroll, emb_verify, rng
        )
        llr_batch[b] = llr
        target_batch[b] = target_bipolar

    return llr_batch, target_batch


def make_eval_batch(handler, encoder, modulation, pairs, rng):
    n = len(pairs)
    llr_batch = np.zeros((n, N, Z), dtype=np.float32)
    target_batch = np.zeros((n, N, Z), dtype=np.float32)

    for b, (emb_enroll, emb_verify) in enumerate(pairs):
        llr, target_bipolar = _encode_pair(
            handler, encoder, modulation, emb_enroll, emb_verify, rng
        )
        llr_batch[b] = llr
        target_batch[b] = target_bipolar

    return llr_batch, target_batch


def evaluate_val_chunked(sess, tg, val_llr, val_target, eval_chunk_size):
    """Tra ve (val_loss, val_bit_acc, val_exact_match) - trung binh co
    trong so theo so mau that cua tung lo."""
    n = val_llr.shape[0]
    fetches = [tg["loss"], tg["bit_correct"], tg["exact_match"]]

    if n <= eval_chunk_size:
        loss_v, bit_v, exact_v = sess.run(
            fetches, feed_dict={tg["xa"]: val_llr, tg["target"]: val_target}
        )
        return float(loss_v), float(bit_v), float(exact_v)

    losses, bits, exacts, sizes = [], [], [], []
    for start in range(0, n, eval_chunk_size):
        end = min(start + eval_chunk_size, n)
        loss_v, bit_v, exact_v = sess.run(
            fetches,
            feed_dict={
                tg["xa"]: val_llr[start:end],
                tg["target"]: val_target[start:end],
            },
        )
        losses.append(loss_v)
        bits.append(bit_v)
        exacts.append(exact_v)
        sizes.append(end - start)

    w = np.array(sizes, dtype=np.float64)
    return (
        float(np.average(losses, weights=w)),
        float(np.average(bits, weights=w)),
        float(np.average(exacts, weights=w)),
    )


def train_one_config(
    trainable_iters,
    n_epochs=N_EPOCHS,
    patience=PATIENCE,
    verbose=True,
    save_weights=False,
    eval_chunk_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
):
    """Train 1 cau hinh. QUAN TRONG: best_epoch/early-stopping gio dua
    tren `exact_match` (ty le mau dung TOAN BO 160 bit message), KHONG
    con dua tren `bit_correct` (trung binh tung bit tren ca 832 bit) -
    vi exact_match moi la con so khop voi genuine_success that.
    """
    handler = WiFaKeyHandler()
    encoder = Encode.Proto_LDPC(N, m, Z)
    modulation = SoftDistanceLLR(
        scale=SCALE, min_mag=MIN_MAG, max_mag=MAX_MAG, masked_mag=MASKED_MAG
    )

    train_pairs, val_pairs = load_tune_genuine_pairs()
    if verbose:
        print(
            f"train={len(train_pairs)} cap, val={len(val_pairs)} cap "
            f"(val KHONG tinh gradient, danh gia TOAN BO moi epoch)"
        )

    tg = build_trainable_decoder(
        ORIGINAL_WEIGHTS_PATH,
        ORIGINAL_BIASES_PATH,
        trainable_iters=trainable_iters,
        learning_rate=learning_rate,
    )
    if verbose:
        print(
            f"-> Dong bang vong 0..{tg['freeze_until']-1}, fine-tune vong "
            f"{tg['freeze_until']}..{ITERS_MAX-1}, learning_rate={learning_rate:g}."
        )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    rng = np.random.default_rng(0)

    val_llr, val_target = make_eval_batch(
        handler, encoder, modulation, val_pairs, np.random.default_rng(999)
    )

    best_val_exact, best_epoch, best_weights = -1.0, -1, None
    epochs_since_improve = 0
    history = []

    with tf.Session(graph=tg["graph"], config=config) as sess:
        sess.run(tg["init_op"])

        baseline_loss, baseline_bit, baseline_exact = evaluate_val_chunked(
            sess, tg, val_llr, val_target, eval_chunk_size
        )
        history.append(
            {
                "epoch": 0,
                "train_acc": None,
                "val_bit_acc": baseline_bit,
                "val_exact_match": baseline_exact,
                "val_loss": baseline_loss,
            }
        )
        if verbose:
            print(
                f"  epoch   0 (baseline, truoc fine-tune): "
                f"val_bit_acc={baseline_bit:.4f}  val_exact_match={baseline_exact:.4f}  "
                f"val_loss={baseline_loss:.4f}"
            )
        best_val_exact = baseline_exact

        steps_per_epoch = max(len(train_pairs) // BATCH_SIZE, 1)

        for epoch in range(n_epochs):
            train_loss_list, train_acc_list = [], []
            for _ in range(steps_per_epoch):
                llr_batch, target_batch = make_training_batch(
                    handler, encoder, modulation, train_pairs, BATCH_SIZE, rng
                )
                _, loss_val, acc_val = sess.run(
                    [tg["train_op"], tg["loss"], tg["bit_correct"]],
                    feed_dict={tg["xa"]: llr_batch, tg["target"]: target_batch},
                )
                train_loss_list.append(loss_val)
                train_acc_list.append(acc_val)

            val_loss, val_bit, val_exact = evaluate_val_chunked(
                sess, tg, val_llr, val_target, eval_chunk_size
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_acc": float(np.mean(train_acc_list)),
                    "val_bit_acc": val_bit,
                    "val_exact_match": val_exact,
                    "val_loss": val_loss,
                }
            )
            if verbose:
                print(
                    f"  epoch {epoch+1:3d}: train_bit_acc={np.mean(train_acc_list):.4f}  "
                    f"val_bit_acc={val_bit:.4f}  val_exact_match={val_exact:.4f}  "
                    f"val_loss={val_loss:.4f}"
                )

            if val_exact > best_val_exact:
                best_val_exact, best_epoch, epochs_since_improve = (
                    val_exact,
                    epoch + 1,
                    0,
                )
                if save_weights:
                    best_weights = {
                        i: (
                            sess.run(tg["net_dict"][f"Weights_Var{i}"]),
                            sess.run(tg["net_dict"][f"Biases_Var{i}"]),
                        )
                        for i in range(ITERS_MAX)
                    }
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    if verbose:
                        print(
                            f"  -> Dung som: val_exact_match khong cai thien sau {patience} epoch."
                        )
                    break

        if save_weights and best_weights is not None:
            os.makedirs(OUTPUT_WEIGHTS_PATH, exist_ok=True)
            os.makedirs(OUTPUT_BIASES_PATH, exist_ok=True)
            for i in range(ITERS_MAX):
                w_val, b_val = best_weights[i]
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
            if verbose:
                print(
                    f"  -> Da luu trong so tai epoch {best_epoch} (val_exact_match={best_val_exact:.4f})"
                )

    return best_val_exact, best_epoch, history


def main():
    print(f"=== Train voi TRAINABLE_ITERS={TRAINABLE_ITERS} ===")
    best_val_exact, best_epoch, history = train_one_config(
        trainable_iters=TRAINABLE_ITERS,
        n_epochs=N_EPOCHS,
        patience=PATIENCE,
        verbose=True,
        save_weights=True,
    )
    print(
        f"\nHoan tat. Val exact_match tot nhat: {best_val_exact:.4f} tai epoch {best_epoch}."
    )
    print(f"   (Baseline truoc fine-tune: {history[0]['val_exact_match']:.4f})")
    print(f"   Trong so moi luu tai: {OUTPUT_WEIGHTS_PATH}")
    print(f"   (Trong so GOC khong he bi dung vao)")


if __name__ == "__main__":
    main()
