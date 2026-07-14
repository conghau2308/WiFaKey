import numpy as np
import os
import tensorflow.compat.v1 as tf
import hashlib
from .wifakey_lib import Encode, Modulation, utils 

tf.disable_v2_behavior() 

class WiFaKeyHandler:
    
    def __init__(self, data_path='wifakey_module/data', weights_path='wifakey_module/data/Weights_Var_MS', biases_path='wifakey_module/data/Biases_Var_MS'):
        print("[WiFaKey] Initializing... Please wait.")
        
        self.N = 52
        self.m = 42
        self.Z = 16 
        self.code_n = self.N
        self.code_k = self.N - self.m
        self.iters_max = 25 
        self.batch_size = 1 
        self.feature_length = self.code_n * self.Z # 832
        self.key_length = self.code_k * self.Z # 160
        
        # Original paper params for Z=16 (1536 bits -> 2 blocks of 512)
        # Github original repo uses 512 bit blocks, with 320 zeros padding for Z=16
        # self.num_blocks = 2
        # self.block_raw_bits = 512
        # self.pad_bits = self.feature_length - self.block_raw_bits
        
        self.data_path = data_path
        self.weights_path = weights_path
        self.biases_path = biases_path
        
        # kappa_path = os.path.join(self.data_path, 'kappa.npy')
        # if os.path.exists(kappa_path):
        #     self.kappa = float(np.load(kappa_path))
        # else:
        #     self.kappa = 0.0
        # # Lower κ ⇒ more mask-1 bits ⇒ stronger impostor separation (lower FAR), higher FRR.
        # # Tune empirically, e.g. WIFAKEY_KAPPA_SUB=0.02 after re-calibration if FAR > 0.

        # # kappa_sub = os.environ.get("WIFAKEY_KAPPA_SUB")
        # kappa_sub = 0.0275
        self.kappa = 0.3125
        # if kappa_sub is not None:
        #     self.kappa = float(np.clip(self.kappa - float(kappa_sub), 0.0, 0.999))
        #     print(f"[WiFaKey] κ after WIFAKEY_KAPPA_SUB: {self.kappa}")

        print(f"[WiFaKey] Loading LDPC encoder...")
        self.encoder = Encode.Proto_LDPC(self.N, self.m, self.Z)

        m_matrix_path = os.path.join(self.data_path, 'M_matrix.npy')
        if not os.path.exists(m_matrix_path):
            raise FileNotFoundError(f"Missing M_matrix.npy in {self.data_path}.")
        self.M_matrix = np.load(m_matrix_path)
        
        interval_path = os.path.join(self.data_path, 'binarization_intervals.npy')
        self.intervals = np.load(interval_path)
        # Length of flattened LSSC output (mask_r length); used by API validation
        n_thr = int(np.asarray(self.intervals).size)
        self.full_binary_length = self.M_matrix.shape[0] * n_thr
        
        pcm_path = os.path.join(self.data_path, 'BaseGraph', 'BaseGraph2_Set0.txt')
        self.code_PCM_base = np.loadtxt(pcm_path, int, delimiter=None)

        print(f"[WiFaKey] Building Neural-MS (TF1.x)...")
        self._build_decoder()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        
        print("[WiFaKey] Initializing global variables (loading weights)...")
        self.sess.run(self.init_op)

        print("[WiFaKey] Starting decoder...")
        dummy_bits = np.random.randint(0, 2, (1, 52*16)).astype(np.uint8)
        dummy_llr = (2 * dummy_bits - 1).astype(np.float32).reshape((1, 52, 16))
        _ = self.sess.run(self.decoder_output, feed_dict={self.xa: dummy_llr})
        print("[WiFaKey] Decoder started successfully.")
        print("[WiFaKey] All set. Ready.")

    def __del__(self):
        if hasattr(self, 'sess'):
            self.sess.close()
            print("[WiFaKey] Session closed.")

    def _build_decoder(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            code_PCM = self.code_PCM_base.copy()
            for i in range(0, code_PCM.shape[0]):
                for j in range(0, code_PCM.shape[1]):
                    if (code_PCM[i, j] == -1): code_PCM[i, j] = 0
                    else: code_PCM[i, j] = 1
            sum_edge_c = np.sum(code_PCM, axis=1)
            sum_edge_v = np.sum(code_PCM, axis=0)
            sum_edge = np.sum(sum_edge_v)
            W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
            W_skipconn2even = np.zeros((self.N, sum_edge), dtype=np.float32)
            W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
            W_output = np.zeros((sum_edge, self.N), dtype=np.float32)
            k = 0
            for j in range(0, code_PCM.shape[1], 1):
                for i in range(0, code_PCM.shape[0], 1):
                    if (code_PCM[i, j] == 1):
                        num_of_conn = int(np.sum(code_PCM[:, j]))
                        idx = np.argwhere(code_PCM[:, j] == 1)
                        for l in range(0, num_of_conn, 1):
                            vec_tmp = np.zeros((sum_edge), dtype=np.float32)
                            for r in range(0, code_PCM.shape[0], 1):
                                if (code_PCM[r, j] == 1 and idx[l][0] != r):
                                    idx_row = np.cumsum(code_PCM[r, 0:j + 1])[-1] - 1
                                    odd_layer_node_count = 0
                                    if r > 0:
                                        odd_layer_node_count = np.cumsum(sum_edge_c[0:r])[-1]
                                    vec_tmp[idx_row + odd_layer_node_count] = 1
                            W_odd2even[:, k] = vec_tmp.transpose()
                            k += 1
                        break
            k = 0
            for j in range(0, code_PCM.shape[1], 1):
                for i in range(0, code_PCM.shape[0], 1):
                    if (code_PCM[i, j] == 1):
                        idx_row = np.cumsum(code_PCM[i, 0:j + 1])[-1] - 1
                        odd_layer_node_count_1 = 0
                        odd_layer_node_count_2 = np.cumsum(sum_edge_c[0:i + 1])[-1]
                        if i > 0:
                            odd_layer_node_count_1 = np.cumsum(sum_edge_c[0:i])[-1]
                        W_even2odd[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
                        W_even2odd[k, odd_layer_node_count_1 + idx_row] = 0.0
                        k += 1
            k = 0
            for j in range(0, code_PCM.shape[1], 1):
                for i in range(0, code_PCM.shape[0], 1):
                    if (code_PCM[i, j] == 1):
                        idx_row = np.cumsum(code_PCM[i, 0:j + 1])[-1] - 1
                        odd_layer_node_count = 0
                        if i > 0:
                            odd_layer_node_count = np.cumsum(sum_edge_c[0:i])[-1]
                        W_output[odd_layer_node_count + idx_row, k] = 1.0
                k += 1
            k = 0
            for j in range(0, code_PCM.shape[1], 1):
                for i in range(0, code_PCM.shape[0], 1):
                    if (code_PCM[i, j] == 1):
                        W_skipconn2even[j, k] = 1.0
                        k += 1
            code_PCM1 = self.code_PCM_base
            Z_array = np.array([16, 3, 10, 6])
            neurons_per_odd_layer = np.sum(sum_edge_v)
            Lift_M1 = np.zeros((neurons_per_odd_layer * Z_array[0], neurons_per_odd_layer * Z_array[0]), np.float32)
            Lift_M2 = np.zeros((neurons_per_odd_layer * Z_array[0], neurons_per_odd_layer * Z_array[0]), np.float32)
            k = 0
            for j in range(0, code_PCM1.shape[1]):
                for i in range(0, code_PCM1.shape[0]):
                    if (code_PCM1[i, j] != -1):
                        Lift_num = code_PCM1[i, j] % Z_array[0]
                        for h in range(0, Z_array[0], 1):
                            Lift_M1[k * Z_array[0] + h, k * Z_array[0] + (h + Lift_num) % Z_array[0]] = 1
                        k = k + 1
            k = 0
            for i in range(0, code_PCM1.shape[0]):
                for j in range(0, code_PCM1.shape[1]):
                    if (code_PCM1[i, j] != -1):
                        Lift_num = code_PCM1[i, j] % Z_array[0]
                        for h in range(0, Z_array[0], 1):
                            Lift_M2[k * Z_array[0] + h, k * Z_array[0] + (h + Lift_num) % Z_array[0]] = 1
                        k = k + 1
            net_dict = {}
            for i in range(0, self.iters_max, 1):
                w_path = os.path.join(self.weights_path, f'Weights_Var{i}.txt')
                b_path = os.path.join(self.biases_path, f'Biases_Var{i}.txt')
                w = np.loadtxt(w_path, delimiter=',', dtype=np.float32)
                b = np.loadtxt(b_path, delimiter=',', dtype=np.float32)
                net_dict[f"Weights_Var{i}"] = tf.Variable(w.copy(), name=f"Weights_Var{i}")
                net_dict[f"Biases_Var{i}"] = tf.Variable(b.copy(), name=f"Biases_Var{i}")
            self.xa = tf.placeholder(tf.float32, shape=[self.batch_size, self.N, self.Z], name='xa')
            xa_input = tf.transpose(self.xa, [0, 2, 1])
            net_dict["LLRa{0}".format(0)] = tf.zeros((self.batch_size, self.Z, sum_edge), dtype=tf.float32)
            for i in range(0, self.iters_max, 1):
                x0 = tf.matmul(xa_input, W_skipconn2even)
                x1 = tf.matmul(net_dict["LLRa{0}".format(i)], W_odd2even)
                x2 = tf.add(x0, x1)
                x2 = tf.transpose(x2, [0, 2, 1])
                x2 = tf.reshape(x2, [self.batch_size, neurons_per_odd_layer * self.Z])
                x2 = tf.matmul(x2, Lift_M1.transpose())
                x2 = tf.reshape(x2, [self.batch_size, neurons_per_odd_layer, self.Z])
                x2 = tf.transpose(x2, [0, 2, 1])
                x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
                W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
                x_tile_mul = tf.multiply(x_tile, W_input_reshape)
                x2_1 = tf.reshape(x_tile_mul, [self.batch_size, self.Z, neurons_per_odd_layer, neurons_per_odd_layer])
                x2_abs = tf.add(tf.abs(x2_1), 10000 * (1 - tf.to_float(tf.abs(x2_1) > 0)))
                x3 = tf.reduce_min(x2_abs, axis=3)
                x2_2 = -x2_1
                x4 = tf.add(tf.zeros((self.batch_size, self.Z, neurons_per_odd_layer, neurons_per_odd_layer)), 1 - 2 * tf.to_float(x2_2 < 0))
                x4_prod = -tf.reduce_prod(x4, axis=3)
                x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
                x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
                x_output_0 = tf.reshape(x_output_0, [self.batch_size, self.Z * neurons_per_odd_layer])
                x_output_0 = tf.matmul(x_output_0, Lift_M2)
                x_output_0 = tf.reshape(x_output_0, [self.batch_size, neurons_per_odd_layer, self.Z])
                x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
                x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0),net_dict["Weights_Var{0}".format(i)]), net_dict["Biases_Var{0}".format(i)])
                x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
                net_dict["LLRa{0}".format(i+1)] = tf.multiply(x_output_1, tf.sign(x_output_0))
                y_output_2 = tf.matmul(net_dict["LLRa{0}".format(i+1)], W_output)
                y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
                y_output_4 = tf.add(self.xa, y_output_3)
                net_dict["ya_output{0}".format(i)] = tf.reshape(y_output_4, [self.batch_size, self.N * self.Z], name='ya_output'.format(i))
            self.decoder_output = net_dict[f"ya_output{self.iters_max-1}"]
            self.init_op = tf.global_variables_initializer()

    def _binarize_full(self, feature_vector_float: np.ndarray) -> np.ndarray:
        projected = np.dot(feature_vector_float, self.M_matrix)

        feature_vector_2d = np.expand_dims(projected, axis=0)

        binary_vector_expanded = utils.lssc_binary(feature_vector_2d, interval=self.intervals).flatten()
        return binary_vector_expanded


    def enroll(self, feature_vector_float: np.ndarray) -> tuple[np.ndarray, np.ndarray, bytes]:

        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        # Paper's masking (bitwise AND)
        u = np.random.uniform(0.0, 1.0, size=len(b_full))
        mask_r = (u >= self.kappa).astype(np.uint8)
        b_masked = (b_full & mask_r).astype(np.uint8)

        if len(b_masked) < self.feature_length:
            raise ValueError("Not enough bits after masking")

        b_selected = b_masked[:self.feature_length]

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)

        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)

        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        return helper_data, mask_r, key_hash
        

    def verify(self, feature_vector_float: np.ndarray, helper_data: np.ndarray, mask_r: np.ndarray, stored_key_hash: bytes) -> bool:

        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        b_masked = (b_full & mask_r).astype(np.uint8)
        b_selected = b_masked[:self.feature_length]
        y_noisy_bits = np.logical_xor(b_selected, helper_data)

        y_llr = Modulation.BPSK(y_noisy_bits).reshape((1, self.N, self.Z))
        y_pred_llr = self.sess.run(self.decoder_output, feed_dict={self.xa: y_llr})

        decoded_codeword = (y_pred_llr > 0).astype(int).flatten()
        reconstructed_key = decoded_codeword[:self.key_length]
        recon_hash = hashlib.sha256(reconstructed_key.tobytes()).digest()

        if recon_hash == stored_key_hash:
            print("[WiFaKey] Verify SUCCESS.")
            return True
        else:
            print("[WiFaKey] Verify FAILED.")
            return False