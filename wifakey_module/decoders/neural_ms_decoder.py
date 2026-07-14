"""Neural Min-Sum (5G NR LDPC) decoder wrapper.

Default trained parameters are ``Weights_Var_MS`` / ``Biases_Var_MS``
under ``data_path``.

Re-builds the same TensorFlow 1.x graph used by ``WiFaKeyHandler`` so we can
benchmark the pure error-correction capability without paying for the
biometric-side machinery (M_matrix, LSSC binarization, kappa masking, ...).

Convention (mirrors the original handler):
    * Input LLR ``xa`` is shaped (1, N=52, Z=16) with values in {-1, +1}.
    * ``-1``  ↔ bit 0   (matches ``2 * bit - 1`` / ``Modulation.BPSK``)
    * ``+1``  ↔ bit 1
    * Output LLR > 0  ↔ decoded bit 1.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import tensorflow.compat.v1 as tf

from ..wifakey_lib import Encode

tf.disable_v2_behavior()


_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)


class NeuralMSDecoder:
    name = "neural_ms"

    def __init__(
        self,
        data_path: str = _DEFAULT_DATA_DIR,
        weights_path: Optional[str] = None,
        biases_path: Optional[str] = None,
        iters_max: int = 25,
        force_cpu: bool = False,
    ) -> None:
        self.N = 52
        self.m = 42
        self.Z = 16
        self.iters_max = iters_max
        self.batch_size = 1
        self.n = self.N * self.Z          # 832
        self.k = (self.N - self.m) * self.Z  # 160
        self.rate = self.k / self.n

        self.data_path = data_path
        default_weights = os.path.join(data_path, "Weights_Var_MS")
        default_biases = os.path.join(data_path, "Biases_Var_MS")

        self.weights_path = (
            weights_path if weights_path is not None else default_weights
        )
        self.biases_path = (
            biases_path if biases_path is not None else default_biases
        )
        if not os.path.isdir(self.weights_path):
            raise FileNotFoundError(
                f"Neural-MS weights directory not found: {self.weights_path}"
            )
        if not os.path.isdir(self.biases_path):
            raise FileNotFoundError(
                f"Neural-MS biases directory not found: {self.biases_path}"
            )

        pcm_path = os.path.join(data_path, "BaseGraph", "BaseGraph2_Set0.txt")
        self.code_PCM_base = np.loadtxt(pcm_path, int, delimiter=None)

        self.encoder = Encode.Proto_LDPC(self.N, self.m, self.Z)

        self._build_decoder()

        config = tf.ConfigProto()
        if force_cpu:
            config = tf.ConfigProto(device_count={"GPU": 0})
        else:
            config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.init_op)

        # Warm-up so that the first measured decode is not penalized.
        dummy_bits = np.zeros((1, self.n), dtype=np.uint8)
        dummy_llr = (2 * dummy_bits - 1).astype(np.float32).reshape((1, self.N, self.Z))
        _ = self.sess.run(self.decoder_output, feed_dict={self.xa: dummy_llr})

    def __del__(self) -> None:
        try:
            if hasattr(self, "sess") and self.sess is not None:
                self.sess.close()
        except Exception:
            pass

    def encode(self, msg_bits: np.ndarray) -> np.ndarray:
        msg_bits = np.asarray(msg_bits, dtype=np.int64).reshape(1, self.k)
        codeword = self.encoder.encode_LDPC(msg_bits).flatten().astype(np.uint8)
        return codeword

    def decode(self, received_bits: np.ndarray) -> np.ndarray:
        received_bits = np.asarray(received_bits, dtype=np.uint8).reshape(self.n)
        llr = (2 * received_bits - 1).astype(np.float32).reshape(1, self.N, self.Z)
        y_pred = self.sess.run(self.decoder_output, feed_dict={self.xa: llr})
        decoded = (y_pred > 0).astype(np.uint8).flatten()
        return decoded[: self.k]

    def _build_decoder(self) -> None:
        self.graph = tf.Graph()
        with self.graph.as_default():
            code_PCM = self.code_PCM_base.copy()
            for i in range(0, code_PCM.shape[0]):
                for j in range(0, code_PCM.shape[1]):
                    if code_PCM[i, j] == -1:
                        code_PCM[i, j] = 0
                    else:
                        code_PCM[i, j] = 1
            sum_edge_c = np.sum(code_PCM, axis=1)
            sum_edge_v = np.sum(code_PCM, axis=0)
            sum_edge = int(np.sum(sum_edge_v))

            W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
            W_skipconn2even = np.zeros((self.N, sum_edge), dtype=np.float32)
            W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
            W_output = np.zeros((sum_edge, self.N), dtype=np.float32)

            k = 0
            for j in range(code_PCM.shape[1]):
                for i in range(code_PCM.shape[0]):
                    if code_PCM[i, j] == 1:
                        num_of_conn = int(np.sum(code_PCM[:, j]))
                        idx = np.argwhere(code_PCM[:, j] == 1)
                        for ll in range(num_of_conn):
                            vec_tmp = np.zeros(sum_edge, dtype=np.float32)
                            for r in range(code_PCM.shape[0]):
                                if code_PCM[r, j] == 1 and idx[ll][0] != r:
                                    idx_row = np.cumsum(code_PCM[r, 0 : j + 1])[-1] - 1
                                    odd_layer_node_count = 0
                                    if r > 0:
                                        odd_layer_node_count = np.cumsum(sum_edge_c[0:r])[-1]
                                    vec_tmp[idx_row + odd_layer_node_count] = 1
                            W_odd2even[:, k] = vec_tmp.transpose()
                            k += 1
                        break

            k = 0
            for j in range(code_PCM.shape[1]):
                for i in range(code_PCM.shape[0]):
                    if code_PCM[i, j] == 1:
                        idx_row = np.cumsum(code_PCM[i, 0 : j + 1])[-1] - 1
                        odd_layer_node_count_1 = 0
                        odd_layer_node_count_2 = np.cumsum(sum_edge_c[0 : i + 1])[-1]
                        if i > 0:
                            odd_layer_node_count_1 = np.cumsum(sum_edge_c[0:i])[-1]
                        W_even2odd[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
                        W_even2odd[k, odd_layer_node_count_1 + idx_row] = 0.0
                        k += 1

            k = 0
            for j in range(code_PCM.shape[1]):
                for i in range(code_PCM.shape[0]):
                    if code_PCM[i, j] == 1:
                        idx_row = np.cumsum(code_PCM[i, 0 : j + 1])[-1] - 1
                        odd_layer_node_count = 0
                        if i > 0:
                            odd_layer_node_count = np.cumsum(sum_edge_c[0:i])[-1]
                        W_output[odd_layer_node_count + idx_row, k] = 1.0
                k += 1

            k = 0
            for j in range(code_PCM.shape[1]):
                for i in range(code_PCM.shape[0]):
                    if code_PCM[i, j] == 1:
                        W_skipconn2even[j, k] = 1.0
                        k += 1

            code_PCM1 = self.code_PCM_base
            Z_array = np.array([16, 3, 10, 6])
            neurons_per_odd_layer = int(np.sum(sum_edge_v))
            Lift_M1 = np.zeros(
                (neurons_per_odd_layer * Z_array[0], neurons_per_odd_layer * Z_array[0]),
                np.float32,
            )
            Lift_M2 = np.zeros(
                (neurons_per_odd_layer * Z_array[0], neurons_per_odd_layer * Z_array[0]),
                np.float32,
            )

            k = 0
            for j in range(code_PCM1.shape[1]):
                for i in range(code_PCM1.shape[0]):
                    if code_PCM1[i, j] != -1:
                        Lift_num = code_PCM1[i, j] % Z_array[0]
                        for h in range(Z_array[0]):
                            Lift_M1[
                                k * Z_array[0] + h,
                                k * Z_array[0] + (h + Lift_num) % Z_array[0],
                            ] = 1
                        k += 1

            k = 0
            for i in range(code_PCM1.shape[0]):
                for j in range(code_PCM1.shape[1]):
                    if code_PCM1[i, j] != -1:
                        Lift_num = code_PCM1[i, j] % Z_array[0]
                        for h in range(Z_array[0]):
                            Lift_M2[
                                k * Z_array[0] + h,
                                k * Z_array[0] + (h + Lift_num) % Z_array[0],
                            ] = 1
                        k += 1

            net_dict = {}
            for i in range(self.iters_max):
                w_path = os.path.join(self.weights_path, f"Weights_Var{i}.txt")
                b_path = os.path.join(self.biases_path, f"Biases_Var{i}.txt")
                w = np.loadtxt(w_path, delimiter=",", dtype=np.float32)
                b = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
                net_dict[f"Weights_Var{i}"] = tf.Variable(w.copy(), name=f"Weights_Var{i}")
                net_dict[f"Biases_Var{i}"] = tf.Variable(b.copy(), name=f"Biases_Var{i}")

            self.xa = tf.placeholder(
                tf.float32, shape=[self.batch_size, self.N, self.Z], name="xa"
            )
            xa_input = tf.transpose(self.xa, [0, 2, 1])
            net_dict["LLRa{0}".format(0)] = tf.zeros(
                (self.batch_size, self.Z, sum_edge), dtype=tf.float32
            )

            for i in range(self.iters_max):
                x0 = tf.matmul(xa_input, W_skipconn2even)
                x1 = tf.matmul(net_dict[f"LLRa{i}"], W_odd2even)
                x2 = tf.add(x0, x1)
                x2 = tf.transpose(x2, [0, 2, 1])
                x2 = tf.reshape(x2, [self.batch_size, neurons_per_odd_layer * self.Z])
                x2 = tf.matmul(x2, Lift_M1.transpose())
                x2 = tf.reshape(x2, [self.batch_size, neurons_per_odd_layer, self.Z])
                x2 = tf.transpose(x2, [0, 2, 1])
                x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
                W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
                x_tile_mul = tf.multiply(x_tile, W_input_reshape)
                x2_1 = tf.reshape(
                    x_tile_mul,
                    [self.batch_size, self.Z, neurons_per_odd_layer, neurons_per_odd_layer],
                )
                x2_abs = tf.add(tf.abs(x2_1), 10000 * (1 - tf.to_float(tf.abs(x2_1) > 0)))
                x3 = tf.reduce_min(x2_abs, axis=3)
                x2_2 = -x2_1
                x4 = tf.add(
                    tf.zeros(
                        (self.batch_size, self.Z, neurons_per_odd_layer, neurons_per_odd_layer)
                    ),
                    1 - 2 * tf.to_float(x2_2 < 0),
                )
                x4_prod = -tf.reduce_prod(x4, axis=3)
                x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
                x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
                x_output_0 = tf.reshape(
                    x_output_0, [self.batch_size, self.Z * neurons_per_odd_layer]
                )
                x_output_0 = tf.matmul(x_output_0, Lift_M2)
                x_output_0 = tf.reshape(
                    x_output_0, [self.batch_size, neurons_per_odd_layer, self.Z]
                )
                x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
                x_output_1 = tf.add(
                    tf.multiply(tf.abs(x_output_0), net_dict[f"Weights_Var{i}"]),
                    net_dict[f"Biases_Var{i}"],
                )
                x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
                net_dict[f"LLRa{i+1}"] = tf.multiply(x_output_1, tf.sign(x_output_0))
                y_output_2 = tf.matmul(net_dict[f"LLRa{i+1}"], W_output)
                y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
                y_output_4 = tf.add(self.xa, y_output_3)
                net_dict[f"ya_output{i}"] = tf.reshape(
                    y_output_4, [self.batch_size, self.N * self.Z], name=f"ya_output{i}"
                )

            self.decoder_output = net_dict[f"ya_output{self.iters_max - 1}"]
            self.init_op = tf.global_variables_initializer()
