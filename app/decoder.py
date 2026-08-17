import sys
import numpy as np
import tensorflow.compat.v1 as tf
from pathlib import Path

# Thêm đường dẫn để import wifakey_module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wifakey_module"))

try:
    from wifakey_module.wifakey_handler import Decoder
    from wifakey_module.base_graph import BaseGraph
except ImportError as e:
    raise ImportError("Không thể import wifakey_module. Hãy đảm bảo thư mục wifakey_module có trong cùng cấp.") from e


class DecoderService:
    """
    Wrapper cho decoder Neural-MS, chỉ cần khởi tạo một lần và dùng lại.
    """
    def __init__(self, weights_dir: str = None):
        self.graph = tf.Graph()
        self.sess = None
        self.input_placeholder = None
        self.decoded_bits = None

        # Khởi tạo base graph (N=52, M=42, Z=16)
        self.base_graph = BaseGraph(N=52, M=42, Z=16)  # tuỳ chỉnh nếu cần

        # Load weights (mặc định lấy từ wifakey_module/weights/)
        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parent.parent / "wifakey_module" / "weights"
        self.weights_dir = Path(weights_dir)

        self._build_graph()
        self._init_session()

    def _build_graph(self):
        with self.graph.as_default():
            # Input: LLR dạng BPSK (±1), shape (batch, N, Z)
            self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.base_graph.N, self.base_graph.Z), name="input_llr")

            # Tạo decoder theo cấu trúc Neural-MS
            # Sử dụng class Decoder từ wifakey_module
            # Giả định Decoder có method build_graph(input, weights, biases)
            # Nếu không, ta sẽ tự xây dựng ở đây (xem phần fallback bên dưới)
            self.decoder = Decoder(self.base_graph, self.weights_dir)
            self.decoded_bits = self.decoder.decode(self.input_placeholder)  # trả về bits (0/1) shape (batch, N*Z)

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(tf.global_variables_initializer())

    def decode(self, llr: np.ndarray) -> np.ndarray:
        """
        Nhận LLR (mảng 832 số thực) -> trả về key bits (160 bit) dạng numpy int8 (0/1)
        """
        # Reshape về (1, 52, 16)
        llr_arr = np.array(llr, dtype=np.float32).reshape(1, self.base_graph.N, self.base_graph.Z)
        # Nếu LLR là số thực (soft), ta có thể hard-decision trước khi đưa vào decoder (vì decoder gốc dùng hard BPSK)
        # Hoặc nếu decoder đã được fine-tune cho soft LLR, ta có thể truyền trực tiếp.
        # Ở đây, ta giả định decoder gốc chỉ nhận ±1, nên chuyển hard:
        hard_bits = (llr_arr >= 0).astype(np.int32)  # bit 0 nếu LLR>=0, else 1
        bpsk = 1 - 2 * hard_bits  # 0->1, 1->-1
        feed = {self.input_placeholder: bpsk.astype(np.float32)}

        # Decode
        decoded_codeword = self.sess.run(self.decoded_bits, feed_dict=feed)  # shape (1, 832)
        # Lấy 160 bit đầu tiên làm key
        key_bits = decoded_codeword[0, :160]  # mảng int (0/1)
        return key_bits.astype(np.uint8)

    def close(self):
        if self.sess:
            self.sess.close()