"""
export_neural_ms.py

Xuất mô hình Neural‑MS decoder sang ONNX.
Yêu cầu: tensorflow 1.x, tf2onnx (cài bằng pip install tf2onnx)
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ép chạy trên CPU, tránh OOM GPU 4GB

import sys
import gc
import numpy as np
import tensorflow.compat.v1 as tf
import tf2onnx
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import optimizer as tf2onnx_optimizer

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler


def main():
    # Khởi tạo handler để xây dựng đồ thị (graph) của Neural‑MS
    handler = WiFaKeyHandler(
        data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
    )

    # Lấy tensor input và output từ handler
    input_tensor = handler.xa
    output_tensor = handler.decoder_output
    input_name = input_tensor.name
    output_name = output_tensor.name

    onnx_path = os.path.join(os.path.dirname(__file__), "neural_ms.onnx")

    # --- Kiểm tra inference mẫu TRƯỚC khi đóng session ---
    dummy_input = np.random.randn(1, 52, 16).astype(np.float32)
    original_output = handler.sess.run(
        output_tensor, feed_dict={input_tensor: dummy_input}
    )
    print(f"Output gốc (5 giá trị đầu): {original_output.flatten()[:5]}")

    # --- Đóng băng graph: chuyển Variables thành Constants ---
    output_node_name = output_name.split(":")[0]
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        handler.sess,
        handler.sess.graph.as_graph_def(),
        output_node_names=[output_node_name],
    )

    # Giải phóng session gốc trước khi convert, tránh giữ nhiều bản graph trong RAM
    handler.sess.close()
    del handler
    gc.collect()

    # --- Export sang ONNX, BỎ QUA bước TF Grappler optimize (chỗ gây MemoryError) ---
    # Grappler dựng cả một "virtual device cluster" để tối ưu graph, rất tốn RAM.
    # Ta build graph mới trực tiếp và dùng optimizer nhẹ hơn của chính tf2onnx thay thế.
    print("Đang xuất mô hình ONNX...")

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(frozen_graph_def, name="")

        onnx_graph = process_tf_graph(
            tf_graph,
            input_names=[input_name],
            output_names=[output_name],
            opset=13,
        )
        onnx_graph = tf2onnx_optimizer.optimize_graph(onnx_graph)
        model_proto = onnx_graph.make_model("neural_ms")

        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())

    print(f"Đã xuất thành công: {onnx_path}")


if __name__ == "__main__":
    main()
