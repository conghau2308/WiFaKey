"""
test_onnx_inference.py

Kiểm tra mô hình ONNX bằng ONNX Runtime và so sánh với TensorFlow gốc.
"""

import numpy as np
import onnxruntime as ort
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)


def main():
    # Đường dẫn tới file ONNX
    onnx_path = os.path.join(
        _PROJECT_ROOT, "build-wifakey-core", "wifakey-core", "neural_ms.onnx"
    )

    # 1. Tạo input giả (LLR) giống hệt trong test Rust (embedding = 0.5, secret, salt, ...)
    # Để đơn giản, ta dùng LLR toàn 0 (tương đương với margin = 0 và noisy_bit = 0)
    dummy_llr = np.zeros((1, 52, 16), dtype=np.float32)

    # 2. Chạy inference bằng ONNX Runtime
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: dummy_llr})
    onnx_output = result[0].flatten()[:160]
    print(f"ONNX output (first 160 values): {onnx_output[:20]}...")

    # 3. Chạy inference bằng TensorFlow gốc (nếu có thể)
    try:
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()
        from wifakey_module.wifakey_handler import WiFaKeyHandler

        handler = WiFaKeyHandler(
            data_path=os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
        )
        tf_output = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: dummy_llr}
        )
        tf_output = tf_output.flatten()[:160]
        print(f"TF output  (first 160 values): {tf_output[:20]}...")

        # So sánh
        if np.allclose(onnx_output, tf_output, atol=1e-5):
            print("✅ ONNX và TensorFlow cho kết quả GIỐNG NHAU.")
        else:
            print("❌ ONNX và TensorFlow cho kết quả KHÁC NHAU!")
            print("   Điều này giải thích vì sao pipeline bị lỗi trong Rust.")
    except Exception as e:
        print(f"Không thể chạy TensorFlow để so sánh: {e}")
        print("Tuy nhiên, bạn có thể kiểm tra ONNX output ở trên.")
        print(
            "Nếu output toàn giá trị gần 0 hoặc toàn dương, thì mô hình ONNX có vấn đề."
        )


if __name__ == "__main__":
    main()
