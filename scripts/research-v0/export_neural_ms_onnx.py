"""
Export Neural-MS LDPC Decoder (TF1) sang ONNX format.

Vì model dùng TF1 tf.placeholder + sess.run() (không có SavedModel),
cần freeze graph trước rồi dùng tf2onnx để convert.

Input:  xa — (1, 52, 16) float32, LLR values: +1 = bit 0, -1 = bit 1
Output: ya_output24 — (1, 832) float32, LLR sau decode (>0 = bit 0)

Usage:
    cd g:/Final/WiFaKey_252
    pip install tf2onnx
    python export_neural_ms_onnx.py

Sau khi chạy xong: exports/neural_ms_decoder.onnx
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
PB_PATH = os.path.join(OUTPUT_DIR, "neural_ms_decoder.pb")
ONNX_PATH = os.path.join(OUTPUT_DIR, "neural_ms_decoder.onnx")


def freeze_and_export_pb():
    """Bước 1: Load TF1 graph, freeze variables → lưu .pb"""
    import tensorflow.compat.v1 as tf
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    from wifakey_module.decoders.neural_ms_decoder import NeuralMSDecoder

    print("Loading Neural-MS decoder (TF1 graph)...")
    decoder = NeuralMSDecoder()

    print("Freezing TF1 graph variables -> constants...")
    output_node_name = "ya_output24"
    frozen_graph_def = convert_variables_to_constants(
        decoder.sess,
        decoder.sess.graph_def,
        [output_node_name],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tf.io.write_graph(frozen_graph_def, OUTPUT_DIR, "neural_ms_decoder.pb", as_text=False)
    print(f"Frozen graph saved: {PB_PATH}")

    # Verify frozen graph với dummy input
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open(PB_PATH, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            xa = sess.graph.get_tensor_by_name("xa:0")
            output = sess.graph.get_tensor_by_name(f"{output_node_name}:0")

            dummy_llr = np.ones((1, 52, 16), dtype=np.float32)
            result = sess.run(output, feed_dict={xa: dummy_llr})
            assert result.shape == (1, 832), f"Unexpected shape: {result.shape}"
            print(f"Frozen graph verification OK — output shape: {result.shape}")

    decoder.sess.close()
    return frozen_graph_def


def convert_pb_to_onnx():
    """Bước 2: Dùng tf2onnx convert .pb → .onnx"""
    try:
        import tf2onnx
    except ImportError:
        print("\ntf2onnx chưa được cài. Chạy: pip install tf2onnx")
        print("Sau đó chạy lại script này.")
        print("\nHoặc convert thủ công bằng command line:")
        print(
            f"python -m tf2onnx.convert "
            f"--input {PB_PATH} "
            f"--output {ONNX_PATH} "
            f'--inputs "xa:0[1,52,16]" '
            f'--outputs "ya_output24:0"'
        )
        return False

    import subprocess

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--input", PB_PATH,
        "--output", ONNX_PATH,
        "--inputs", "xa:0[1,52,16]",
        "--outputs", "ya_output24:0",
        "--opset", "14",
    ]
    print(f"\nChạy tf2onnx: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"tf2onnx lỗi:\n{result.stderr}")
        return False

    print(result.stdout)
    print(f"ONNX export thành công: {ONNX_PATH}")
    return True


def verify_onnx():
    """Bước 3: Verify ONNX model khớp với TF1 output"""
    import onnxruntime as ort
    from wifakey_module.decoders.neural_ms_decoder import NeuralMSDecoder

    print("\nVerifying ONNX output khớp với TF1 original...")

    # Chạy TF1
    decoder_tf = NeuralMSDecoder()
    test_bits = np.random.randint(0, 2, size=(1, 52 * 16), dtype=np.uint8)
    llr_tf = (1 - 2 * test_bits).astype(np.float32).reshape(1, 52, 16)
    tf_out = decoder_tf.sess.run(decoder_tf.decoder_output, feed_dict={decoder_tf.xa: llr_tf})
    decoder_tf.sess.close()

    # Chạy ONNX
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["ya_output24:0"], {"xa:0": llr_tf})

    # So sánh decoded bits
    tf_bits = (tf_out > 0).astype(np.uint8).flatten()
    onnx_bits = (onnx_out[0] > 0).astype(np.uint8).flatten()
    match = np.sum(tf_bits == onnx_bits)
    print(f"Bit match: {match}/{len(tf_bits)} ({100*match/len(tf_bits):.1f}%)")
    assert match == len(tf_bits), "ONNX output không khớp TF1!"
    print("Verification OK — ONNX output khớp 100% với TF1.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Bước 1: Freeze và lưu .pb
    freeze_and_export_pb()

    # Bước 2: Convert sang ONNX
    success = convert_pb_to_onnx()
    if not success:
        return

    # Bước 3: Verify
    try:
        verify_onnx()
    except Exception as e:
        print(f"Verification error: {e}")

    file_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"\nFile size: {file_size_mb:.1f} MB")
    print(f"Output: {ONNX_PATH}")


if __name__ == "__main__":
    main()
