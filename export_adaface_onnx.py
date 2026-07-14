"""
Export AdaFace IR-101 model sang ONNX format để dùng với ONNX Runtime Web.

Input:  (1, 3, 112, 112) float32, normalized to [-1, 1]
Output: (1, 512) float32, L2-normalized embedding

Usage:
    cd g:/Final/WiFaKey_252
    python export_adaface_onnx.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor.adaface_handler import AdaFaceExtractor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "adaface_ir101.onnx")


class AdaFaceONNXWrapper(nn.Module):
    """Wrapper đảm bảo output luôn là tensor (1, 512), không phải tuple."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # L2 normalize
        norm = torch.norm(out, dim=1, keepdim=True).clamp(min=1e-12)
        return out / norm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading AdaFace model...")
    extractor = AdaFaceExtractor(device="cpu")

    wrapper = AdaFaceONNXWrapper(extractor.model)
    wrapper.eval()

    dummy_input = torch.zeros(1, 3, 112, 112, dtype=torch.float32)

    # Verify forward pass trước khi export
    with torch.no_grad():
        out = wrapper(dummy_input)
    assert out.shape == (1, 512), f"Unexpected output shape: {out.shape}"
    print(f"Forward pass OK — output shape: {out.shape}")

    print(f"Exporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        OUTPUT_PATH,
        opset_version=14,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    print(f"Export thành công: {OUTPUT_PATH}")

    # Verify ONNX model
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, 3, 112, 112), dtype=np.float32)
        result = sess.run(["embedding"], {"input": dummy_np})
        assert result[0].shape == (1, 512), f"ONNX output shape sai: {result[0].shape}"

        # Kiểm tra L2 norm ≈ 1
        norm = np.linalg.norm(result[0][0])
        assert abs(norm - 1.0) < 1e-3, f"Embedding chưa được L2 normalize: norm={norm}"
        print(f"ONNX verification OK — embedding norm: {norm:.6f}")
    except ImportError:
        print("onnxruntime không có sẵn, bỏ qua verification bước ONNX Runtime.")

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
