import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import os
import sys

# Cần wrap lại — model gốc có thể trả về tuple (embedding, norm), nhưng ONNX
# export cần output SẠCH chỉ 1 tensor, nếu không Rust/ort sẽ khó xử lý.
class AdaFaceONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

# --- Dùng lại đúng cách load model production của bạn ---

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


from feature_extractor.adaface_handler import AdaFaceExtractor  # sửa import cho khớp project bạn

extractor = AdaFaceExtractor(device='cpu')  # CPU để export, tránh phụ thuộc CUDA lúc export
wrapped_model = AdaFaceONNXWrapper(extractor.model)
wrapped_model.eval()

dummy_input = torch.randn(1, 3, 112, 112)

torch.onnx.export(
    wrapped_model, dummy_input, "adaface_ir101.onnx",
    input_names=["input"], output_names=["embedding"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    dynamo=False,  # <-- THÊM dòng này — dùng exporter cũ (TorchScript-based), tương thích onnxruntime hiện tại
)
print("✅ Đã export: adaface_ir101.onnx")

# --- Introspect ngay để xác nhận cấu trúc ---
session = ort.InferenceSession("adaface_ir101.onnx", providers=["CPUExecutionProvider"])
print("\n=== INPUTS ===")
for i in session.get_inputs():
    print(f"  name={i.name}, shape={i.shape}, type={i.type}")
print("=== OUTPUTS ===")
for o in session.get_outputs():
    print(f"  name={o.name}, shape={o.shape}, type={o.type}")

# --- Tự kiểm tra: export đúng chưa, so PyTorch vs ONNX trên cùng input ---
with torch.no_grad():
    torch_out = wrapped_model(dummy_input).numpy()
onnx_out = session.run(None, {"input": dummy_input.numpy()})[0]
diff = np.abs(torch_out - onnx_out).max()
print(f"\nSai số tối đa PyTorch vs ONNX: {diff}")
print("→ Nếu < 1e-4, export đúng. Nếu lớn hơn nhiều, báo tôi ngay.")