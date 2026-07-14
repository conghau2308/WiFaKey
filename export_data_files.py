"""
Convert data files (numpy, txt) sang JSON/binary để browser load được.
Copy InsightFace buffalo_l ONNX files vào exports/.
Copy anti-spoofing.onnx vào exports/.

Usage:
    cd g:/Final/WiFaKey_252
    python export_data_files.py

Output trong exports/:
    M_matrix.json
    binarization_intervals.json
    G_matrix.json
    anti-spoofing.onnx
    det_10g.onnx        (InsightFace face detection)
    2d106det.onnx       (InsightFace landmark - nếu có)
"""

import os
import sys
import json
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "exports")


def export_numpy_to_json(npy_path: str, out_path: str, name: str):
    arr = np.load(npy_path)
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    data = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.flatten().tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: {out_path} ({size_kb:.1f} KB)")


def export_g_matrix(txt_path: str, out_path: str):
    """Convert LDPC generator matrix text file → JSON."""
    matrix = np.loadtxt(txt_path, dtype=np.int32, delimiter=",")
    print(f"  G_matrix: shape={matrix.shape}, dtype={matrix.dtype}")
    data = {
        "shape": list(matrix.shape),
        "dtype": "int32",
        "data": matrix.flatten().tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: {out_path} ({size_kb:.1f} KB)")


def copy_onnx_file(src: str, dst: str, name: str):
    if not os.path.exists(src):
        print(f"  WARNING: {name} không tìm thấy tại {src}")
        return False
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"  Copied {name}: {dst} ({size_mb:.2f} MB)")
    return True


def find_insightface_models():
    """Tìm buffalo_l ONNX models trong InsightFace cache."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".insightface", "models", "buffalo_l"),
        os.path.join(home, "AppData", "Roaming", ".insightface", "models", "buffalo_l"),
        os.path.join(home, ".cache", "insightface", "models", "buffalo_l"),
        "C:\\Users\\Admin\\.insightface\\models\\buffalo_l",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # 1. M_matrix.npy
    print("Exporting M_matrix...")
    export_numpy_to_json(
        os.path.join(BASE_DIR, "wifakey_module", "data", "M_matrix.npy"),
        os.path.join(OUTPUT_DIR, "M_matrix.json"),
        "M_matrix",
    )

    # 2. binarization_intervals.npy
    print("\nExporting binarization_intervals...")
    export_numpy_to_json(
        os.path.join(BASE_DIR, "wifakey_module", "data", "binarization_intervals.npy"),
        os.path.join(OUTPUT_DIR, "binarization_intervals.json"),
        "binarization_intervals",
    )

    # 3. G_matrix (LDPC generator matrix)
    print("\nExporting G_matrix...")
    g_matrix_path = os.path.join(
        BASE_DIR, "wifakey_module", "data", "BaseGraph_GM", "LDPC_GM_BG2_16.txt"
    )
    export_g_matrix(g_matrix_path, os.path.join(OUTPUT_DIR, "G_matrix.json"))

    # 4. Anti-spoofing ONNX
    print("\nCopying anti-spoofing.onnx...")
    copy_onnx_file(
        os.path.join(BASE_DIR, "vision_module", "anti-spoofing.onnx"),
        os.path.join(OUTPUT_DIR, "anti-spoofing.onnx"),
        "anti-spoofing.onnx",
    )

    # 5. InsightFace buffalo_l ONNX models
    print("\nLooking for InsightFace buffalo_l models...")
    buffalo_dir = find_insightface_models()
    if buffalo_dir:
        print(f"  Found: {buffalo_dir}")
        for model_file in ["det_10g.onnx", "2d106det.onnx", "w600k_r50.onnx"]:
            src = os.path.join(buffalo_dir, model_file)
            dst = os.path.join(OUTPUT_DIR, model_file)
            copy_onnx_file(src, dst, model_file)
    else:
        print("  InsightFace buffalo_l không tìm thấy trong cache.")
        print("  Chạy ứng dụng một lần để InsightFace tự download buffalo_l,")
        print("  sau đó chạy lại script này.")
        print("  Hoặc download thủ công từ: https://github.com/deepinsight/insightface")

    print("\nDone. Tất cả files trong exports/:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        unit = "MB" if size > 1024 * 1024 else "KB"
        val = size / (1024 * 1024) if size > 1024 * 1024 else size / 1024
        print(f"  {f:40s} {val:.1f} {unit}")


if __name__ == "__main__":
    main()
