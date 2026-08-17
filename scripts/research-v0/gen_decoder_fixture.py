"""
Generates a cross-check fixture for the LDPC-decode migration:
random c' (noisy codeword) -> SHA-256(reconstructed_key), computed with the
same ONNX decoder + BPSK + key-extraction + hashing logic that used to live
in WiFaKeyONNX.get_hash_k (now removed from the client).

Authentication_Service's LdpcDecoderService must reproduce this exact digest
for the same c' bytes — run its cross-check test against decoder_fixture.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

EXPORTS_DIR = Path("./exports")
N, Z, KEY_LEN, FEATURE_LEN = 52, 16, 160, 832

session = ort.InferenceSession(
    str(EXPORTS_DIR / "neural_ms_decoder.onnx"), providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

rng = np.random.default_rng(seed=42)
c_prime = rng.integers(0, 2, size=FEATURE_LEN, dtype=np.uint8)

llr = (c_prime.astype(np.float32) * 2.0 - 1.0).reshape(1, N, Z)
decoded_llr = session.run([output_name], {input_name: llr})[0].flatten()
key = (decoded_llr[:KEY_LEN] > 0).astype(np.uint8)
expected_hash = hashlib.sha256(key.tobytes()).digest()

fixture = {
    "c_prime_b64": __import__("base64").b64encode(c_prime.tobytes()).decode(),
    "expected_hash_b64": __import__("base64").b64encode(expected_hash).decode(),
}
out_path = Path("./decoder_fixture.json")
out_path.write_text(json.dumps(fixture, indent=2))
print(f"Wrote {out_path}: {fixture}")
