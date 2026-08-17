import numpy as np
import hashlib
from research.modulation.v2_empirical_llr import EmpiricalLLR


def ldpc_encode(random_key: np.ndarray, encoder) -> np.ndarray:
    """Mã hoá LDPC: key → codeword (832 bit)."""
    return encoder.encode_LDPC(random_key).flatten().astype(np.uint8)


def ldpc_decode(handler, noisy_bits: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """Giải mã LDPC với Empirical LLR + Neural‑MS."""
    emp_mod = EmpiricalLLR(lookup_path="experiments/out_step3/reliability_lookup.npz")
    llr = emp_mod.modulate(noisy_bits, context={"margin": margin}).flatten()
    llr = llr.reshape(1, handler.N, handler.Z)
    y_pred = handler.sess.run(handler.decoder_output, feed_dict={handler.xa: llr})
    y_pred = y_pred.flatten()
    return (y_pred > 0).astype(int)[: handler.key_length]


def verify_key(decoded_key: np.ndarray, key_hash: bytes) -> bool:
    """So sánh hash của key vừa giải mã với key_hash lưu trữ."""
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash
