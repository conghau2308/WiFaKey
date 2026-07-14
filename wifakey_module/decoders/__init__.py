"""ECC decoder wrappers used by the BSC benchmark."""

from .base import ECCDecoder

__all__ = [
    "ECCDecoder",
    "NeuralMSDecoder",
    "BCHDecoder",
]


def __getattr__(name):  # PEP 562: lazy import so we don't drag TF/galois on every import
    if name == "NeuralMSDecoder":
        from .neural_ms_decoder import NeuralMSDecoder

        return NeuralMSDecoder
    if name == "BCHDecoder":
        from .bch_decoder import BCHDecoder

        return BCHDecoder
    raise AttributeError(f"module 'wifakey_module.decoders' has no attribute {name!r}")
