"""Common interface for ECC decoders used in the BSC benchmark."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ECCDecoder(Protocol):
    """Minimal interface every benchmarked decoder must implement.

    Encoders/decoders work in the binary domain (uint8 arrays of 0/1).
    ``encode`` converts a length-``k`` message into a length-``n`` codeword.
    ``decode`` takes a length-``n`` channel observation (already XORed with
    the BSC noise) and returns the recovered length-``k`` message.
    """

    name: str
    n: int  # codeword length
    k: int  # message length
    rate: float

    def encode(self, msg_bits: np.ndarray) -> np.ndarray: ...

    def decode(self, received_bits: np.ndarray) -> np.ndarray: ...
