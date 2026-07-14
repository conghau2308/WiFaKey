"""Binary BCH decoder via ``galois.BCH`` (Berlekamp–Massey hard decision).

Default ``BCH(1023, 193)``: among valid narrow-sense BCH subcodes of
length 1023, ``k=193`` yields rate ``R ≈ 0.1887`` and ``t = 118``, which is
the closest valid match to the WiFaKey Neural-MS reference
(``R = 160/832 ≈ 0.1923``).  Block length ``n = 1023`` is also the same
order as the other two decoders (``n = 832``), so BLER curves compare on
similar redundancy *and* a similar number of channel uses per block.

Pass ``k=10`` (with ``n=511``) to reproduce the paper's low-rate BCH
illustration.  Pass ``k=94`` (with ``n=511``) for the previous default.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import galois

    _GALOIS_OK = True
    _IMPORT_ERR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    galois = None
    _GALOIS_OK = False
    _IMPORT_ERR = exc


# Closest valid ``galois.BCH(1023, k)`` rate to 160/832 ≈ 0.192308 (see scan):
#   k=193 -> R=0.18866, t=118 (best match)
_DEFAULT_BCH_N = 1023
_DEFAULT_BCH_K = 193


class BCHDecoder:
    def __init__(self, n: int = _DEFAULT_BCH_N, k: int = _DEFAULT_BCH_K) -> None:
        if not _GALOIS_OK:
            raise ImportError(
                "galois is required for BCHDecoder. Install with `pip install galois`. "
                f"Original error: {_IMPORT_ERR}"
            )
        # ``galois`` will auto-pick the designed distance that yields exactly k.
        self.bch = galois.BCH(n, k)
        self.n = int(self.bch.n)
        self.k = int(self.bch.k)
        self.t = int(self.bch.t)
        self.rate = self.k / self.n
        self.name = f"bch_{self.n}_{self.k}"
        self._GF2 = galois.GF(2)

    def encode(self, msg_bits: np.ndarray) -> np.ndarray:
        msg = np.asarray(msg_bits, dtype=np.uint8).reshape(self.k)
        gf_msg = self._GF2(msg)
        codeword = self.bch.encode(gf_msg)
        return np.asarray(codeword, dtype=np.uint8).reshape(self.n)

    def decode(
        self,
        received_bits: np.ndarray,
        p: Optional[float] = None,  # unused (hard decision)
    ) -> np.ndarray:
        rx = np.asarray(received_bits, dtype=np.uint8).reshape(self.n)
        gf_rx = self._GF2(rx)
        # ``output="message"`` is the default; explicit for clarity.
        decoded_msg = self.bch.decode(gf_rx, output="message")
        # In some galois versions decode may return (msg, n_errors).
        if isinstance(decoded_msg, tuple):
            decoded_msg = decoded_msg[0]
        return np.asarray(decoded_msg, dtype=np.uint8).reshape(self.k)
