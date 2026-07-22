"""
Module tự sinh bởi 19_empirical_llr_fit.py — KHÔNG sửa tay.
Thay thế trực tiếp cho công thức tham số scale=60/sigma cũ.

Dùng:
    from empirical_llr import margin_to_llr, margin_to_p
    llr = margin_to_llr(margin_array)   # cắm thẳng vào input soft cho Neural-MS
"""
import os
import numpy as np

_DATA = np.load(os.path.join(os.path.dirname(__file__), "reliability_lookup.npz"))
_MARGIN_BP = _DATA["margin_breakpoints"]
_P_BP = _DATA["p_breakpoints"]
_EPS = float(_DATA["eps"])


def margin_to_p(margin):
    """margin (khoang cach toi nguong, cung dinh nghia voi buoc 2) -> xac suat lat bit,
    da hieu chinh thuc nghiem (isotonic, fit tren tune_train) va clip vao [eps, 0.5-eps]."""
    margin = np.asarray(margin, dtype=np.float64)
    p = np.interp(margin, _MARGIN_BP, _P_BP, left=_P_BP[0], right=_P_BP[-1])
    return np.clip(p, _EPS, 0.5 - _EPS)


def margin_to_llr(margin):
    """LLR = log((1-p)/p) — thay the hoan toan cong thuc scale=60 cu."""
    p = margin_to_p(margin)
    return np.log((1.0 - p) / p)
