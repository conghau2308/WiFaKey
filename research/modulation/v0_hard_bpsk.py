"""
v0_hard_bpsk — BASELINE, KHÔNG PHẢI CẢI TIẾN.

Đây chỉ là wrapper mỏng gọi thẳng hàm gốc trong
`wifakey_module/wifakey_lib/Modulation.py`. Không copy lại logic, không
sửa gì — để đảm bảo baseline luôn phản ánh CHÍNH XÁC hành vi code cũ.

Mọi so sánh cải tiến đều lấy kết quả của v0 làm mốc (control group).
"""

import numpy as np
import sys
import os

# Import trực tiếp từ code gốc, không copy-paste logic
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_lib import Modulation as _orig_modulation  # noqa: E402
from research.common.base_modulation import BaseModulation


class HardBPSK(BaseModulation):
    name = "v0_hard_bpsk"

    def modulate(
        self, noisy_bits: np.ndarray, context: dict | None = None
    ) -> np.ndarray:
        # Gọi thẳng hàm gốc: 0 -> -1, 1 -> +1, không dùng context
        return _orig_modulation.BPSK(noisy_bits).astype(np.float32)
