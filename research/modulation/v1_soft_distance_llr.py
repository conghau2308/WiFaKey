"""
v1_soft_distance_llr — CẢI TIẾN #1 (chi phí thấp nhất, không cần train lại decoder).

Ý tưởng:
    Bản gốc (v0) coi MỌI bit đáng tin như nhau: LLR luôn là ±1.
    Nhưng với thermometer code (LSSC), bit sinh ra từ giá trị embedding GẦN
    ngưỡng binarization có xác suất bị lật (do sai khác giữa 2 lần chụp) CAO
    HƠN NHIỀU so với bit sinh từ giá trị nằm giữa 1 khoảng rõ ràng.

    v1 tận dụng thông tin "khoảng cách tới ngưỡng gần nhất" (đã tính ở bước
    quantizer, xem research/quantizer/v0_lssc_with_confidence.py) để scale
    biên độ LLR: xa ngưỡng -> tin cậy cao -> |LLR| lớn.
                 gần ngưỡng -> tin cậy thấp -> |LLR| nhỏ.

    Dấu (sign) của LLR giữ nguyên logic gốc (0 -> âm, 1 -> dương), CHỈ đổi
    biên độ. Vì decoder Neural-MS hiện tại nhận đầu vào là float (không giới
    hạn ±1 cứng), việc đổi biên độ không cần sửa kiến trúc decoder.

Tham số cần tune qua experiments (KHÔNG hard-code 1 giá trị "đúng" duy nhất):
    scale       : hệ số nhân khoảng cách -> biên độ LLR
    min_mag     : biên độ tối thiểu (tránh LLR = 0 tuyệt đối gây mất thông tin dấu)
    max_mag     : biên độ tối đa (clip để không lệch quá xa dải giá trị mà
                  decoder gốc từng thấy khi train, tránh hành vi lạ)
"""

import numpy as np
from research.common.base_modulation import BaseModulation


class SoftDistanceLLR(BaseModulation):
    name = "v1_soft_distance_llr"

    def __init__(
        self,
        scale: float = 1.0,
        min_mag: float = 0.1,
        max_mag: float = 3.0,
        masked_mag: float = 1.0,
    ):
        self.scale = scale
        self.min_mag = min_mag
        self.max_mag = max_mag
        # Biên độ gán cho bit bị mask (mask_r=0). PHẢI là giá trị TRUNG TÍNH
        # (khớp hard BPSK, mặc định 1.0) - KHÔNG dùng max_mag. Vì tại các vị
        # trí này, y_noisy = helper_data GIỐNG HỆT NHAU cho cả genuine lẫn
        # impostor (không phụ thuộc embedding) - gán độ tin cậy quá cao ở đây
        # làm decoder dễ hội tụ đúng key cho CẢ kẻ mạo danh, gây tăng FAR
        # nghiêm trọng (đã kiểm chứng thực nghiệm: max_mag -> FAR=39.2%).
        self.masked_mag = masked_mag

    def modulate(
        self, noisy_bits: np.ndarray, context: dict | None = None
    ) -> np.ndarray:
        if context is None or "distance" not in context:
            raise ValueError(
                f"[{self.name}] cần context['distance'] (từ "
                f"research/quantizer/v0_lssc_with_confidence.py). "
                f"Nếu không có, dùng v0_hard_bpsk thay thế."
            )

        distance = context["distance"]
        assert (
            distance.shape == noisy_bits.shape
        ), "distance và noisy_bits phải cùng shape"

        magnitude = np.clip(distance * self.scale, self.min_mag, self.max_mag)

        # QUAN TRỌNG: bit bị mask (mask==0) bị ép về 0 TẤT ĐỊNH, không phụ
        # thuộc embedding -> y_noisy tại đó = helper_data, biết trước TUYỆT
        # ĐỐI (không phải "không chắc chắn"). Gán magnitude=max_mag (tin cậy
        # cao nhất) cho các vị trí này, KHÔNG dùng distance thật (vốn không
        # phản ánh gì về độ tin cậy ở đây).
        mask = context.get("mask")
        if mask is not None:
            magnitude = np.where(mask.astype(bool), magnitude, self.masked_mag)

        sign = (
            2 * noisy_bits.astype(np.float32) - 1
        )  # 0 -> -1, 1 -> +1 (giữ quy ước gốc)

        return (sign * magnitude).astype(np.float32)
