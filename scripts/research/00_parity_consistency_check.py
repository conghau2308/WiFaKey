"""
00_parity_consistency_check.py

Kiểm tra đại số THUẦN TÚY (không dùng TF, không dùng decoder, không dùng
trọng số đã train) xem: codeword sinh ra từ G_matrix (dùng lúc enroll) có
thỏa mãn phương trình parity-check H·c^T mod 2 = 0 với ma trận H mà
decoder tự xây dựng lúc _build_decoder() hay không.

Đây là test NỀN TẢNG nhất có thể có - nếu test này fail, nghĩa là G_matrix
(LDPC_GM_BG2_16.txt) và H được decoder dựng từ BaseGraph2_Set0.txt KHÔNG
PHẢI hai nửa của cùng 1 bộ mã. Không có cách nào decoder giải đúng được
trong trường hợp này, bất kể trọng số/kiến trúc/soft-LLR gì đi nữa.

Cách chạy:
    python scripts/research/00_parity_consistency_check.py
"""

import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_lib.Encode import Proto_LDPC


def build_lifted_H(base_pcm: np.ndarray, Z: int) -> np.ndarray:
    """
    Dựng ma trận H đầy đủ (lifted) từ base graph, dùng ĐÚNG quy ước shift
    mà wifakey_handler._build_decoder() đang dùng nội bộ:
        Lift_num = base_pcm[i, j] % Z
        P^shift[k, (k + shift) % Z] = 1
    (Copy lại chính xác quy ước này để test phản ánh đúng những gì decoder
    thực sự "hiểu" là parity-check matrix của nó.)
    """
    n_check_blocks, n_var_blocks = base_pcm.shape
    H = np.zeros((n_check_blocks * Z, n_var_blocks * Z), dtype=np.uint8)
    for i in range(n_check_blocks):
        for j in range(n_var_blocks):
            shift = base_pcm[i, j]
            if shift == -1:
                continue
            s = int(shift) % Z
            for k in range(Z):
                H[i * Z + k, j * Z + (k + s) % Z] = 1
    return H


def main():
    N, m, Z = 52, 42, 16
    data_dir = os.path.join(_PROJECT_ROOT, "wifakey_module", "data")

    # 1) G_matrix - dùng ĐÚNG class encode gốc, không copy lại logic
    encoder = Proto_LDPC(N, m, Z)
    G_matrix = encoder.G_matrix
    print(
        f"G_matrix shape: {G_matrix.shape}  (kỳ vọng: (key_length=160, feature_length=832))"
    )

    # 2) H - dựng lại THEO ĐÚNG quy ước decoder đang dùng (base_pcm % Z)
    pcm_path = os.path.join(data_dir, "BaseGraph", "BaseGraph2_Set0.txt")
    base_pcm = np.loadtxt(pcm_path, int, delimiter=None)
    print(
        f"BaseGraph2_Set0 shape: {base_pcm.shape}  "
        f"(kỳ vọng: ({N - m}, {N}) = ({N-m}, {N}))"
    )

    H_full = build_lifted_H(base_pcm, Z)
    print(f"H lifted shape: {H_full.shape}  (kỳ vọng: ({(N-m)*Z}, {N*Z}))")

    # 3) Sinh nhiều codeword ngẫu nhiên, kiểm tra H @ c^T mod 2 == 0
    n_trials = 50
    n_pass = 0
    max_violation = 0
    for _ in range(n_trials):
        random_key = np.random.randint(0, 2, size=(1, encoder.code_k * Z), dtype=int)
        codeword = encoder.encode_LDPC(random_key).flatten().astype(np.uint8) % 2

        syndrome = (H_full @ codeword) % 2
        n_violations = int(np.sum(syndrome))
        max_violation = max(max_violation, n_violations)
        if n_violations == 0:
            n_pass += 1

    print(f"\n=== KẾT QUẢ ===")
    print(f"Codeword thỏa mãn H·c=0: {n_pass}/{n_trials}")
    print(
        f"Số phương trình parity bị vi phạm nhiều nhất trong 1 lần thử: {max_violation} "
        f"(trên tổng {H_full.shape[0]} phương trình)"
    )

    if n_pass < n_trials:
        print("\n❌ XÁC NHẬN: G_matrix (encode) và H (decode) KHÔNG khớp nhau.")
        print("   Đây chính là nguyên nhân Test 1 (decoder thuần túy) fail 100%.")
        print("   Khả năng cao: BaseGraph2_Set0.txt lưu shift value theo Z_max của")
        print("   3GPP TS 38.212 (không phải riêng cho Z=16), và cần công thức:")
        print("       shift(Z) = floor(shift_base * Z / Z_max)")
        print("   thay vì 'shift_base % Z' đang dùng trong _build_decoder().")
        print("   Cần xác nhận Z_max thật của BaseGraph2_Set0.txt (thường là 384 theo")
        print("   chuẩn 5G NR cho iLS=0) và sửa công thức lifting cho khớp với cách")
        print("   G_matrix (LDPC_GM_BG2_16.txt) đã được tạo ra ban đầu.")
    else:
        print("\n✅ G_matrix và H khớp nhau về mặt đại số.")
        print("   => Lỗi Test 1 KHÔNG phải do mismatch G/H, mà nằm ở:")
        print("      - Trọng số Weights_Var*/Biases_Var* (sai file, sai thứ tự, hỏng)")
        print("      - Hoặc lỗi trong _build_decoder() ở phần khác (W_odd2even,")
        print("        W_even2odd, W_output... phần dựng graph message-passing)")
        print(
            "      - Hoặc quy ước dấu/thứ tự reshape giữa codeword và (N,Z) không khớp"
        )


if __name__ == "__main__":
    main()
