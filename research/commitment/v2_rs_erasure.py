"""
v2_rs_erasure.py

C.5 (tiếp) — Reed-Solomon erasure decoding trên bit đã decode, dùng |LLR|
đầu ra của Neural-MS để xác định symbol nào "đáng ngờ" tại mỗi lần verify
cụ thể (động, không cố định vị trí như bản reduced_key_128 cũ).

Bối cảnh / lý do tồn tại file này (xem lịch sử research/commitment/):
    - reduced_key_128 (bỏ CỐ ĐỊNH 32 bit cuối) gần như không cải thiện GMR
      (42.11% vs 42.45%) vì lỗi thật rải rác ngẫu nhiên trên 160 vị trí,
      không ưu tiên rơi vào 32 vị trí cố định được chọn bỏ.
    - Đo tương quan: 56.3% bit lỗi thật trùng với 32 bit |LLR| thấp nhất
      (so với kỳ vọng ngẫu nhiên 20%) -> decoder "biết" nó đang không chắc
      ở đâu, và đúng chỗ đó thường là nơi lỗi thật xảy ra.
    -> Hướng đúng: bỏ ĐỘNG theo |LLR| tại mỗi lần verify, không bỏ cố định
      theo vị trí. Đây chính là bài toán erasure decoding kinh điển: "biết
      vị trí nghi ngờ, không biết giá trị đúng".

Cách làm SAI đã loại bỏ (xem thảo luận trước): hash riêng từng khối nhỏ
(vd 20 khối x 8 bit, verify theo đa số khối khớp) -- lỗ hổng nghiêm trọng
vì 1 khối 8 bit chỉ có 2^8=256 khả năng, brute-force offline trực tiếp từ
helper data công khai, không cần biết gì về sinh trắc học.

Cách làm ĐÚNG (file này): true_secret KHÔNG BAO GIỜ lộ ở dạng thô hay dạng
tách nhỏ. Toàn bộ 128 bit secret được mã hoá qua Reed-Solomon THÀNH 1 khối
duy nhất trước khi vào LDPC; hash vẫn là sha256 trên nguyên khối 128-bit
(2^128 entropy, không brute-force được, không có "khối con" nào để tấn công
riêng lẻ). RS chỉ đóng vai trò SỬA LẠI các symbol đã biết là đáng ngờ sau
khi LDPC/Neural-MS decode xong -- không phải cơ chế so khớp từng phần.

    Enroll:
        true_secret (128 bit, CSPRNG)
        --[RS(20,16) trên GF(2^8), nsym=4]--> message (160 bit, 20 byte)
        codeword = encode_LDPC(message)            -- HẠ TẦNG GIỮ NGUYÊN
        helper_data = b_selected XOR codeword       -- HẠ TẦNG GIỮ NGUYÊN
        key_hash = sha256(true_secret)              -- 128-bit, 1 khối duy nhất

    Verify:
        decoded_message, llr_magnitude = decode qua Neural-MS (như hiện tại)
        erasure_symbols = 4 symbol (8-bit) có min(|LLR|) thấp nhất trong
                           số các bit thuộc symbol đó (nếu 1 bit trong symbol
                           bị nghi ngờ, coi cả symbol là erasure -- quy về
                           đơn vị symbol, không phải bit thô)
        reconstructed_secret = RS erasure-decode(decoded_message, erasure_symbols)
        so sánh sha256(reconstructed_secret) với key_hash

RÀNG BUỘC ĐÃ XÁC NHẬN QUA TEST THẬT VỚI THƯ VIỆN `reedsolo` (KHÔNG ĐOÁN):
    - RSCodec(nsym).encode(data 16 byte) -> ĐÚNG 20 byte, không có padding
      ẩn -- erase_pos là index 0..19 trực tiếp trong mảng 20 byte nhận về,
      khớp 1-1 với thứ tự byte của message 160-bit (systematic, message đi
      thẳng vào 160 bit đầu của codeword LDPC, không đảo thứ tự).
    - decode(received, erase_pos=[...]) trả về tuple 3 phần:
      (decoded_msg, decoded_msgecc, errata_pos) -- decoded_msg là 16 byte
      data gốc (true_secret), KHÔNG kèm ECC.
    - Đúng bằng sức sửa (nsym erasure) -> decode đúng 100% (đã test).
      Vượt sức sửa (nsym+1 erasure trở lên) -> raise ReedSolomonError
      "Too many erasures to correct" (đã test) -- PHẢI catch exception này
      trong verify(), coi là xác thực thất bại, không phải lỗi hệ thống.

THAM SỐ DỄ ĐỔI (giống pattern effective_key_length của reduced_key cũ) để
quét thử tỷ lệ RS khác nhau (vd RS(20,12) sửa 8 symbol, entropy còn 96 bit)
mà không cần sửa logic: `rs_nsym` (số symbol ECC = sức sửa erasure tối đa)
và `secret_bytes` (số byte secret thật, phải thoả secret_bytes*8 +
rs_nsym*8 == key_length).

CHƯA benchmark trên dữ liệu thật -- cần chạy qua run_single_config.py
(--variant rs_erasure) để có số GMR/FAR thật trước khi kết luận.
"""

import hashlib

import numpy as np
from reedsolo import RSCodec, ReedSolomonError

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """bits: mảng 0/1 uint8, độ dài phải chia hết cho 8. MSB-first per byte
    (np.packbits mặc định), phải dùng NHẤT QUÁN với _bytes_to_bits."""
    return np.packbits(bits.astype(np.uint8)).tobytes()


def _bytes_to_bits(data: bytes, n_bits: int) -> np.ndarray:
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits[:n_bits]


def _symbol_confidence(llr_per_symbol: np.ndarray, method: str) -> np.ndarray:
    """Quy |LLR| per-bit (shape: n_symbols x 8) về 1 điểm "đáng ngờ" cho
    mỗi symbol -- điểm CÀNG THẤP = symbol càng đáng nghi, càng ưu tiên
    chọn làm erasure.

    'min'  : điểm = bit yếu nhất trong symbol (bảo thủ -- chỉ cần 1 bit
             không chắc là coi cả symbol đáng nghi). NHƯỢC ĐIỂM đã quan
             sát qua diagnostic: với 8 mẫu/symbol, min dễ bị chi phối bởi
             thống kê giá trị cực trị (extreme-value) -- hầu hết symbol
             đều có VÀI bit LLR thấp do biến động ngẫu nhiên, khiến việc
             xếp hạng 20 symbol theo min kém phân biệt (100% ca fail ở
             tầng <=4 symbol lỗi vẫn chọn sai symbol).
    'mean' : điểm = trung bình |LLR| của cả 8 bit trong symbol. Ít bị chi
             phối bởi 1 bit ngoại lệ hơn min -- một symbol thật sự chứa
             bit lỗi thường có xu hướng đồng thời có margin nhỏ ở NHIỀU
             bit lân cận (do cùng nằm gần biên quyết định của decoder),
             nên trung bình có thể phân biệt tốt hơn thống kê cực trị.
             GIẢ THUYẾT cần kiểm chứng bằng dữ liệu thật, chưa có gì đảm bảo
             chắc chắn tốt hơn 'min'.
    """
    if method == "min":
        return llr_per_symbol.min(axis=1)
    elif method == "mean":
        return llr_per_symbol.mean(axis=1)
    else:
        raise ValueError(
            f"symbol_confidence_agg không hợp lệ: {method!r} "
            f"(chỉ hỗ trợ 'min' hoặc 'mean')"
        )


class RSErasureWiFaKeyHandler(SecureWiFaKeyHandler):
    def __init__(
        self,
        *args,
        rs_nsym: int = 8, # Sau khi đo đạc thì đây là kết quả tốt nhất (chưa tính đến bảo mật)
        secret_bytes: int = 16,
        symbol_confidence_agg: str = "min",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if symbol_confidence_agg not in ("min", "mean"):
            raise ValueError(
                f"symbol_confidence_agg không hợp lệ: {symbol_confidence_agg!r} "
                f"(chỉ hỗ trợ 'min' hoặc 'mean')"
            )
        self.symbol_confidence_agg = symbol_confidence_agg

        if self.key_length % 8 != 0:
            raise ValueError(
                f"key_length ({self.key_length}) phải chia hết cho 8 để dùng "
                f"RS trên byte-symbol (GF(2^8))."
            )
        expected_n_bytes = secret_bytes + rs_nsym
        if expected_n_bytes * 8 != self.key_length:
            raise ValueError(
                f"secret_bytes ({secret_bytes}) + rs_nsym ({rs_nsym}) = "
                f"{expected_n_bytes} byte ({expected_n_bytes * 8} bit) phải "
                f"KHỚP ĐÚNG key_length hiện tại ({self.key_length} bit). "
                f"Ví dụ mặc định: secret_bytes=16 (128 bit), rs_nsym=4 "
                f"(32 bit) -> 20 byte = 160 bit = key_length hiện tại."
            )

        self.rs_nsym = rs_nsym
        self.secret_bytes = secret_bytes
        self.rs_n_bytes = expected_n_bytes  # = key_length // 8, số symbol RS
        self.rsc = RSCodec(rs_nsym)

    # ------------------------------------------------------------------
    def enroll(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full), size=self.feature_length, replace=False
        )
        selection_indices.sort()

        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1

        b_selected = b_full[selection_indices]

        # true_secret: CSPRNG thật, KHÔNG cố định bit nào -- toàn bộ entropy
        # bảo mật nằm ở đây (128 bit).
        true_secret_bits = np.random.randint(
            0, 2, size=self.secret_bytes * 8, dtype=np.uint8
        )
        true_secret_bytes = _bits_to_bytes(true_secret_bits)

        # RS encode: 16 byte data -> 20 byte (data + 4 byte ECC), append ở
        # cuối (đã xác nhận qua test: encode() không đảo thứ tự, không pad).
        rs_encoded = bytes(self.rsc.encode(true_secret_bytes))
        message_bits = _bytes_to_bits(rs_encoded, self.key_length)
        message_bits_2d = message_bits.reshape(1, -1).astype(int)

        codeword = self.encoder.encode_LDPC(message_bits_2d).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)

        # key_hash chỉ trên true_secret (128 bit, 1 khối duy nhất, không
        # tách nhỏ) -- KHÔNG hash rs_encoded (sẽ lộ cấu trúc RS/ECC).
        key_hash = hashlib.sha256(true_secret_bytes).digest()

        return helper_data, selection_mask, key_hash

    # ------------------------------------------------------------------
    def verify(
        self, feature_vector_float, helper_data, selection_mask, stored_key_hash
    ) -> bool:
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        selection_indices = np.where(selection_mask == 1)[0]

        b_selected = b_full[selection_indices]
        y_noisy_bits = np.logical_xor(b_selected, helper_data)

        from wifakey_module.wifakey_lib import Modulation

        y_llr = (
            Modulation.BPSK(y_noisy_bits)
            .astype(np.float32)
            .reshape((1, self.N, self.Z))
        )
        y_pred_llr = self.sess.run(self.decoder_output, feed_dict={self.xa: y_llr})

        llr_flat = y_pred_llr.flatten()
        decoded_codeword = (llr_flat > 0).astype(np.uint8)

        reconstructed_message_bits = decoded_codeword[: self.key_length]
        message_llr_mag = np.abs(llr_flat[: self.key_length])

        # Quy |LLR| per-bit về per-symbol (byte): điểm không tin cậy của 1
        # symbol = min(|LLR|) trong số 8 bit thuộc symbol đó -- nếu CHỈ 1
        # bit trong symbol bị nghi ngờ, coi cả symbol là erasure (bảo thủ,
        # đúng như đã bàn: RS hoạt động theo symbol, không theo bit lẻ).
        llr_per_symbol = message_llr_mag.reshape(self.rs_n_bytes, 8)
        symbol_confidence = _symbol_confidence(
            llr_per_symbol, self.symbol_confidence_agg
        )

        # Chọn đúng rs_nsym symbol kém tin cậy nhất làm erasure -- ĐỘNG,
        # khác nhau mỗi lần verify (không cố định vị trí như reduced_key_128).
        erase_symbols = np.argsort(symbol_confidence)[: self.rs_nsym].tolist()

        received = bytearray(_bits_to_bytes(reconstructed_message_bits))

        try:
            decoded_secret, _decoded_with_ecc, _errata_pos = self.rsc.decode(
                received, erase_pos=erase_symbols
            )
        except ReedSolomonError:
            # Vượt sức sửa (>rs_nsym symbol thực sự sai trong số các symbol
            # đã đánh dấu erasure, hoặc lỗi nằm ngoài các symbol đã chọn) --
            # coi là xác thực thất bại, không phải lỗi hệ thống.
            return False

        recon_hash = hashlib.sha256(bytes(decoded_secret)).digest()
        return recon_hash == stored_key_hash


# ----------------------------------------------------------------------
# Self-test KHÔNG phụ thuộc biometric/TF -- chỉ kiểm tra logic bit<->byte
# và hành vi RSCodec đúng như phần code trên giả định. Chạy độc lập:
#     python research/commitment/v2_rs_erasure.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rsc = RSCodec(4)

    true_secret_bits = np.random.randint(0, 2, size=128, dtype=np.uint8)
    true_secret_bytes = _bits_to_bytes(true_secret_bits)

    rs_encoded = bytes(rsc.encode(true_secret_bytes))
    message_bits = _bytes_to_bits(rs_encoded, 160)
    assert message_bits.shape[0] == 160

    # Round-trip không lỗi
    received = bytearray(_bits_to_bytes(message_bits))
    decoded_secret, _, _ = rsc.decode(received, erase_pos=[])
    assert bytes(decoded_secret) == true_secret_bytes, "Round-trip clean thất bại"

    # Giả lập 4 symbol sai (bằng đúng sức sửa) tại vị trí bất kỳ, khai báo
    # đúng erase_pos -> phải sửa đúng 100%.
    rng = np.random.default_rng(0)
    corrupt_positions = sorted(rng.choice(20, size=4, replace=False).tolist())
    received2 = bytearray(_bits_to_bytes(message_bits))
    for p in corrupt_positions:
        received2[p] ^= 0xFF
    decoded_secret2, _, _ = rsc.decode(received2, erase_pos=corrupt_positions)
    assert (
        bytes(decoded_secret2) == true_secret_bytes
    ), "Erasure-correct 4 symbol thất bại"

    # Giả lập 5 symbol sai (vượt sức sửa) -> phải raise ReedSolomonError.
    corrupt5 = sorted(rng.choice(20, size=5, replace=False).tolist())
    received3 = bytearray(_bits_to_bytes(message_bits))
    for p in corrupt5:
        received3[p] ^= 0xFF
    try:
        rsc.decode(received3, erase_pos=corrupt5)
        raise AssertionError("Lẽ ra phải raise ReedSolomonError với 5 erasure")
    except ReedSolomonError:
        pass

    print(
        "Self-test PASSED: bit<->byte round-trip, erasure-correct đúng sức "
        "sửa (4 symbol), và raise đúng khi vượt sức sửa (5 symbol)."
    )
