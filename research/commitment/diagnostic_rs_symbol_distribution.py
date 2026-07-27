"""
diagnostic_rs_symbol_distribution.py

Diagnostic cho rs_erasure (v2_rs_erasure.py) -- đo TRỰC TIẾP nguyên nhân
GMR chỉ tăng nhẹ (43.81% vs v1 42.45%, z~0.58, KHÔNG có ý nghĩa thống kê)
dù tương quan bit-level LLR~lỗi đã đo được trước đó là 56.3% (so với kỳ
vọng ngẫu nhiên 20%).

CÂU HỎI CẦN TRẢ LỜI:
    Trong các ca genuine THẤT BẠI của rs_erasure, lỗi thật (so giữa
    message_bits lúc enroll và reconstructed_message_bits lúc verify,
    TRƯỚC khi đưa qua RS decode) trải ra trên BAO NHIÊU symbol (byte,
    8-bit) khác nhau trong số 20 symbol? Và trong số đó, có bao nhiêu
    NẰM TRỌN trong 4 symbol mà cơ chế chọn erasure (theo min |LLR| trong
    symbol) đã đánh dấu?

ĐIỀU KIỆN ĐỦ để RS(20,16,nsym=4) sửa đúng 100%: tập symbol lỗi thật phải
là TẬP CON của tập 4 symbol đã đánh dấu erasure (đã tự xác nhận qua
self-test của v2_rs_erasure.py -- đúng 4 erasure đã khai báo, không có
lỗi "ẩn" ngoài đó, luôn sửa đúng). Vì vậy:
    - Nếu 1 ca FAIL và lỗi thật KHÔNG là tập con erasure -> ĐÚNG như dự
      đoán, xác nhận nguyên nhân "lỗi rải quá rộng, vượt khả năng gom
      vào 4 symbol của cơ chế LLR-ranking hiện tại".
    - Nếu 1 ca FAIL nhưng lỗi thật LÀ tập con erasure -> BẤT THƯỜNG, có
      thể có bug trong code (nên = 0 ca nếu logic đúng).

KHÔNG đổi logic bảo mật/quyết định pass-fail so với v2_rs_erasure.py --
class dưới đây chỉ thêm bản "debug" của enroll/verify để lấy thêm state
trung gian (message_bits gốc, reconstructed bits, symbol nào bị chọn
erasure) phục vụ đo đạc, không ảnh hưởng gì tới benchmark GMR/FAR thật.

Chạy:
    python research/commitment/diagnostic_rs_symbol_distribution.py --max-pairs 200
    (bỏ --max-pairs để chạy full 881 cặp genuine khi cần kết luận chắc chắn)

    # Nếu muốn đo cho rs_nsym khác (vd đã đổi sang RS(20,12)):
    python research/commitment/diagnostic_rs_symbol_distribution.py --rs-nsym 8 --max-pairs 200
"""

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from reedsolo import ReedSolomonError

from research.commitment.v2_rs_erasure import (
    RSErasureWiFaKeyHandler,
    _bits_to_bytes,
    _bytes_to_bits,
    _symbol_confidence,
)


def load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def load_pairs(pairs_csv: str, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


class RSErasureDiagnosticHandler(RSErasureWiFaKeyHandler):
    """Giống hệt RSErasureWiFaKeyHandler -- enroll_debug/verify_debug chỉ
    trả THÊM state trung gian (message_bits, erase_symbols...), không đổi
    bất kỳ quyết định pass/fail hay logic bảo mật nào so với bản gốc."""

    def enroll_debug(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full), size=self.feature_length, replace=False
        )
        selection_indices.sort()
        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected = b_full[selection_indices]

        true_secret_bits = np.random.randint(
            0, 2, size=self.secret_bytes * 8, dtype=np.uint8
        )
        true_secret_bytes = _bits_to_bytes(true_secret_bits)
        rs_encoded = bytes(self.rsc.encode(true_secret_bytes))
        message_bits = _bytes_to_bits(rs_encoded, self.key_length)
        message_bits_2d = message_bits.reshape(1, -1).astype(int)

        codeword = self.encoder.encode_LDPC(message_bits_2d).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(true_secret_bytes).digest()

        return helper_data, selection_mask, key_hash, message_bits, true_secret_bytes

    def verify_debug(self, feature_vector_float, helper_data, selection_mask):
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

        llr_per_symbol = message_llr_mag.reshape(self.rs_n_bytes, 8)
        symbol_confidence = _symbol_confidence(
            llr_per_symbol, self.symbol_confidence_agg
        )
        erase_symbols = np.argsort(symbol_confidence)[: self.rs_nsym].tolist()

        received = bytearray(_bits_to_bytes(reconstructed_message_bits))
        try:
            decoded_secret, _, _ = self.rsc.decode(received, erase_pos=erase_symbols)
        except ReedSolomonError:
            decoded_secret = None

        return reconstructed_message_bits, erase_symbols, decoded_secret


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--rs-nsym",
        type=int,
        default=4,
        help="Phải khớp với rs_nsym đã dùng khi chạy benchmark cần chẩn đoán.",
    )
    ap.add_argument(
        "--symbol-agg",
        choices=["min", "mean"],
        default="min",
        help="Thống kê dùng để xếp hạng symbol đáng ngờ (xem docstring "
        "_symbol_confidence trong v2_rs_erasure.py).",
    )
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", "labeled_faces_in_the_wild", "embeddings_cache"
    )
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")

    genuine_rows = load_pairs(
        os.path.join(pairs_dir, "tune_genuine.csv"), args.max_pairs
    )
    print(f"Genuine pairs dùng: {len(genuine_rows)}")

    secret_bytes = (160 - args.rs_nsym * 8) // 8
    handler = RSErasureDiagnosticHandler(
        data_path=data_dir,
        weights_path=weights_path,
        biases_path=biases_path,
        rs_nsym=args.rs_nsym,
        secret_bytes=secret_bytes,
        symbol_confidence_agg=args.symbol_agg,
    )

    n_error_symbols_fail = []
    n_error_symbols_pass_with_error = []
    subset_ok_fail = 0
    subset_notok_fail = 0
    n_perfect = 0
    n_total = 0
    n_pass = 0
    n_errors_runtime = 0

    for row in genuine_rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])

            helper_data, selection_mask, key_hash, message_bits, true_secret_bytes = (
                handler.enroll_debug(e1)
            )
            reconstructed_bits, erase_symbols, decoded_secret = handler.verify_debug(
                e2, helper_data, selection_mask
            )

            # QUAN TRỌNG: tiêu chí pass/fail PHẢI giống hệt verify() thật --
            # so sánh hash, KHÔNG chỉ dựa vào việc decode() có raise hay
            # không. decode() có thể "thành công" (không raise) nhưng trả
            # về SAI secret (miscorrection) khi số lỗi thật vượt xa sức sửa
            # -- giới hạn đã biết của bounded-distance decoder. Chỉ có so
            # hash mới xác nhận đúng secret gốc, đúng như hệ thống thật
            # (RSErasureWiFaKeyHandler.verify) đang làm.
            if decoded_secret is not None:
                recon_hash = hashlib.sha256(bytes(decoded_secret)).digest()
                rs_success = recon_hash == key_hash
            else:
                rs_success = False

            n_total += 1
            if rs_success:
                n_pass += 1

            diff = np.not_equal(message_bits, reconstructed_bits)
            if not diff.any():
                n_perfect += 1
                continue

            diff_per_symbol = diff.reshape(handler.rs_n_bytes, 8)
            error_symbols = set(np.where(diff_per_symbol.any(axis=1))[0].tolist())
            n_err_sym = len(error_symbols)

            is_subset = error_symbols.issubset(set(erase_symbols))

            if rs_success:
                n_error_symbols_pass_with_error.append(n_err_sym)
            else:
                n_error_symbols_fail.append(n_err_sym)
                if is_subset:
                    subset_ok_fail += 1
                else:
                    subset_notok_fail += 1

        except Exception as e:
            n_errors_runtime += 1
            print(f"  [WARN] lỗi cặp ({row}): {e}", file=sys.stderr)

    print(
        f"\n=== KẾT QUẢ DIAGNOSTIC rs_erasure "
        f"(nsym={args.rs_nsym}, symbol_agg={args.symbol_agg}) ==="
    )
    print(f"Tổng cặp genuine xử lý: {n_total} (lỗi runtime: {n_errors_runtime})")
    print(
        f"RS decode thành công (verify PASS): {n_pass}/{n_total} = "
        f"{100 * n_pass / n_total:.2f}%"
    )
    print(
        f"Ca hoàn toàn không lỗi bit nào (HD=0, decoder đúng 100%): "
        f"{n_perfect}/{n_total} = {100 * n_perfect / n_total:.2f}%"
    )

    n_fail = n_total - n_pass
    print(f"\n--- Trong số {n_fail} ca THẤT BẠI ---")
    if n_error_symbols_fail:
        c = Counter(n_error_symbols_fail)
        print("Phân bố số symbol lỗi thật (trong 20 symbol):")
        for k in sorted(c):
            marker = " <= trong sức sửa (nsym)" if k <= args.rs_nsym else ""
            print(
                f"  {k:2d} symbol lỗi: {c[k]:4d} ca "
                f"({100 * c[k] / len(n_error_symbols_fail):.1f}%){marker}"
            )
        print(f"Trung bình: {np.mean(n_error_symbols_fail):.2f} symbol lỗi/ca")
        print(f"Median: {np.median(n_error_symbols_fail):.1f} symbol lỗi/ca")

    if n_fail > 0:
        print(f"\nTrong {n_fail} ca fail:")
        print(
            f"  Lỗi thật KHÔNG nằm trong {args.rs_nsym} symbol đã chọn erasure: "
            f"{subset_notok_fail} ca ({100 * subset_notok_fail / n_fail:.1f}%) "
            f"-- ĐÚNG dự đoán, xác nhận nguyên nhân"
        )
        print(
            f"  Lỗi thật NẰM TRONG {args.rs_nsym} symbol đã chọn erasure nhưng "
            f"vẫn fail: {subset_ok_fail} ca ({100 * subset_ok_fail / n_fail:.1f}%) "
            f"-- BẤT THƯỜNG, cần kiểm tra lại code nếu > 0"
        )

    if n_error_symbols_pass_with_error:
        print(f"\n--- Trong số ca PASS nhưng vẫn có lỗi bit (RS sửa được) ---")
        c2 = Counter(n_error_symbols_pass_with_error)
        for k in sorted(c2):
            print(f"  {k:2d} symbol lỗi: {c2[k]:4d} ca")
        print(
            f"(Tối đa lý thuyết phải <= {args.rs_nsym} symbol -- nếu thấy số "
            f"lớn hơn ở đây, có bug)"
        )


if __name__ == "__main__":
    main()
