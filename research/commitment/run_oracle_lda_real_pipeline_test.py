"""
run_oracle_lda_real_pipeline_test.py

MỤC ĐÍCH
--------
Đo FRR/FAR THẬT (qua enroll()/verify() thật, KHÔNG phải oracle continuous-
embedding nữa) khi hoán đổi M_matrix gốc bằng M_matrix oracle-LDA (đã trực
giao hoá), TRÊN ĐÚNG TẬP VALIDATE đã dùng ở oracle_lda_separation_test.py —
để trả lời: "+2.5-2.8% separation ở mức continuous-embedding có sống sót
qua binarize + mask + LDPC decode (decoder KHÔNG retrain, vẫn dùng trọng số
train trên phân phối bit cũ) hay không?"

QUAN TRỌNG — ĐỌC TRƯỚC KHI DIỄN GIẢI KẾT QUẢ:
Decoder Neural-MS hiện tại được train trên phân phối bit sinh ra từ M_matrix
GỐC. Đổi M_matrix sẽ đổi phân phối bit đầu vào cho decoder (dù đã trực giao
hoá để giữ marginal distribution mỗi chiều tương tự nhau, decoder vẫn có thể
nhạy với tương quan/joint-distribution giữa các chiều mà phép trực giao hoá
không kiểm soát). Vì vậy:
  - Nếu FRR/FAR CẢI THIỆN ngay cả khi CHƯA retrain decoder -> tín hiệu rất
    mạnh, đáng đầu tư retrain decoder + re-enroll toàn hệ thống.
  - Nếu KHÔNG cải thiện (hoặc tệ hơn) -> CHƯA kết luận được gì chắc chắn,
    vì có thể decoder chỉ đơn giản chưa quen phân phối bit mới -> cần thử
    thêm bước retrain decoder mới biết chắc (tốn kém hơn, chỉ nên làm nếu
    có lý do khác ủng hộ).

Chạy (BẮT BUỘC tách 3 lệnh riêng, mỗi lệnh 1 process — an toàn cho GPU yếu):
    python run_oracle_lda_real_pipeline_test.py --handler baseline
    python run_oracle_lda_real_pipeline_test.py --handler oracle_lda
    python run_oracle_lda_real_pipeline_test.py --compare

Mỗi process chỉ khởi tạo ĐÚNG 1 decoder Neural-MS + 1 TF session, chạy xong
benchmark rồi thoát hẳn — process kết thúc thì OS tự thu hồi toàn bộ VRAM,
tránh lỗi "MemoryError: bad allocation" khi 2 session tồn tại song song
(TF1 với allow_growth=True không phải lúc nào cũng trả VRAM lại ngay cả
sau session.close(), nên tách process là cách chắc chắn nhất trên máy yếu).
"""

from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import sys
import csv
import random
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DATASET_NAME = "demogpairs"
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed", DATASET_NAME)
CACHE_DIR = os.path.join(DATA_DIR, "embeddings_cache")
IMAGE_METADATA_CSV = os.path.join(DATA_DIR, "image_metadata.csv")
GENUINE_CSV = os.path.join(DATA_DIR, "pairs", "audit_genuine.csv")
IMPOSTOR_SAME_CSV = os.path.join(DATA_DIR, "pairs", "audit_impostor_samefold.csv")
IMPOSTOR_CROSS_CSV = os.path.join(DATA_DIR, "pairs", "audit_impostor_crossfold.csv")

SELECT_RATIO = 0.7
SPLIT_SEED = 42  # PHẢI khớp oracle_lda_separation_test.py / build_oracle_lda_matrix.py
REG_EPS_CHOSEN = 2.0  # PHẢI khớp giá trị đã dùng trong build_oracle_lda_matrix.py
NEW_MATRIX_PATH = os.path.join(
    PROJECT_ROOT,
    "research",
    "modulation",
    "dimension_selection",
    f"M_matrix_oracle_lda_regeps{REG_EPS_CHOSEN}.npy",
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "oracle_lda_results"
)

# Đặt seed 1 lần, giống thói quen bạn đã áp dụng, để toàn bộ chuỗi enroll()
# (vốn dùng np.random bên trong) tái lập được giữa các lần chạy.
GLOBAL_SEED = 123


# =====================================================================
# LOAD SPLIT + PAIRS (giống hệt oracle_lda_separation_test.py)
# =====================================================================
def load_identity_fold_map() -> dict[str, str]:
    mapping = {}
    with open(IMAGE_METADATA_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["cache_filename"]:
                mapping[row["identity"]] = row["fold"]
    return mapping


def build_select_validate_split(identity_to_fold, select_ratio, seed):
    rng = random.Random(seed)
    ids_by_fold = defaultdict(list)
    for ident, fold in identity_to_fold.items():
        ids_by_fold[fold].append(ident)
    select_ids, validate_ids = set(), set()
    for fold, ids in sorted(ids_by_fold.items()):
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        n_select = int(round(len(ids_sorted) * select_ratio))
        select_ids.update(ids_sorted[:n_select])
        validate_ids.update(ids_sorted[n_select:])
    return select_ids, validate_ids


@dataclass
class Pair:
    cache_a: str
    cache_b: str
    identity_a: str = ""
    identity_b: str = ""


def load_genuine_pairs(validate_ids):
    pairs = []
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["identity"] in validate_ids:
                pairs.append(
                    Pair(
                        row["cache_filename_1"],
                        row["cache_filename_2"],
                        row["identity"],
                        row["identity"],
                    )
                )
    return pairs


def load_impostor_pairs_same_fold(validate_ids):
    pairs = []
    with open(IMPOSTOR_SAME_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["identity_1"] in validate_ids and row["identity_2"] in validate_ids:
                pairs.append(
                    Pair(
                        row["cache_filename_1"],
                        row["cache_filename_2"],
                        row["identity_1"],
                        row["identity_2"],
                    )
                )
    return pairs


def load_impostor_pairs_cross_fold(validate_ids):
    pairs = []
    with open(IMPOSTOR_CROSS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["identity_1"] in validate_ids and row["identity_2"] in validate_ids:
                pairs.append(
                    Pair(
                        row["cache_filename_1"],
                        row["cache_filename_2"],
                        row["identity_1"],
                        row["identity_2"],
                    )
                )
    return pairs


_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    """Cache trong RAM — cùng 1 ảnh có thể xuất hiện ở nhiều cặp (nhất là khi
    tăng TARGET_IMPOSTOR_PER_CATEGORY), tránh đọc đĩa lặp lại."""
    if cache_filename not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[cache_filename] = np.load(
            os.path.join(CACHE_DIR, cache_filename)
        )
    return _EMBEDDING_CACHE[cache_filename]


# =====================================================================
# BENCHMARK THẬT — enroll() trên ảnh A, verify() bằng ảnh B
# =====================================================================
def run_benchmark(handler, pairs: list[Pair], expect_success: bool) -> dict:
    """
    expect_success=True  -> đây là genuine pairs, kỳ vọng verify() trả True
                             (thất bại = false_reject, đóng góp vào FRR)
    expect_success=False -> đây là impostor pairs, kỳ vọng verify() trả False
                             (thành công = impostor_success/false_accept, đóng góp vào FAR)
    """
    n_total = len(pairs)
    n_verify_true = 0
    flagged_pairs = []  # các cặp có kết quả KHÁC kỳ vọng — dùng để traceback
    _devnull = io.StringIO()
    for idx, p in enumerate(pairs, start=1):
        emb_a = load_embedding(p.cache_a)
        emb_b = load_embedding(p.cache_b)
        with contextlib.redirect_stdout(_devnull):
            helper_data, mask_r, key_hash = handler.enroll(emb_a)
            result = handler.verify(emb_b, helper_data, mask_r, key_hash)
        if result:
            n_verify_true += 1

        # expect_success=True (genuine): fail bất thường là result=False -> false_reject
        # expect_success=False (impostor): fail bất thường là result=True -> impostor_success
        is_unexpected = result != expect_success
        if is_unexpected:
            flagged_pairs.append(
                dict(
                    identity_a=p.identity_a,
                    cache_a=p.cache_a,
                    identity_b=p.identity_b,
                    cache_b=p.cache_b,
                    verify_result=bool(result),
                )
            )

        if idx % 500 == 0 or idx == n_total:
            print(
                f"    ... {idx}/{n_total} cặp (đang thấy {n_verify_true} verify()=True, "
                f"{len(flagged_pairs)} bất thường)"
            )

    if expect_success:
        n_success = n_verify_true
        rate = n_success / n_total if n_total else float("nan")
        return dict(
            n_total=n_total,
            n_success=n_success,
            FRR=1 - rate,
            flagged_pairs=flagged_pairs,
        )
    else:
        n_success = n_verify_true  # đây chính là impostor_success (false_accept)
        rate = n_success / n_total if n_total else float("nan")
        return dict(
            n_total=n_total, n_success=n_success, FAR=rate, flagged_pairs=flagged_pairs
        )


def close_handler(handler, name: str):
    """Đóng TF session trước khi thoát process — không bắt buộc khi chạy
    tách process (OS sẽ tự thu hồi VRAM khi process kết thúc), nhưng vẫn
    làm cho sạch."""
    import gc

    print(f"  Đang đóng session của '{name}'...")
    try:
        handler.sess.close()
    except Exception as e:
        print(f"  (cảnh báo khi đóng session: {e})")
    del handler
    gc.collect()


def run_full_benchmark(
    handler_name: str, handler, genuine_pairs, impostor_same_pairs, impostor_cross_pairs
) -> dict:
    print(f"\n--- {handler_name} ---")
    gen_stats = run_benchmark(handler, genuine_pairs, expect_success=True)
    same_stats = run_benchmark(handler, impostor_same_pairs, expect_success=False)
    cross_stats = run_benchmark(handler, impostor_cross_pairs, expect_success=False)
    combined_n = same_stats["n_total"] + cross_stats["n_total"]
    combined_success = same_stats["n_success"] + cross_stats["n_success"]
    return dict(
        FRR=gen_stats["FRR"],
        genuine_n=gen_stats["n_total"],
        genuine_success=gen_stats["n_success"],
        FAR_same=same_stats["FAR"],
        FAR_cross=cross_stats["FAR"],
        FAR_combined=combined_success / combined_n if combined_n else float("nan"),
        impostor_success_same=same_stats["n_success"],
        impostor_success_cross=cross_stats["n_success"],
        flagged_genuine=gen_stats["flagged_pairs"],
        flagged_impostor_same=same_stats["flagged_pairs"],
        flagged_impostor_cross=cross_stats["flagged_pairs"],
    )


def run_one_handler(which: str):
    """Chạy MỘT handler duy nhất trong process này, lưu kết quả ra JSON.
    Gọi lệnh này 2 LẦN, MỖI LẦN 1 PROCESS RIÊNG (đóng hẳn cửa sổ terminal/
    process giữa 2 lần nếu máy yếu) để đảm bảo VRAM được giải phóng hoàn
    toàn — an toàn hơn hẳn so với chạy tuần tự trong cùng 1 process, vì
    TF1 (allow_growth=True) đôi khi không trả VRAM về hệ điều hành ngay
    cả sau khi session.close()."""
    assert which in ("baseline", "oracle_lda")

    np.random.seed(GLOBAL_SEED)

    print("[1/4] Dựng lại ĐÚNG split select/validate đã dùng ở oracle test...")
    identity_to_fold = load_identity_fold_map()
    _, validate_ids = build_select_validate_split(
        identity_to_fold, SELECT_RATIO, SPLIT_SEED
    )
    print(f"  validate: {len(validate_ids)} identity")

    print("\n[2/4] Load pairs (đã lọc theo validate_ids)...")
    genuine_pairs = load_genuine_pairs(validate_ids)
    impostor_same_pairs = load_impostor_pairs_same_fold(validate_ids)
    impostor_cross_pairs = load_impostor_pairs_cross_fold(validate_ids)
    print(
        f"  genuine={len(genuine_pairs)} | impostor_same_fold={len(impostor_same_pairs)} "
        f"| impostor_cross_fold={len(impostor_cross_pairs)}"
    )

    print(f"\n[3/4] Khởi tạo handler '{which}'...")
    if which == "baseline":
        from wifakey_module.wifakey_handler import WiFaKeyHandler

        handler = WiFaKeyHandler()
    else:
        from research.commitment.wifakey_handler_lda_variant import WiFaKeyHandlerLDAVariant

        handler = WiFaKeyHandlerLDAVariant(m_matrix_override_path=NEW_MATRIX_PATH)

    print(f"\n[4/4] Chạy benchmark cho '{which}'...")
    result = run_full_benchmark(
        which, handler, genuine_pairs, impostor_same_pairs, impostor_cross_pairs
    )
    close_handler(handler, which)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"result_{which}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n=== Đã lưu kết quả '{which}' -> {out_path} ===")
    print("Chạy tiếp process khác (--handler baseline hoặc --handler oracle_lda)")
    print("rồi cuối cùng chạy --compare để xem bảng so sánh.")


def run_compare():
    paths = {
        "baseline": os.path.join(RESULTS_DIR, "result_baseline.json"),
        "oracle_lda": os.path.join(RESULTS_DIR, "result_oracle_lda.json"),
    }
    results = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(
                f"*** THIẾU: {path} — hãy chạy `python {os.path.basename(__file__)} --handler {name}` trước. ***"
            )
            return
        with open(path, "r", encoding="utf-8") as f:
            results[name] = json.load(f)

    print(
        f"{'Variant':12s} {'FRR':>8s} {'gen_ok':>8s} {'/':1s}{'total':>6s} "
        f"{'FAR_same':>9s} {'FAR_cross':>10s} {'FAR_combined':>13s} {'imp_ok_same':>12s} {'imp_ok_cross':>13s}"
    )
    for name, r in results.items():
        print(
            f"{name:12s} {r['FRR']:8.4f} {r['genuine_success']:8d} {'/':1s}{r['genuine_n']:6d} "
            f"{r['FAR_same']:9.4f} {r['FAR_cross']:10.4f} {r['FAR_combined']:13.4f} "
            f"{r['impostor_success_same']:12d} {r['impostor_success_cross']:13d}"
        )

    print("\nCÁCH ĐỌC: so 'oracle_lda' vs 'baseline' ở FRR và FAR_combined.")
    print("- FRR thấp hơn + FAR_combined thấp hơn (hoặc bằng) baseline -> cải")
    print("  thiện thật, KHÔNG cần retrain decoder -> đáng đầu tư re-enroll.")
    print("- FRR/FAR không đổi rõ rệt hoặc tệ hơn -> +2.8% separation ở mức")
    print("  continuous-embedding CHƯA sống sót qua decoder hiện tại -> muốn")
    print("  biết chắc cần thử retrain decoder trên phân phối bit mới.")

    # ================================================================
    # TRACEBACK: liệt kê cụ thể cặp nào lọt/fail, so overlap 2 handler
    # ================================================================
    def pair_key(fp: dict) -> tuple:
        return (fp["identity_a"], fp["cache_a"], fp["identity_b"], fp["cache_b"])

    print("\n" + "=" * 70)
    print("CHI TIẾT CÁC CẶP BẤT THƯỜNG (impostor_success / false_reject)")
    print("=" * 70)

    for category in [
        "flagged_impostor_same",
        "flagged_impostor_cross",
        "flagged_genuine",
    ]:
        label = {
            "flagged_impostor_same": "IMPOSTOR_SUCCESS (same-fold)",
            "flagged_impostor_cross": "IMPOSTOR_SUCCESS (cross-fold)",
            "flagged_genuine": "FALSE_REJECT (genuine)",
        }[category]
        print(f"\n--- {label} ---")

        base_flagged = {
            pair_key(fp): fp for fp in results["baseline"].get(category, [])
        }
        lda_flagged = {
            pair_key(fp): fp for fp in results["oracle_lda"].get(category, [])
        }

        only_base = set(base_flagged) - set(lda_flagged)
        only_lda = set(lda_flagged) - set(base_flagged)
        both = set(base_flagged) & set(lda_flagged)

        print(f"  Chỉ baseline bị ({len(only_base)}):")
        for k in only_base:
            print(f"    {k[0]}/{k[1]}  <->  {k[2]}/{k[3]}")

        print(f"  Chỉ oracle_lda bị ({len(only_lda)}):")
        for k in only_lda:
            print(f"    {k[0]}/{k[1]}  <->  {k[2]}/{k[3]}")

        print(
            f"  CẢ HAI cùng bị (giới hạn cấu trúc, không phụ thuộc M_matrix) ({len(both)}):"
        )
        for k in both:
            print(f"    {k[0]}/{k[1]}  <->  {k[2]}/{k[3]}")

    print("\nCÁCH ĐỌC PHẦN TRACEBACK:")
    print("- 'CẢ HAI cùng bị' ở impostor_success -> đúng những cặp M_matrix nào")
    print("  cũng không cứu được (kiểu Venus/Serena) -> giới hạn của bản thân")
    print("  cặp danh tính đó, không phải lỗi của oracle_lda.")
    print("- 'Chỉ oracle_lda bị' ở impostor_success -> ĐÁNG LO, oracle_lda tạo")
    print("  ra lỗ hổng MỚI mà baseline không có -> cần xem lại.")
    print("- 'Chỉ oracle_lda bị' ở FALSE_REJECT -> đây là cái giá FRR thật của")
    print("  M_matrix mới, xem thử identity đó có đặc điểm gì chung không.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handler",
        choices=["baseline", "oracle_lda"],
        help="Chạy benchmark cho 1 handler duy nhất trong process này, lưu kết quả ra JSON.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Đọc kết quả JSON của cả 2 handler (đã chạy trước đó) và in bảng so sánh.",
    )
    args = parser.parse_args()

    if args.compare:
        run_compare()
    elif args.handler:
        run_one_handler(args.handler)
    else:
        parser.error(
            "Cần chỉ định --handler baseline, --handler oracle_lda, hoặc --compare.\n"
            "Chạy đúng thứ tự (mỗi lệnh 1 process riêng):\n"
            f"  python {os.path.basename(__file__)} --handler baseline\n"
            f"  python {os.path.basename(__file__)} --handler oracle_lda\n"
            f"  python {os.path.basename(__file__)} --compare"
        )


if __name__ == "__main__":
    main()
