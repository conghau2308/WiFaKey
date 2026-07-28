"""
run_oracle_lda_cplfw_test.py

MỤC ĐÍCH
--------
Test M_matrix oracle-LDA (fit HOÀN TOÀN từ DemogPairs, chưa từng thấy
CPLFW) trên select_genuine.csv/select_impostor.csv của CPLFW — đây là bài
test tổng quát hóa QUAN TRỌNG HƠN việc chỉ tăng N trên DemogPairs, vì CPLFW
là phân phối dữ liệu hoàn toàn độc lập.

KHÁC VỚI run_oracle_lda_real_pipeline_test.py (DemogPairs):
  - KHÔNG cần tách select/validate ở đây — vì M_matrix không được fit từ
    CPLFW nên toàn bộ select_genuine.csv/select_impostor.csv của CPLFW đều
    dùng được làm tập đo, không lo rò rỉ.
  - KHÔNG có same-fold/cross-fold (CPLFW không có nhãn nhân khẩu học) —
    chỉ 1 con số FRR, 1 con số FAR.
  - Cache filename theo quy ước "{name}_{imagenum:04d}.npy" (imagenum luôn
    "0" -> "{name}_0000.npy"), khớp 03a_extract_embeddings_cplfw.py.

Chạy (tách 3 process riêng, giống bài test DemogPairs, để tránh tràn VRAM):
    python run_oracle_lda_cplfw_test.py --handler baseline
    python run_oracle_lda_cplfw_test.py --handler oracle_lda
    python run_oracle_lda_cplfw_test.py --compare
"""

from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import sys
import csv
import numpy as np
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DATASET_NAME = "cplfw"
CACHE_DIR = os.path.join(
    PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)
GENUINE_CSV = os.path.join(
    PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs", "select_genuine.csv"
)
IMPOSTOR_CSV = os.path.join(
    PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs", "select_impostor.csv"
)

REG_EPS_CHOSEN = 2.0  # PHẢI khớp giá trị đã dùng trong build_oracle_lda_matrix.py
NEW_MATRIX_PATH = os.path.join(
    PROJECT_ROOT,
    "research",
    "modulation",
    "dimension_selection",
    f"M_matrix_oracle_lda_regeps{REG_EPS_CHOSEN}.npy",
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "oracle_lda_cplfw_results"
)

GLOBAL_SEED = 123  # khớp thói quen seed cố định của bạn


# =====================================================================
# LOAD PAIRS — cột name_enroll/imagenum_enroll/name_verify/imagenum_verify
# =====================================================================
def cplfw_cache_filename(name: str, imagenum) -> str:
    return f"{name}_{int(imagenum):04d}.npy"


@dataclass
class Pair:
    cache_a: str
    cache_b: str


def load_pairs(csv_path: str) -> list[Pair]:
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache_a = cplfw_cache_filename(row["name_enroll"], row["imagenum_enroll"])
            cache_b = cplfw_cache_filename(row["name_verify"], row["imagenum_verify"])
            pairs.append(Pair(cache_a, cache_b))
    return pairs


_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    if cache_filename not in _EMBEDDING_CACHE:
        path = os.path.join(CACHE_DIR, cache_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy {path} — kiểm tra lại quy ước tên file cache CPLFW "
                f"(kỳ vọng '{{name}}_{{imagenum:04d}}.npy', vd 'Anders_Fogh_Rasmussen_1_0000.npy')."
            )
        _EMBEDDING_CACHE[cache_filename] = np.load(path)
    return _EMBEDDING_CACHE[cache_filename]


# =====================================================================
# BENCHMARK — giống hệt logic ở bài test DemogPairs
# =====================================================================
def run_benchmark(handler, pairs: list[Pair], expect_success: bool) -> dict:
    n_total = len(pairs)
    n_verify_true = 0
    _devnull = io.StringIO()
    for idx, p in enumerate(pairs, start=1):
        emb_a = load_embedding(p.cache_a)
        emb_b = load_embedding(p.cache_b)
        with contextlib.redirect_stdout(_devnull):
            helper_data, mask_r, key_hash = handler.enroll(emb_a)
            result = handler.verify(emb_b, helper_data, mask_r, key_hash)
        if result:
            n_verify_true += 1
        if idx % 200 == 0 or idx == n_total:
            print(
                f"    ... {idx}/{n_total} cặp (đang thấy {n_verify_true} verify()=True)"
            )

    n_success = n_verify_true
    rate = n_success / n_total if n_total else float("nan")
    if expect_success:
        return dict(n_total=n_total, n_success=n_success, FRR=1 - rate)
    return dict(n_total=n_total, n_success=n_success, FAR=rate)


def close_handler(handler, name: str):
    import gc

    print(f"  Đang đóng session của '{name}'...")
    try:
        handler.sess.close()
    except Exception as e:
        print(f"  (cảnh báo khi đóng session: {e})")
    del handler
    gc.collect()


def run_one_handler(which: str):
    assert which in ("baseline", "oracle_lda")
    np.random.seed(GLOBAL_SEED)

    print(
        "[1/3] Load pairs CPLFW (dùng TOÀN BỘ select_genuine/select_impostor — "
        "M_matrix chưa từng thấy CPLFW nên không cần tách select/validate)..."
    )
    genuine_pairs = load_pairs(GENUINE_CSV)
    impostor_pairs = load_pairs(IMPOSTOR_CSV)
    print(f"  genuine={len(genuine_pairs)} | impostor={len(impostor_pairs)}")

    print(f"\n[2/3] Khởi tạo handler '{which}'...")
    if which == "baseline":
        from wifakey_module.wifakey_handler import WiFaKeyHandler

        handler = WiFaKeyHandler()
    else:
        from research.commitment.wifakey_handler_lda_variant import WiFaKeyHandlerLDAVariant

        handler = WiFaKeyHandlerLDAVariant(m_matrix_override_path=NEW_MATRIX_PATH)

    print(f"\n[3/3] Chạy benchmark cho '{which}'...")
    print("  -- genuine --")
    gen_stats = run_benchmark(handler, genuine_pairs, expect_success=True)
    print("  -- impostor --")
    imp_stats = run_benchmark(handler, impostor_pairs, expect_success=False)
    close_handler(handler, which)

    result = dict(
        FRR=gen_stats["FRR"],
        genuine_n=gen_stats["n_total"],
        genuine_success=gen_stats["n_success"],
        FAR=imp_stats["FAR"],
        impostor_n=imp_stats["n_total"],
        impostor_success=imp_stats["n_success"],
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"result_{which}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n=== Đã lưu kết quả '{which}' -> {out_path} ===")


def run_compare():
    paths = {
        "baseline": os.path.join(RESULTS_DIR, "result_baseline.json"),
        "oracle_lda": os.path.join(RESULTS_DIR, "result_oracle_lda.json"),
    }
    results = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"*** THIẾU: {path} — hãy chạy `--handler {name}` trước. ***")
            return
        with open(path, "r", encoding="utf-8") as f:
            results[name] = json.load(f)

    print(
        f"{'Variant':12s} {'FRR':>8s} {'gen_ok/total':>14s} {'FAR':>8s} {'imp_ok/total':>14s}"
    )
    for name, r in results.items():
        print(
            f"{name:12s} {r['FRR']:8.4f} {r['genuine_success']:6d}/{r['genuine_n']:<6d}  "
            f"{r['FAR']:8.4f} {r['impostor_success']:6d}/{r['impostor_n']:<6d}"
        )

    print("\nĐây là bài test QUAN TRỌNG NHẤT trong toàn bộ chuỗi kiểm chứng: nếu")
    print("oracle_lda cũng giảm impostor_success trên CPLFW (dataset M_matrix")
    print("CHƯA TỪNG THẤY) mà FRR không xấu đi đáng kể -> bằng chứng tổng quát")
    print("hóa mạnh, đủ cơ sở để đầu tư enroll lại toàn hệ thống + cân nhắc")
    print("retrain decoder.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handler", choices=["baseline", "oracle_lda"])
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        run_compare()
    elif args.handler:
        run_one_handler(args.handler)
    else:
        parser.error(
            "Cần chỉ định --handler baseline, --handler oracle_lda, hoặc --compare.\n"
            f"  python {os.path.basename(__file__)} --handler baseline\n"
            f"  python {os.path.basename(__file__)} --handler oracle_lda\n"
            f"  python {os.path.basename(__file__)} --compare"
        )


if __name__ == "__main__":
    main()
