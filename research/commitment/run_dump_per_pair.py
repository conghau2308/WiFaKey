"""
run_dump_per_pair.py

Chạy 1 variant (v1 hoặc rs_erasure) trên toàn bộ genuine+impostor pairs của
1 dataset/tier, lưu kết quả PASS/FAIL THEO TỪNG CẶP CỤ THỂ ra file JSON --
dùng làm input cho compare_mcnemar.py (paired McNemar's exact test giữa 2
variant, cùng tinh thần với run_ab_paired.py).

KHÔNG sửa gì ở run_single_config.py / v1_selection_puncturing.py /
v2_rs_erasure.py -- chỉ là 1 lớp orchestration mới bên ngoài, dùng lại
đúng các handler đã có. Giữ nguyên nguyên tắc "1 process = 1 lần chạy"
(tránh rò rỉ VRAM TF1.x) -- chạy riêng cho từng variant, mỗi lần ra 1 file
JSON riêng, rồi ghép lại bằng compare_mcnemar.py.

KEY GHÉP CẶP trong JSON:
    "{name_enroll}_{imagenum_enroll:04d}__{name_verify}_{imagenum_verify:04d}"
đủ để so khớp CHÍNH XÁC cùng 1 cặp khuôn mặt cụ thể giữa 2 lần chạy khác
variant, không phụ thuộc thứ tự dòng trong CSV hay thứ tự xử lý.

LƯU Ý VỀ Ý NGHĨA THỐNG KÊ CỦA PHÉP GHÉP CẶP NÀY:
    v1 và rs_erasure enroll() với ngẫu nhiên ĐỘC LẬP mỗi lần (selection_
    indices / true_secret không seed theo cặp) -- nên đây KHÔNG phải kiểu
    "cùng random_key/mask" như run_ab_paired.py ghép cặp giữa các
    modulation variant (dùng chung 1 lần enroll()). McNemar ở đây trả lời
    câu hỏi khác nhưng vẫn hợp lệ: "trên CÙNG 1 cặp khuôn mặt cụ thể,
    variant nào decode đúng thường xuyên hơn qua nhiều lần thử ngẫu
    nhiên" -- loại bỏ được nhiễu do độ khó riêng của từng cặp khuôn mặt
    (vốn dao động rất nhiều giữa các cặp), chỉ khác cách diễn giải so với
    paired-theo-cùng-random-key. Nếu sau này cần paired CHẶT hơn (cùng
    random_key/selection y hệt giữa 2 variant), sẽ cần sửa RNG cục bộ
    trong 2 handler đó sang RNG seed-được -- việc này CHƯA làm ở đây theo
    đúng yêu cầu không đụng code cũ.

Cách chạy (2 lệnh riêng, mỗi lệnh 1 process, ví dụ trên CPLFW dùng tier
'select' vì dataset này không có tune_*.csv):

    python research/commitment/run_dump_per_pair.py --variant v1 \
        --tier select \
        --pairs-dir datasets/processed/cplfw/pairs \
        --cache-dir datasets/processed/cplfw/embeddings_cache \
        --output-json research/commitment/logs/per_pair_v1_cplfw.json

    python research/commitment/run_dump_per_pair.py --variant rs_erasure --rs-nsym 8 \
        --tier select \
        --pairs-dir datasets/processed/cplfw/pairs \
        --cache-dir datasets/processed/cplfw/embeddings_cache \
        --output-json research/commitment/logs/per_pair_rs8_cplfw.json

    # Trên LFW (tier mặc định "tune"):
    python research/commitment/run_dump_per_pair.py --variant v1 \
        --output-json research/commitment/logs/per_pair_v1_lfw.json
    python research/commitment/run_dump_per_pair.py --variant rs_erasure --rs-nsym 8 \
        --output-json research/commitment/logs/per_pair_rs8_lfw.json
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


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


def pair_key(row: dict) -> str:
    return (
        f"{row['name_enroll']}_{int(row['imagenum_enroll']):04d}__"
        f"{row['name_verify']}_{int(row['imagenum_verify']):04d}"
    )


def build_handler(args, data_dir, weights_path, biases_path):
    if args.variant == "v1":
        from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler

        handler = SecureWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
        label = "v1_uniform_selection"

    elif args.variant == "rs_erasure":
        from research.commitment.v2_rs_erasure import RSErasureWiFaKeyHandler

        secret_bytes = (160 - args.rs_nsym * 8) // 8
        handler = RSErasureWiFaKeyHandler(
            data_path=data_dir,
            weights_path=weights_path,
            biases_path=biases_path,
            rs_nsym=args.rs_nsym,
            secret_bytes=secret_bytes,
            symbol_confidence_agg=args.symbol_agg,
        )
        label = f"rs_erasure_nsym{args.rs_nsym}"

    else:
        raise ValueError(f"variant không hỗ trợ: {args.variant}")

    return handler, label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=["v1", "rs_erasure"])
    ap.add_argument("--rs-nsym", type=int, default=4)
    ap.add_argument("--symbol-agg", choices=["min", "mean"], default="min")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--tier",
        default="tune",
        help="Prefix file pairs, vd 'tune' -> tune_genuine.csv (LFW), "
        "'select' -> select_genuine.csv (CPLFW, không có tune_*.csv).",
    )
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--output-json", required=True)
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

    genuine_path = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")
    impostor_path = os.path.join(pairs_dir, f"{args.tier}_impostor.csv")

    if not os.path.exists(genuine_path):
        print(f"[LỖI] Không tìm thấy {genuine_path}", file=sys.stderr)
        sys.exit(1)

    genuine_rows = load_pairs(genuine_path, args.max_pairs)
    impostor_rows = []
    has_impostor = os.path.exists(impostor_path)
    if has_impostor:
        impostor_rows = load_pairs(impostor_path, args.max_pairs)
    else:
        print(
            f"[WARN] Không tìm thấy {impostor_path} -- bỏ qua impostor "
            f"(chỉ đo genuine/GMR).",
            file=sys.stderr,
        )

    print(f"Genuine pairs: {len(genuine_rows)} | Impostor pairs: {len(impostor_rows)}")

    handler, label = build_handler(args, data_dir, weights_path, biases_path)

    results = {}  # key -> {"is_genuine": bool, "success": int}
    n_errors = 0

    for rows, is_genuine in [(genuine_rows, True), (impostor_rows, False)]:
        for row in rows:
            try:
                e1 = load_embedding(
                    cache_dir, row["name_enroll"], row["imagenum_enroll"]
                )
                e2 = load_embedding(
                    cache_dir, row["name_verify"], row["imagenum_verify"]
                )
                helper_data, mask, key_hash = handler.enroll(e1)
                success = handler.verify(e2, helper_data, mask, key_hash)
                results[pair_key(row)] = {
                    "is_genuine": is_genuine,
                    "success": int(success),
                }
            except Exception as e:
                n_errors += 1
                print(f"  [WARN] lỗi cặp ({row}): {e}", file=sys.stderr)

    n_gen = sum(1 for v in results.values() if v["is_genuine"])
    n_gen_ok = sum(1 for v in results.values() if v["is_genuine"] and v["success"])
    n_imp = sum(1 for v in results.values() if not v["is_genuine"])
    n_imp_ok = sum(1 for v in results.values() if not v["is_genuine"] and v["success"])

    gmr = 100 * n_gen_ok / n_gen if n_gen else float("nan")
    far = 100 * n_imp_ok / n_imp if n_imp else float("nan")

    print(f"\n=== {label} ({args.tier}) ===")
    print(f"GMR: {n_gen_ok}/{n_gen} = {gmr:.2f}%")
    if n_imp:
        print(f"FAR: {n_imp_ok}/{n_imp} = {far:.2f}%")
    print(f"Lỗi/exception: {n_errors}")

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "label": label,
        "variant": args.variant,
        "tier": args.tier,
        "rs_nsym": args.rs_nsym if args.variant == "rs_erasure" else None,
        "symbol_agg": args.symbol_agg if args.variant == "rs_erasure" else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "gmr": gmr,
        "far": far,
        "n_errors": n_errors,
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu per-pair results vào: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
