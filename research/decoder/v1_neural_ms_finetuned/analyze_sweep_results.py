"""
analyze_sweep_results.py

Dùng khi bạn chạy sweep_trainable_iters.py TÁCH LẺ từng trainable_iters
(mỗi lần chỉ để 1 giá trị trong CANDIDATES, chạy xong mới bỏ comment giá
trị tiếp theo) - vì lúc đó `sweep_summary.json` bị GHI ĐÈ mỗi lần chạy và
chỉ chứa đúng 1 dòng của lần chạy gần nhất, không tổng hợp được.

Script này KHÔNG đọc sweep_summary.json. Nó quét toàn bộ file
`history_iters{N}.json` đang có trong sweep_results/ (các file này KHÔNG
bị ghi đè giữa các lần chạy vì tên file khác nhau theo N), tự suy ra
best_epoch/best_val_exact_match cho từng trainable_iters từ chính lịch sử
train, rồi xếp hạng toàn bộ - bất kể bạn đã chạy rải rác bao nhiêu lần,
lúc nào.

SỬA SO VỚI BẢN GỐC: bản gốc dùng entry["val_acc"] -- key này KHÔNG TỒN TẠI
trong history do train.py sinh ra (chỉ có "val_bit_acc" và
"val_exact_match"), sẽ crash ngay với KeyError. Đã sửa dùng đúng
"val_exact_match" -- khớp đúng bài học đã tự phát hiện và sửa trong
train.py (best_epoch/early-stopping phải dựa trên exact_match, KHÔNG phải
bit_correct/val_bit_acc, vì exact_match mới là con số khớp với những gì
verify() thật kiểm tra).

Cách chạy:
    python research/decoder/v1_neural_ms_finetuned/analyze_sweep_results.py
"""

import os
import re
import json
import glob

OUT_DIR = os.path.join(os.path.dirname(__file__), "sweep_results")
FILE_PATTERN = re.compile(r"history_iters(\d+)\.json$")


def load_all_histories():
    """Trả về dict {trainable_iters: history_list}, quét mọi
    history_iters*.json đang có trong sweep_results/."""
    histories = {}
    paths = glob.glob(os.path.join(OUT_DIR, "history_iters*.json"))
    if not paths:
        raise FileNotFoundError(
            f"Không tìm thấy file history_iters*.json nào trong {OUT_DIR}. "
            f"Chạy sweep_trainable_iters.py trước (dù chỉ 1 trainable_iters "
            f"mỗi lần cũng được)."
        )

    for path in paths:
        match = FILE_PATTERN.search(os.path.basename(path))
        if not match:
            continue  # bỏ qua file lạ không đúng định dạng tên
        trainable_iters = int(match.group(1))
        with open(path, "r") as f:
            history = json.load(f)
        if not history:
            print(f"  ⚠ {os.path.basename(path)} rỗng, bỏ qua.")
            continue
        histories[trainable_iters] = history

    return histories


def best_from_history(history):
    """Suy ra (best_val_exact, best_epoch) từ 1 history_list, giống đúng
    tiêu chí early stopping trong train_one_config: epoch có val_exact_match
    cao nhất (nếu nhiều epoch bằng nhau, lấy epoch SỚM NHẤT trong số đó, vì
    đó là epoch mà early stopping trong train.py sẽ chốt lại trước tiên).

    QUAN TRỌNG: dùng val_exact_match (tỷ lệ mẫu đúng TOÀN BỘ key_length bit
    message), KHÔNG dùng val_bit_acc (trung bình đúng/sai từng bit trên cả
    832 bit, gồm 672 bit parity không ai quan tâm) -- đây chính là điều đã
    được sửa trong train.py, không nên lặp lại nhầm lẫn đó ở đây.

    LƯU Ý: history[0] có thể là baseline (epoch=0, train_acc=None) - đo
    val_exact_match TRƯỚC khi fine-tune bất kỳ bước nào. Baseline này vẫn
    được xét vào best_val_exact/best_epoch (đúng ngữ nghĩa "cao nhất trong
    toàn bộ lịch sử"), nhưng KHÔNG tính vào n_epochs_ran vì nó không phải
    một epoch training thật."""
    best_val_exact = -1.0
    best_epoch = -1
    for entry in history:
        if entry["val_exact_match"] > best_val_exact:
            best_val_exact = entry["val_exact_match"]
            best_epoch = entry["epoch"]
    return best_val_exact, best_epoch


def main():
    histories = load_all_histories()

    results = []
    for trainable_iters, history in histories.items():
        best_val_exact, best_epoch = best_from_history(history)
        n_epochs_ran = sum(1 for entry in history if entry["epoch"] >= 1)
        baseline_entries = [e for e in history if e["epoch"] == 0]
        baseline_val_exact = (
            baseline_entries[0]["val_exact_match"] if baseline_entries else None
        )
        stopped_early = n_epochs_ran < 100  # N_EPOCHS mặc định trong train.py
        results.append(
            {
                "trainable_iters": trainable_iters,
                "baseline_val_exact_match": baseline_val_exact,
                "best_val_exact_match": best_val_exact,
                "best_epoch": best_epoch,
                "n_epochs_ran": n_epochs_ran,
                "stopped_early": stopped_early,
            }
        )

    results.sort(key=lambda r: r["best_val_exact_match"], reverse=True)

    print(f"\n{'='*80}")
    print(f"=== TỔNG HỢP {len(results)} CẤU HÌNH ĐÃ CHẠY (từ {OUT_DIR}) ===")
    print(f"{'='*80}")
    print(
        f"  {'trainable_iters':>16}  {'baseline_exact':>14}  {'best_exact_match':>17}  "
        f"{'best_epoch':>10}  {'epochs_ran':>10}  {'early_stop':>10}"
    )
    for r in results:
        baseline_str = (
            f"{r['baseline_val_exact_match']:.4f}"
            if r["baseline_val_exact_match"] is not None
            else "n/a"
        )
        print(
            f"  {r['trainable_iters']:>16d}  {baseline_str:>14}  "
            f"{r['best_val_exact_match']:>17.4f}  "
            f"{r['best_epoch']:>10d}  {r['n_epochs_ran']:>10d}  "
            f"{'có' if r['stopped_early'] else 'không':>10}"
        )

    summary_path = os.path.join(OUT_DIR, "sweep_summary_aggregated.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n(Đã ghi bảng tổng hợp đầy đủ vào: {summary_path})")

    if len(results) < 5:
        missing = sorted(
            set([4, 8, 12, 16, 20]) - set(r["trainable_iters"] for r in results)
        )
        if missing:
            print(
                f"\n⚠ Mới có {len(results)}/5 cấu hình. Còn thiếu "
                f"trainable_iters={missing} - bỏ comment các giá trị này trong "
                f"CANDIDATES của sweep_trainable_iters.py và chạy tiếp trước khi "
                f"kết luận cấu hình tốt nhất."
            )

    best = results[0]
    print(
        f"\n💡 Trong số các cấu hình ĐÃ CHẠY, tốt nhất: "
        f"trainable_iters={best['trainable_iters']} "
        f"(val_exact_match={best['best_val_exact_match']:.4f}, epoch={best['best_epoch']})"
    )
    print(
        f"   So với baseline (trước fine-tune) của chính cấu hình đó: "
        f"{best['baseline_val_exact_match']:.4f}"
    )
    print(f"   Chạy lại DUY NHẤT cấu hình này với save_weights=True để lưu trọng số:")
    print(f"   >>> from train import train_one_config")
    print(
        f"   >>> train_one_config(trainable_iters={best['trainable_iters']}, "
        f"save_weights=True)"
    )


if __name__ == "__main__":
    main()
