"""
sweep_trainable_iters.py

Thay vì đoán trainable_iters theo cảm tính rồi chạy run_ab thủ công từng
lần, script này quét qua nhiều giá trị trainable_iters, mỗi giá trị tự
train với early stopping (dựa trên VAL, tách từ tập 'tune'), rồi so sánh
val_bit_acc để chọn cấu hình tốt nhất - hoàn toàn tự động, không cần đoán.

QUAN TRỌNG: việc chọn trainable_iters này VẪN chỉ dùng tập 'tune' (train+val
nội bộ), KHÔNG đụng vào tập 'select'. Sau khi sweep xong và chọn được cấu
hình tốt nhất, mới chạy run_ab_soft_llr.py (dùng tập 'select') ĐÚNG 1 LẦN
để có con số so sánh cuối cùng, khách quan.

Cách chạy:
    python research/decoder/v1_neural_ms_finetuned/sweep_trainable_iters.py
"""

import os
import json
from train import train_one_config, ITERS_MAX

CANDIDATES = [
    4,
    # 8,
    # 12,
    # 16,
    # 20,
]  # số vòng lặp cuối được phép fine-tune, thử từng giá trị
OUT_DIR = os.path.join(os.path.dirname(__file__), "sweep_results")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for trainable_iters in CANDIDATES:
        print(f"\n{'='*60}")
        print(f"=== Thử trainable_iters={trainable_iters} ===")
        print(f"{'='*60}")

        best_val_acc, best_epoch, history = train_one_config(
            trainable_iters=trainable_iters,
            n_epochs=100,
            patience=8,
            verbose=True,
            save_weights=True,  # Chưa lưu trọng số ở bước sweep - chỉ để chọn cấu hình
            learning_rate=5e-4,
        )

        results.append(
            {
                "trainable_iters": trainable_iters,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            }
        )

        with open(
            os.path.join(OUT_DIR, f"history_iters{trainable_iters}.json"), "w"
        ) as f:
            json.dump(history, f, indent=2)

    results.sort(key=lambda r: r["best_val_acc"], reverse=True)

    print(f"\n{'='*60}")
    print("=== KẾT QUẢ SWEEP (sắp xếp theo val_acc giảm dần) ===")
    print(f"{'='*60}")
    for r in results:
        print(
            f"  trainable_iters={r['trainable_iters']:3d}  "
            f"best_val_acc={r['best_val_acc']:.4f}  best_epoch={r['best_epoch']}"
        )

    with open(os.path.join(OUT_DIR, "sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    best = results[0]
    print(
        f"\n💡 Cấu hình tốt nhất: trainable_iters={best['trainable_iters']} "
        f"(val_acc={best['best_val_acc']:.4f})"
    )
    print(f"   Chạy lại DUY NHẤT cấu hình này với save_weights=True để lưu")
    print(f"   trọng số cuối cùng, ví dụ:")
    print(f"   >>> from train import train_one_config")
    print(
        f"   >>> train_one_config(trainable_iters={best['trainable_iters']}, "
        f"save_weights=True)"
    )


if __name__ == "__main__":
    main()
