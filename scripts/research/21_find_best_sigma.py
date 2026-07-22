import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_handler import WiFaKeyHandler
from research.modulation.v2_symbol_level_llr import calibrate_sigma_from_ber
from research.decoder.v1_neural_ms_finetuned.train import load_tune_genuine_pairs

handler = WiFaKeyHandler()
train_pairs, _ = load_tune_genuine_pairs()
best_sigma, results = calibrate_sigma_from_ber(
    handler, train_pairs, sigma_candidates=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
)
print("sigma tốt nhất:", best_sigma)
for r in results:
    print(f"  sigma={r[0]:.4f}  BER_dự_đoán={r[1]:.4f}  BER_thật={r[2]:.4f}")
