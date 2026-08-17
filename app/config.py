import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn tới module wifakey_module (có thể là symlink hoặc copy)
WIFAKEY_MODULE_PATH = BASE_DIR / "wifakey_module"

# Đường dẫn tới weights (nếu cần, nhưng wifakey_module đã có sẵn)
WEIGHTS_DIR = WIFAKEY_MODULE_PATH / "weights"  # tuỳ chỉnh theo cấu trúc thực tế