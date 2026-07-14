import os
import sys
import shutil
import torch
import numpy as np
from transformers import AutoModel
from huggingface_hub import hf_hub_download

# --- FUNCTION FROM HUGGING FACE
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            print(f"📥 Đang tải: {file}...")
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model

def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download and os.path.exists(save_path):
        shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

class AdaFaceExtractor:
    def __init__(self, model_path=None, device='cuda'):
        
        if device == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔌 Thiết bị sử dụng cho AdaFace: {self.device}")

        if model_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, "model")
            
        self.repo_id = 'minchul/cvlface_adaface_ir101_webface12m'
        
        try:
            print(f"🔄 Đang tải/load model từ repo: {self.repo_id}...")
            print(f"📂 Cache folder: {model_path}")
            
            self.model = load_model_by_repo_id(self.repo_id, model_path)
            
            self.model.eval()
            self.model.to(self.device)
            
            print("✅ Model AdaFace (HuggingFace) đã sẵn sàng.")
            
        except Exception as e:
            print(f"❌ Lỗi khi load model HuggingFace: {e}")
            raise e

    def _preprocess_image(self, rgb_image: np.ndarray) -> torch.Tensor:

        if rgb_image is None or rgb_image.shape != (112, 112, 3):
             raise ValueError(f"Đầu vào phải là ảnh 112x112. Nhận được: {rgb_image.shape if rgb_image is not None else 'None'}")

        tensor = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float()

        tensor = tensor.unsqueeze(0)
        
        tensor.div_(255).sub_(0.5).div_(0.5)
        
        return tensor

    def get_feature_vector(self, processed_face_image: np.ndarray) -> np.ndarray:
        # Preprocess
        input_tensor = self._preprocess_image(processed_face_image)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            features = self.model(input_tensor)
            
            # Check if features is a tuple (embedding, norm)
            if isinstance(features, tuple):
                features = features[0]
        
        # Post-process: Tensor -> Numpy
        feature_vector = features.squeeze().cpu().numpy()

        # L2 Normalization
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm

        return feature_vector