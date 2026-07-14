import os
import cv2
import numpy as np
import sys
from tqdm import tqdm
import glob
import itertools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = r"G:\archive_1\test" 

try:
    from vision_module.face_processor import FaceProcessor
    from feature_extractor.adaface_handler import AdaFaceExtractor
    from wifakey_module.wifakey_handler import WiFaKeyHandler
except ImportError as e:
    print(f"[Import error:] {e}")
    sys.exit(1)

# Global models
face_processor = None
adaface = None
wifakey = None

def init_models():
    global face_processor, adaface, wifakey
    print("⏳ Initializing models...")
    try:
        face_processor = FaceProcessor()
        adaface = AdaFaceExtractor()
        wifakey = WiFaKeyHandler()
        print("✅ Models are ready.")
        return True
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return False

def load_and_extract_all(folder_path):
    print(f"\n📂 Loading all images from: {folder_path}")
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    
    if not image_paths:
        print("❌ No images found!")
        return []

    valid_features = []
    for img_path in tqdm(image_paths, desc="Extracting features"):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue

        aligned_face, status = face_processor.process(img)
        if aligned_face is None:
            try:
                aligned_face = cv2.resize(img, (112, 112))
                aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            except: continue
        
        try:
            vector = adaface.get_feature_vector(aligned_face)
            valid_features.append({"name": filename, "vector": vector})
        except: continue
            
    return valid_features

def run_inter_class_test():
    if not init_models():
        return

    features = load_and_extract_all(DATASET_ROOT)
    num_samples = len(features)
    if num_samples < 2:
        print("⚠️ Need at least 2 images to test.")
        return

    pairs_iter = list(itertools.combinations(range(num_samples), 2))
    total_pairs_count = len(pairs_iter)

    print("\n" + "="*60)
    print("🚀 STARTING INTER-CLASS TEST (IMPOSTOR TEST)")
    print(f"   Total images: {num_samples}")
    print(f"   Total impostor pairs: {total_pairs_count}")
    print("="*60)

    true_rejects = 0  
    false_accepts = 0 
    dangerous_pairs = [] 

    pbar = tqdm(pairs_iter, desc="Running impostor attacks")

    for i, j in pbar:
        user_A = features[i] 
        user_B = features[j] 
        
        try:
            helper_data, mask_r, key_hash = wifakey.enroll(user_A['vector'])
        except Exception as e:
            continue 

        try:
            is_match = wifakey.verify(
                user_B['vector'], 
                helper_data, 
                mask_r, 
                key_hash
            )
            
            if is_match:
                false_accepts += 1
                dangerous_pairs.append(f"{user_A['name']} <-> {user_B['name']}")
            else:
                true_rejects += 1
        except:
            true_rejects += 1

    print("\n" + "="*60)
    print("📊 SECURITY RESULTS (INTER-CLASS TEST)")
    print("="*60)
    print(f"🛡️ True Rejects : {true_rejects}")
    print(f"🚨 False Accepts     : {false_accepts}")
    print("-" * 40)

    if total_pairs_count > 0:
        far = (false_accepts / total_pairs_count) * 100
        print(f"📉 False Accept Rate (FAR) : {far:.4f}%")
        
    else:
        print("⚠️ No test data available.")
if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        run_inter_class_test()
    else:
        print(f"❌ Path does not exist: {DATASET_ROOT}")