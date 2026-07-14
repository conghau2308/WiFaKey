import os
import cv2
import numpy as np
import sys
from tqdm import tqdm
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = r"G:\archive_3\Selfies ID Images dataset"

try:
    from vision_module.face_processor import FaceProcessor
    from feature_extractor.adaface_handler import AdaFaceExtractor
    from wifakey_module.wifakey_handler import WiFaKeyHandler
except ImportError as e:
    print(f"[IMPORT ERROR] {e}")
    print("Please ensure this file is located in the root directory of the WiFaKey project.")
    sys.exit(1)

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
        print("✅ Models are ready on GPU.")
        return True
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return False

def process_single_user(user_folder_path, user_id):

    image_paths = sorted(glob.glob(os.path.join(user_folder_path, "*.jpg")))
    
    if len(image_paths) < 2:
        return 0, 0, 0 # pairs, success, fail

    valid_features = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None: continue

        aligned_face, status = face_processor.process(img)
        if aligned_face is None:
            continue

        try:
            vector = adaface.get_feature_vector(aligned_face)
            valid_features.append({
                "name": filename,
                "vector": vector
            })
        except:
            continue
            
    if len(valid_features) < 2:
        return 0, 0, 0

    local_pairs = 0
    local_success = 0
    local_fail = 0
    
    num_samples = len(valid_features)
    
    for i in range(num_samples):
        enroll_item = valid_features[i]
        try:
            helper_data, mask_r, key_hash = wifakey.enroll(enroll_item['vector'])
        except Exception as e:
            print(f"❌ Enroll Error User {user_id}: {e}")
            continue

        for j in range(i + 1, num_samples):
            verify_item = valid_features[j]
            local_pairs += 1
            
            try:
                is_match = wifakey.verify(
                    verify_item['vector'], 
                    helper_data, 
                    mask_r, 
                    key_hash
                )
                
                if is_match:
                    local_success += 1
                else:
                    local_fail += 1
                    print(f"   ❌ User {user_id}: {enroll_item['name']} vs {verify_item['name']} -> Failed")
            except Exception as e:
                local_fail += 1

    return local_pairs, local_success, local_fail

def run_dataset_test():
    print("="*60)
    print(f"🚀 STARTING DATASET TEST: {DATASET_ROOT}")
    print("="*60)
    
    if not init_models():
        return

    subfolders = [f.path for f in os.scandir(DATASET_ROOT) if f.is_dir() and f.name.isdigit()]
    
    subfolders.sort(key=lambda f: int(os.path.basename(f)))
    
    print(f"📂 Found {len(subfolders)} users (folders).")
    
    total_dataset_pairs = 0
    total_dataset_success = 0
    total_dataset_fail = 0
    users_with_error = []

    pbar = tqdm(subfolders, desc="Processing each User")

    for folder_path in pbar:
        user_id = os.path.basename(folder_path)
        
        pairs, success, fail = process_single_user(folder_path, user_id)
        
        total_dataset_pairs += pairs
        total_dataset_success += success
        total_dataset_fail += fail
        
        if fail > 0:
            users_with_error.append(user_id)

    print("\n" + "="*60)
    print("📊 RESULTS FOR THE ENTIRE DATASET")
    print("="*60)
    print(f"👥 Total users tested          : {len(subfolders)}")
    print(f"🔗 Total pairs compared        : {total_dataset_pairs}")
    print("-" * 40)
    print(f"✅ Total Matches (Success)     : {total_dataset_success}")
    print(f"❌ Total Mismatches (Fail)     : {total_dataset_fail}")
    print("-" * 40)

    if total_dataset_pairs > 0:
        accuracy = (total_dataset_success / total_dataset_pairs) * 100
        frr = 100 - accuracy
        
        print(f"🎯 True Acceptance Rate (TAR): {accuracy:.2f}%")
        print(f"📉 False Rejection Rate (FRR) : {frr:.2f}%")
        
        if len(users_with_error) > 0:
            print("\n⚠️ Users with failed image pairs (Verify failed):")
            print(f"   {users_with_error[:20]} ... (Total: {len(users_with_error)})")
    else:
        print("⚠️ No valid image pairs found for testing.")

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        run_dataset_test()
    else:
        print(f"❌ Path does not exist: {DATASET_ROOT}")