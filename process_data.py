import tarfile
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def process_data_streaming(limit=None):
    tar_path = 'train_images_labels_targets.tar'
    
    # Define Destination Paths
    base_dest = os.path.join('data', 'train')
    dirs = {
        'pre': os.path.join(base_dest, 'pre_images'),
        'post': os.path.join(base_dest, 'post_images'),
        'loc': os.path.join(base_dest, 'loc_masks'),
        'damage': os.path.join(base_dest, 'damage_masks')
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    print("Scanning tar file...")
    with tarfile.open(tar_path, 'r') as tar:
        # Get all members
        members = tar.getmembers()
        
        # Filter for pre-disaster images
        pre_images = [m for m in members if m.name.endswith('_pre_disaster.png') and 'images' in m.name]
        
        if limit:
            pre_images = pre_images[:limit]
            
        print(f"Found {len(pre_images)} samples to process.")
        
        for pre_member in tqdm(pre_images):
            try:
                # 1. Extract Pre Image
                f_pre = tar.extractfile(pre_member)
                if f_pre is None: continue
                pre_img_bytes = np.asarray(bytearray(f_pre.read()), dtype="uint8")
                pre_img = cv2.imdecode(pre_img_bytes, cv2.IMREAD_COLOR)
                
                # 2. Find and Extract Post Image
                post_name = pre_member.name.replace('_pre_disaster.png', '_post_disaster.png')
                try:
                    post_member = tar.getmember(post_name)
                    f_post = tar.extractfile(post_member)
                    post_img_bytes = np.asarray(bytearray(f_post.read()), dtype="uint8")
                    post_img = cv2.imdecode(post_img_bytes, cv2.IMREAD_COLOR)
                except KeyError:
                    continue # Post image not found
                
                # 3. Find and Extract Pre Target (Loc Mask)
                # Path in tar: train/targets/name_target.png
                # pre_member.name is train/images/name.png
                pre_target_name = pre_member.name.replace('images', 'targets').replace('.png', '_target.png')
                try:
                    pre_target_member = tar.getmember(pre_target_name)
                    f_pre_target = tar.extractfile(pre_target_member)
                    pre_target_bytes = np.asarray(bytearray(f_pre_target.read()), dtype="uint8")
                    loc_mask = cv2.imdecode(pre_target_bytes, cv2.IMREAD_GRAYSCALE)
                except KeyError:
                    continue
                    
                # 4. Find and Extract Post Target (Damage Mask)
                post_target_name = post_name.replace('images', 'targets').replace('.png', '_target.png')
                try:
                    post_target_member = tar.getmember(post_target_name)
                    f_post_target = tar.extractfile(post_target_member)
                    post_target_bytes = np.asarray(bytearray(f_post_target.read()), dtype="uint8")
                    damage_target = cv2.imdecode(post_target_bytes, cv2.IMREAD_GRAYSCALE)
                except KeyError:
                    continue

                # --- Processing ---
                filename = os.path.basename(pre_member.name)
                post_filename = os.path.basename(post_name)
                
                # Binary Loc Mask (>0 is building)
                binary_loc = (loc_mask > 0).astype(np.uint8) * 255
                
                # Binary Damage Mask (>=2 is damaged)
                # Assuming xView2 standard: 0=bg, 1=no-damage, 2=minor, 3=major, 4=destroyed
                binary_damage = (damage_target >= 2).astype(np.uint8) * 255
                
                # --- Saving ---
                cv2.imwrite(os.path.join(dirs['pre'], filename), pre_img)
                cv2.imwrite(os.path.join(dirs['post'], post_filename), post_img)
                cv2.imwrite(os.path.join(dirs['loc'], filename), binary_loc)
                cv2.imwrite(os.path.join(dirs['damage'], post_filename), binary_damage)
                
            except Exception as e:
                print(f"Error processing {pre_member.name}: {e}")
                continue

    print("Data processing complete.")

if __name__ == "__main__":
    # Process all found samples (or set a limit if disk space is still tight)
    process_data_streaming()
