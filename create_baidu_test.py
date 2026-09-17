import os
import shutil
import numpy as np
from pathlib import Path
from natsort import natsorted
from scipy.spatial.transform import Rotation

def _get_cop_pose(file):
    with open(file) as f:
        lines = f.readlines()
        
        # In Baidu .camera files, the XYZ coordinates (Center of Projection) 
        # are located on the second to last line.
        xyz_cop_line = lines[-2]
        xyz_cop = np.fromstring(xyz_cop_line, dtype=float, sep=' ')    
        
        # The 3x3 rotation matrix is located on lines 5, 6, and 7
        r1 = np.fromstring(lines[4], dtype=float, sep=' ')
        r2 = np.fromstring(lines[5], dtype=float, sep=' ')
        r3 = np.fromstring(lines[6], dtype=float, sep=' ')
        r =  Rotation.from_matrix(np.array([r1,r2,r3]))
        R_euler = r.as_euler('zyx', degrees=True)
    return xyz_cop, R_euler

def _parse_gt_folder(gt_folder):
    records = []
    camera_files = natsorted(os.listdir(gt_folder))
    for fname in camera_files:
        if not fname.endswith('.camera'):
            continue
        fpath = os.path.join(gt_folder, fname)
        try:
            xyz, euler = _get_cop_pose(fpath)
            img_stem = Path(fname).stem
            records.append({
                'img_stem': img_stem,
                'x': xyz[0],
                'fname': fname,
            })
        except Exception as e:
            print(f"Warning: Could not parse {fpath}: {e}")
    return records

def main():
    BAIDU_DIR = "E:/University/Year_3/Sem3/CV_InformationRetrieval/Course_Project/Datasets/baidu"
    BAIDU_TEST_DIR = "E:/University/Year_3/Sem3/CV_InformationRetrieval/Course_Project/Datasets/baidu_test"

    print("Parsing original Baidu ground truth folders to calculate Median X...")
    db_gt_dir = os.path.join(BAIDU_DIR, "training_gt")
    q_gt_dir  = os.path.join(BAIDU_DIR, "query_gt")

    db_records = _parse_gt_folder(db_gt_dir)
    q_records  = _parse_gt_folder(q_gt_dir)
    all_records = db_records + q_records

    # Extract the X coordinates from all camera poses
    all_x = [r['x'] for r in all_records]
    
    # Calculate the exact geographic midpoint (Median X coordinate)
    # The East Wing has X > median_x (used for training in VLAD-BuFF)
    # The West Wing has X <= median_x (used for testing/inference)
    median_x = float(np.median(all_x))
    print(f"Total images parsed: {len(all_records)} (DB: {len(db_records)}, Query: {len(q_records)})")
    print(f"Median X calculated: {median_x:.4f}")

    # Create directories
    print("Creating baidu_test folders...")
    os.makedirs(os.path.join(BAIDU_TEST_DIR, "training_images_undistort"), exist_ok=True)
    os.makedirs(os.path.join(BAIDU_TEST_DIR, "training_gt"), exist_ok=True)
    os.makedirs(os.path.join(BAIDU_TEST_DIR, "query_images_undistort"), exist_ok=True)
    os.makedirs(os.path.join(BAIDU_TEST_DIR, "query_gt"), exist_ok=True)
    os.makedirs(os.path.join(BAIDU_TEST_DIR, "out"), exist_ok=True)

    # Function to copy files
    def process_records(records, src_img_dir, src_gt_dir, dst_img_dir, dst_gt_dir):
        copied = 0
        for r in records:
            if r['x'] <= median_x:
                # Copy .camera file
                shutil.copy2(os.path.join(src_gt_dir, r['fname']), os.path.join(dst_gt_dir, r['fname']))
                
                # Copy image file. Find if it's .jpg or .png (Baidu is usually .jpg)
                img_name_jpg = r['img_stem'] + ".jpg"
                img_name_png = r['img_stem'] + ".png"
                if os.path.exists(os.path.join(src_img_dir, img_name_jpg)):
                    shutil.copy2(os.path.join(src_img_dir, img_name_jpg), os.path.join(dst_img_dir, img_name_jpg))
                elif os.path.exists(os.path.join(src_img_dir, img_name_png)):
                    shutil.copy2(os.path.join(src_img_dir, img_name_png), os.path.join(dst_img_dir, img_name_png))
                copied += 1
        return copied

    print("Copying test segment files (X <= median_x) into baidu_test...")
    copied_db = process_records(db_records, 
                    os.path.join(BAIDU_DIR, "training_images_undistort"), 
                    os.path.join(BAIDU_DIR, "training_gt"),
                    os.path.join(BAIDU_TEST_DIR, "training_images_undistort"),
                    os.path.join(BAIDU_TEST_DIR, "training_gt"))

    copied_q = process_records(q_records, 
                    os.path.join(BAIDU_DIR, "query_images_undistort"), 
                    os.path.join(BAIDU_DIR, "query_gt"),
                    os.path.join(BAIDU_TEST_DIR, "query_images_undistort"),
                    os.path.join(BAIDU_TEST_DIR, "query_gt"))

    print(f"Successfully copied DB images: {copied_db}")
    print(f"Successfully copied Query images: {copied_q}")

if __name__ == "__main__":
    main()
