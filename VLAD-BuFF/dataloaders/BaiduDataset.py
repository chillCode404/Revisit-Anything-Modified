# BaiduDataset.py
# Training dataset for the Baidu indoor mall dataset, compatible with GSVCitiesDataset interface.
# Designed to plug into the VLAD-BuFF training pipeline (Multi-Similarity Loss).
#
# Key design decisions:
# - Geographic train/test split: images are divided by the MEDIAN X coordinate,
#   computed dynamically at runtime from the full dataset.
#   Train -> East Wing (X > median_x), Test -> West Wing (X <= median_x)
# - Discrete place_id assignment via Grid Quantization:
#   XYZ coordinates are rounded to the nearest DIST_THRESH meters,
#   Euler angles to the nearest ANG_THRESH degrees.
#   Two images that fall in the same spatial+angular bin share a place_id (positive pair).

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
from natsort import natsorted


def _mat_to_euler_zyx(R):
    """
    Convert a 3x3 rotation matrix to ZYX Euler angles (degrees).
    Equivalent to scipy's Rotation.from_matrix(R).as_euler('zyx', degrees=True),
    but avoids a known hang in scipy 1.10.x.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([z, y, x]))
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- Hardcoded path to the Baidu dataset folder ----
# Adjust this to point to your local Baidu dataset root.
BAIDU_DIR = "E:/University/Year_3/Sem3/CV_InformationRetrieval/Course_Project/Datasets/baidu"

if not Path(BAIDU_DIR).exists():
    raise FileNotFoundError(
        f"BAIDU_DIR is hardcoded but does not exist: '{BAIDU_DIR}'. "
        "Please adjust BAIDU_DIR at the top of BaiduDataset.py."
    )

# ---- Thresholds for grouping images into places ----
# These mirror baidu_dataloader.py: dist_thresh=10, ang_thresh=20
DIST_THRESH = 10   # meters - spatial quantization bin size
ANG_THRESH  = 20   # degrees - angular quantization bin size

default_transform = T.Compose([
    T.Resize((224, 224)),  # Baidu images have varying sizes; resize to standard training size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# -----------------------------------------------------------------------
# Helper: parse a single .camera file -> (xyz_cop, euler_angles)
# Replicates get_cop_pose() from Revisit-Anything-Modified/dataloaders/baidu_dataloader.py
# -----------------------------------------------------------------------
def _get_cop_pose(camera_file: str):
    """
    Parse a Baidu .camera file.

    Returns:
        xyz_cop   (np.ndarray, shape [3]): Center-of-Projection in X, Y, Z
        R_euler   (np.ndarray, shape [3]): Euler angles (zyx convention, degrees)
    """
    with open(camera_file) as f:
        lines = f.readlines()
    # Last two lines: second-to-last is the XYZ CoP
    xyz_cop = np.fromstring(lines[-2], dtype=float, sep=' ')
    # Lines 4-6 (0-indexed): 3x3 rotation matrix
    r1 = np.fromstring(lines[4], dtype=float, sep=' ')
    r2 = np.fromstring(lines[5], dtype=float, sep=' ')
    r3 = np.fromstring(lines[6], dtype=float, sep=' ')
    rot = np.array([r1, r2, r3])
    R_euler = _mat_to_euler_zyx(rot)
    return xyz_cop, R_euler


# -----------------------------------------------------------------------
# Main Dataset Class
# -----------------------------------------------------------------------
class BaiduDataset(Dataset):
    """
    Baidu indoor mall dataset for VPR training with Multi-Similarity Loss.

    Interface is intentionally compatible with GSVCitiesDataset so it can be
    dropped into GSVCitiesDataloader (or a joint mixed dataloader) with minimal
    changes.

    Args:
        split (str): "train" or "test".
            - "train" -> East Wing images (X > median_x, ~1490 images)
            - "test"  -> West Wing images (X <= median_x, ~1490 images)
        img_per_place (int): Number of images to sample per place per batch item.
        min_img_per_place (int): Minimum images a place must have to be kept.
        random_sample_from_each_place (bool): If True, randomly sample K images;
            otherwise take the first K in sorted order.
        transform: Torchvision transform to apply to each image.
    """

    def __init__(
        self,
        split="train",
        img_per_place=4,
        min_img_per_place=4,
        random_sample_from_each_place=True,
        transform=default_transform,
    ):
        super().__init__()

        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"
        assert img_per_place <= min_img_per_place, (
            f"img_per_place ({img_per_place}) must be <= min_img_per_place ({min_img_per_place})"
        )

        self.split = split
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform

        # Build a lookup dict: image stem -> absolute image path (both db and query folders)
        self.img_lookup = self._build_image_lookup()

        # Build the main dataframe
        self.dataframe = self._build_dataframe()

        # Unique place ids after filtering
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_image_lookup(self):
        """
        Scan both image folders and build a dict of {stem -> absolute_path}.
        Filenames are guaranteed unique across both folders:
        - database images start with 'nikon'
        - query images do not
        """
        lookup = {}
        db_img_dir = os.path.join(BAIDU_DIR, "training_images_undistort")
        q_img_dir  = os.path.join(BAIDU_DIR, "query_images_undistort")

        for folder in (db_img_dir, q_img_dir):
            for fname in os.listdir(folder):
                stem = Path(fname).stem
                lookup[stem] = os.path.join(folder, fname)

        return lookup

    def _parse_gt_folder(self, gt_folder, is_query=False):
        """
        Parse all .camera files in a ground-truth folder.

        Returns:
            records (list of dicts): Each dict has keys:
                'img_stem', 'x', 'y', 'z', 'pitch', 'yaw', 'roll', 'is_query'
        """
        records = []
        camera_files = natsorted(os.listdir(gt_folder))
        for fname in camera_files:
            if not fname.endswith('.camera'):
                continue
            fpath = os.path.join(gt_folder, fname)
            try:
                xyz, euler = _get_cop_pose(fpath)
            except Exception as e:
                print(f"  WARNING: Could not parse '{fpath}': {e}")
                continue
            img_stem = Path(fname).stem  # filename without .camera extension
            records.append({
                'img_stem': img_stem,
                'x': xyz[0],
                'y': xyz[1],
                'z': xyz[2],
                'pitch': euler[0],
                'yaw':   euler[1],
                'roll':  euler[2],
                'is_query': is_query,
            })
        return records

    def _build_dataframe(self):
        """
        Build the full dataframe of images with place_id assigned.

        Steps:
        1. Parse all .camera files from both training_gt and query_gt.
        2. Compute median_x dynamically over the whole dataset.
        3. Filter to train or test wing.
        4. Assign a discrete place_id via grid quantization.
        5. Filter out places with fewer than min_img_per_place images.
        6. Print statistics about what was kept / dropped.
        """
        db_gt_dir = os.path.join(BAIDU_DIR, "training_gt")
        q_gt_dir  = os.path.join(BAIDU_DIR, "query_gt")

        print(f"[BaiduDataset] Parsing .camera files from:")
        print(f"  DB GT : {db_gt_dir}")
        print(f"  Q  GT : {q_gt_dir}")

        db_records = self._parse_gt_folder(db_gt_dir, is_query=False)
        q_records  = self._parse_gt_folder(q_gt_dir, is_query=True)
        all_records = db_records + q_records

        df_all = pd.DataFrame(all_records)
        total_images = len(df_all)
        print(f"[BaiduDataset] Total images parsed: {total_images} "
              f"(DB: {len(db_records)}, Query: {len(q_records)})")

        # --- Step 2: Compute median X dynamically over the full dataset ---
        median_x = float(np.median(df_all['x'].values))
        print(f"[BaiduDataset] Median X coordinate (geographic split threshold): {median_x:.4f}")

        # --- Step 3: Geographic split ---
        if self.split == "train":
            df = df_all[df_all['x'] > median_x].copy()
        else:
            df = df_all[df_all['x'] <= median_x].copy()
        wing_name = 'East Wing' if self.split == 'train' else 'West Wing'
        print(f"[BaiduDataset] Split='{self.split}' ({wing_name}): {len(df)} images retained")

        # --- Step 4: Grid quantization -> discrete place_id ---
        # Round XYZ to nearest DIST_THRESH.
        # NOTE: We intentionally do NOT include angular bins here.
        # For VPR training, two images are considered to depict the same "place" if
        # they are taken from the same spatial location (within DIST_THRESH meters),
        # regardless of camera orientation. Including angles would split the same
        # physical location into many unique bins, starving us of positive pairs.
        df['x_bin'] = np.round(df['x'] / DIST_THRESH).astype(int)
        df['y_bin'] = np.round(df['y'] / DIST_THRESH).astype(int)
        df['z_bin'] = np.round(df['z'] / DIST_THRESH).astype(int)

        # Create a composite key and convert to a factorized integer place_id
        df['_bin_key'] = list(zip(df['x_bin'], df['y_bin'], df['z_bin']))
        df['place_id'], _ = pd.factorize(df['_bin_key'])

        total_places_before = df['place_id'].nunique()

        # --- Step 5: Filter by min_img_per_place ---
        counts = df.groupby('place_id')['place_id'].transform('size')
        df_filtered = df[counts >= self.min_img_per_place].copy()

        total_places_after = df_filtered['place_id'].nunique()
        dropped_places     = total_places_before - total_places_after
        dropped_images     = len(df) - len(df_filtered)

        # --- Step 6: Print statistics ---
        print(f"[BaiduDataset] Grid quantization (DIST={DIST_THRESH}m):")
        print(f"  Places before filtering          : {total_places_before}")
        print(f"  Places dropped (< {self.min_img_per_place} imgs)     : {dropped_places}")
        print(f"  Images dropped                   : {dropped_images}")
        print(f"  Places after filtering           : {total_places_after}")
        print(f"  Images after filtering           : {len(df_filtered)}")
        
        num_db = len(df_filtered[df_filtered['is_query'] == False])
        num_q = len(df_filtered[df_filtered['is_query'] == True])
        print(f"  -> Database images (is_query=False) : {num_db}")
        print(f"  -> Query images (is_query=True)     : {num_q}")

        # Warn if any image stem has no matching image file
        missing = df_filtered['img_stem'].apply(lambda s: s not in self.img_lookup)
        if missing.any():
            print(f"  WARNING: {missing.sum()} image(s) have no matching file in the image folders!")

        # Set place_id as index (mirrors GSVCitiesDataset)
        return df_filtered.set_index('place_id')

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        """Number of unique places (not total images)."""
        return len(self.places_ids)

    def __getitem__(self, index):
        place_id = self.places_ids[index]

        # Get all rows belonging to this place
        place = self.dataframe.loc[place_id]

        # place may be a Series (single row) if only one image; wrap it
        if isinstance(place, pd.Series):
            place = place.to_frame().T

        # Sample K images from this place
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:
            place = place.iloc[: self.img_per_place]

        imgs = []
        for _, row in place.iterrows():
            img_stem = row['img_stem']
            img_path = self.img_lookup.get(img_stem)
            img = self._load_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        # Return shape: [K, C, H, W] and [K] place_id tensor
        # (Identical interface to GSVCitiesDataset.__getitem__)
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    @staticmethod
    def _load_image(path):
        if path is None:
            return Image.new("RGB", (224, 224))
        try:
            return Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            print(f"  WARNING: Could not load image '{path}', using blank.")
            return Image.new("RGB", (224, 224))
