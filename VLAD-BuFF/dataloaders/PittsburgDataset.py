import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors

# 25 meters is standard for Pittsburgh dataset evaluations and training positive groupings
DIST_THRESH = 25.0  

default_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class PittsburgDataset(Dataset):
    """
    Unified dataloader for Pittsburgh (Pitts30k) that directly parses .npy files.
    
    If split == 'train':
        Behaves like GSVCitiesDataset: groups images by 25m spatial grids into place_ids,
        and __getitem__ returns a batch of [K, C, H, W] images and their place_id.
    
    If split in ['val', 'test']:
        Behaves like standard evaluation datasets: loads database and queries,
        provides getPositives() for recall calculation, and __getitem__ returns (img, index).
    """
    def __init__(self, 
                 root_dir="E:/University/Year_3/Sem3/CV_InformationRetrieval/Course_Project/Datasets/pitts", 
                 dataset_name="pitts_small",
                 split="val", 
                 img_per_place=4, 
                 min_img_per_place=4, 
                 transform=default_transform):
        
        super().__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        
        # Train parameters
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        
        self.npy_dir = os.path.join(self.root_dir, self.dataset_name, "images", self.split)
        
        if self.split == "train":
            self._init_train()
        elif self.split in ["val", "test"]:
            self._init_eval()
        else:
            raise ValueError(f"Unknown split: {split}")

    def _get_utm(self, paths):
        coords = []
        for path in paths:
            # Extract coordinates from filenames like '@0584291.21@4476924.92@...'
            # Index 1: UTM Easting, Index 2: UTM Northing (in meters)
            utm_e = float(path.split('@')[1])
            utm_n = float(path.split('@')[2])
            coords.append([utm_e, utm_n])
        return np.array(coords)

    def _init_train(self):
        db_npy_path = os.path.join(self.npy_dir, "database.npy")
        q_npy_path = os.path.join(self.npy_dir, "queries.npy")
        if not os.path.exists(db_npy_path):
            raise FileNotFoundError(f"Training database not found at: {db_npy_path}")
            
        print(f"[PittsburgDataset - TRAIN] Loading {db_npy_path}")
        img_paths_raw = list(np.load(db_npy_path))
        
        # Add queries if they exist to pool all available viewpoints for training
        if os.path.exists(q_npy_path):
            print(f"[PittsburgDataset - TRAIN] Loading {q_npy_path}")
            q_paths_raw = list(np.load(q_npy_path))
            # The paths in .npy might be absolute from a different machine.
            img_paths = [os.path.join(self.npy_dir, "database", os.path.basename(p)) for p in img_paths_raw] + \
                        [os.path.join(self.npy_dir, "queries", os.path.basename(p)) for p in q_paths_raw]
            img_paths_raw = img_paths_raw + q_paths_raw
        else:
            img_paths = [os.path.join(self.npy_dir, "database", os.path.basename(p)) for p in img_paths_raw]
        
        # Parse coordinates
        coords = self._get_utm(img_paths_raw)
        
        df = pd.DataFrame({
            'img_path': img_paths,
            'utm_e': coords[:, 0],
            'utm_n': coords[:, 1]
        })
        
        # Grid quantization (25m threshold)
        df['utm_e_bin'] = np.round(df['utm_e'] / DIST_THRESH).astype(int)
        df['utm_n_bin'] = np.round(df['utm_n'] / DIST_THRESH).astype(int)
        df['_bin_key'] = list(zip(df['utm_e_bin'], df['utm_n_bin']))
        df['place_id'], _ = pd.factorize(df['_bin_key'])
        
        total_places_before = df['place_id'].nunique()
        
        # Filter places with < min_img_per_place
        place_counts = df['place_id'].value_counts()
        valid_places = place_counts[place_counts >= self.min_img_per_place].index
        df_filtered = df[df['place_id'].isin(valid_places)].copy()
        
        # Re-factorize to ensure continuous IDs
        df_filtered['place_id'], _ = pd.factorize(df_filtered['place_id'])
        
        total_places_after = df_filtered['place_id'].nunique()
        dropped_places = total_places_before - total_places_after
        dropped_images = len(df) - len(df_filtered)
        
        print(f"[PittsburgDataset] Grid quantization (DIST={DIST_THRESH}m):")
        print(f"  Places before filtering          : {total_places_before}")
        print(f"  Places dropped (< {self.min_img_per_place} imgs)     : {dropped_places}")
        print(f"  Images dropped                   : {dropped_images}")
        print(f"  Places after filtering           : {total_places_after}")
        print(f"  Images after filtering           : {len(df_filtered)}")
        
        # Group by place_id for fast fetching
        self.places_dict = df_filtered.groupby('place_id').apply(
            lambda x: x['img_path'].tolist()
        ).to_dict()
        
        self.total_nb_images = len(df_filtered)
        self.num_places = total_places_after
        self.df = df_filtered

    def _init_eval(self):
        db_npy_path = os.path.join(self.npy_dir, "database.npy")
        q_npy_path = os.path.join(self.npy_dir, "queries.npy")
        
        print(f"[PittsburgDataset - EVAL] Loading {db_npy_path} and {q_npy_path}")
        db_paths_raw = np.load(db_npy_path)
        q_paths_raw = np.load(q_npy_path)
        
        self.db_paths = [os.path.join(self.npy_dir, "database", os.path.basename(p)) for p in db_paths_raw]
        self.q_paths = [os.path.join(self.npy_dir, "queries", os.path.basename(p)) for p in q_paths_raw]
        
        self.images = self.db_paths + self.q_paths
        
        self.numDb = len(self.db_paths)
        self.numQ = len(self.q_paths)
        
        self.utmDb = self._get_utm(db_paths_raw)
        self.utmQ = self._get_utm(q_paths_raw)
        
        self.positives = None

    def __len__(self):
        if self.split == "train":
            return self.num_places
        else:
            return len(self.images)

    def __getitem__(self, index):
        if self.split == "train":
            place_id = index
            img_paths = self.places_dict[place_id]
            
            # Randomly sample K images
            sampled_paths = np.random.choice(img_paths, size=self.img_per_place, replace=False)
            
            imgs = []
            for path in sampled_paths:
                img = self._load_image(path)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)
                
            return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)
            
        else:
            path = self.images[index]
            img = self._load_image(path)
            
            if self.transform is not None:
                img = self.transform(img)
                
            return img, index
            
    def getPositives(self):
        """ Used only during evaluation """
        if self.split == "train":
            raise ValueError("getPositives is not available for training split")
            
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.utmDb)
            distances, self.positives = knn.radius_neighbors(self.utmQ, radius=DIST_THRESH)
            
        return self.positives
        
    @staticmethod
    def _load_image(path):
        if path is None:
            return Image.new("RGB", (224, 224))
        try:
            return Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            print(f"  WARNING: Could not load image '{path}', using blank.")
            return Image.new("RGB", (224, 224))
        except FileNotFoundError:
            print(f"  WARNING: Image not found '{path}', using blank.")
            return Image.new("RGB", (224, 224))

# =====================================================================
# Compatibility Helpers (expected by VLAD-BuFF/train.py eval loops)
# =====================================================================

def get_whole_val_set(input_transform):
    return PittsburgDataset(split="val", transform=input_transform)

def get_whole_test_set(input_transform):
    return PittsburgDataset(split="test", transform=input_transform)

def get_250k_val_set(input_transform):
    return PittsburgDataset(dataset_name="pitts250k", split="val", transform=input_transform)

def get_250k_test_set(input_transform):
    return PittsburgDataset(dataset_name="pitts250k", split="test", transform=input_transform)

def get_whole_training_set(onlyDB=False):
    return PittsburgDataset(split="train", transform=default_transform)
