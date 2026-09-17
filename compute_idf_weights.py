"""
Compute IDF (Inverse Document Frequency) Weights for VLAD Cluster Centers
=========================================================================

This script computes the IDF weight for each VLAD cluster center by scanning
the entire database (reference) images of a given dataset. For each image,
all DINOv2 pixel descriptors are assigned to their nearest cluster center
(hard assignment via cosine similarity). The script then counts how many
images contain each cluster, and computes:

    w_k = log(N / df_k)

where N is the total number of images and df_k is the number of images
containing at least one descriptor assigned to cluster k.

Usage:
    python compute_idf_weights.py --dataset <dataset_name> --vocab-vlad <map|domain>

Output:
    - Saves a dictionary to a .pt file containing:
        - 'idf_weights': Tensor of shape [num_clusters] with IDF weights
        - 'cluster_doc_freq': Tensor of shape [num_clusters] with raw counts
        - 'total_images': int, total number of database images
        - 'dataset': str, dataset name
        - 'domain': str, VLAD vocabulary domain used
    - Also saves a visualization bar chart of the IDF weights.
"""

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import argparse
import os
from typing import Literal
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

from place_rec_global_config import datasets, workdir_data


def compute_idf_weights(
    dataset_name: str,
    vocab_vlad: str = 'map',
    is_finetuned: bool = False,
    desc_layer: int = 31,
    desc_facet: str = 'value',
    num_c: int = 32,
    cache_dir: str = './cache',
    device: str = 'cuda',
):
    """
    Compute IDF weights for VLAD cluster centers over the database.

    For each cluster center k (k=0..num_c-1), counts in how many database
    images at least one pixel descriptor is assigned to cluster k.
    Then computes IDF = log(N / df_k) for each cluster.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (must be a key in `datasets` config).
    vocab_vlad : str
        Vocabulary choice: 'domain' or 'map'.
    desc_dim : int
        DINOv2 descriptor dimension (default 1536 for ViT-G/14).
    desc_layer : int
        Layer index for descriptor extraction.
    desc_facet : str
        Facet for descriptor extraction.
    num_c : int
        Number of VLAD cluster centers.
    cache_dir : str
        Path to the cache directory containing vocabulary.
    device : str
        Torch device ('cuda' or 'cpu').

    Returns
    -------
    dict with keys:
        'idf_weights', 'cluster_doc_freq', 'total_images',
        'dataset', 'domain'
    """
    # =========================================================================
    # 1. Load dataset configuration
    # =========================================================================
    dataset_config = datasets.get(dataset_name, {})
    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")

    domain = (dataset_config['domain_vlad_cluster'] if vocab_vlad == 'domain'
              else dataset_config['map_vlad_cluster'])
    if is_finetuned:
        domain = domain + "NVFinetuned"
        desc_dim = 768
        dino_key = 'dinoNV_h5_filename_r'
    else:
        desc_dim = 1536
        dino_key = 'dino_h5_filename_r'
    print(f"[IDF] Dataset: {dataset_name}, Domain: {domain}, Vocab: {vocab_vlad}")

    # =========================================================================
    # 2. Load cluster centers
    # =========================================================================
    ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
    c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier,
                                 domain, "c_centers.pt")
    assert os.path.isfile(c_centers_file), \
        f"Cluster centers not found at {c_centers_file}!"

    c_centers = torch.load(c_centers_file, map_location=device)
    assert c_centers.shape[0] == num_c, \
        f"Expected {num_c} clusters, got {c_centers.shape[0]}"
    print(f"[IDF] Loaded cluster centers: {c_centers.shape} from {c_centers_file}")

    # Normalize cluster centers for cosine similarity assignment
    c_centers_norm = F.normalize(c_centers, dim=1).to(device)

    # =========================================================================
    # 3. Load DINOv2 descriptors (h5 file for reference/database images)
    # =========================================================================
    workdir = f'{workdir_data}/{dataset_name}/out'
    dino_r_path = f"{workdir}/{dataset_config[dino_key]}"
    assert os.path.isfile(dino_r_path), \
        f"DINO descriptors not found at {dino_r_path}!"

    f = h5py.File(dino_r_path, "r")
    keys = list(f.keys())
    total_images = len(keys)
    print(f"[IDF] Total database images: {total_images}")
    print(f"[IDF] Number of cluster centers: {num_c}")

    # =========================================================================
    # 4. Count cluster document frequency
    # =========================================================================
    # cluster_doc_freq[k] = number of images containing at least one
    # descriptor assigned to cluster k
    cluster_doc_freq = torch.zeros(num_c, dtype=torch.long)

    # Optional: also track per-image cluster histograms for analysis
    # per_image_cluster_hist[img_idx, k] = number of descriptors in cluster k
    per_image_cluster_presence = torch.zeros(total_images, num_c,
                                              dtype=torch.bool)

    print("[IDF] Scanning database images for cluster assignments...")
    for img_idx in tqdm(range(total_images), desc="Computing cluster stats"):
        key = keys[img_idx]

        # Load DINOv2 descriptors for this image
        dino_desc = torch.from_numpy(f[key]['ift_dino'][()]).to(device)
        # Shape: [1, desc_dim, H_patches, W_patches]
        total_elements = dino_desc.shape[2] * dino_desc.shape[3]
        dino_desc = dino_desc.reshape(1, desc_dim, total_elements)

        # Normalize descriptors (L2 normalize along feature dimension)
        dino_desc_norm = F.normalize(dino_desc, dim=1)
        # Shape: [total_elements, desc_dim]
        descs = dino_desc_norm.squeeze(0).permute(1, 0)  # [N_pixels, desc_dim]

        # Hard cluster assignment via cosine similarity
        # labels[i] = argmax of cosine similarity between desc[i] and each center
        labels = torch.argmax(descs @ c_centers_norm.T, dim=1)  # [N_pixels]

        # Find which clusters are present in this image
        unique_clusters = torch.unique(labels)
        per_image_cluster_presence[img_idx, unique_clusters.cpu()] = True
        cluster_doc_freq[unique_clusters.cpu()] += 1

        # Free GPU memory
        del dino_desc, dino_desc_norm, descs, labels
        if device == 'cuda' and img_idx % 50 == 0:
            torch.cuda.empty_cache()

    f.close()

    # =========================================================================
    # 5. Compute IDF weights
    # =========================================================================
    # IDF = log(N / df_k), with smoothing to avoid log(0) or division by zero
    # Using standard IDF formula: log(N / df_k)
    # For clusters that never appear (df_k=0), set IDF to max (log(N))
    idf_weights = torch.zeros(num_c, dtype=torch.float32)
    for k in range(num_c):
        if cluster_doc_freq[k] > 0:
            idf_weights[k] = torch.log(
                torch.tensor(total_images / cluster_doc_freq[k].float())
            )
        else:
            # Cluster never appears — assign maximum possible IDF
            idf_weights[k] = torch.log(torch.tensor(float(total_images)))
            print(f"  [WARNING] Cluster {k} has zero document frequency!")

    # =========================================================================
    # 6. Print summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"  IDF WEIGHT STATISTICS for {dataset_name} (domain={domain})")
    print("=" * 70)
    print(f"  Total database images (N): {total_images}")
    print(f"  Number of clusters:        {num_c}")
    print(f"  {'Cluster':>8s} | {'Doc Freq':>10s} | {'% Images':>10s} | {'IDF Weight':>10s}")
    print(f"  {'-'*8:>8s} | {'-'*10:>10s} | {'-'*10:>10s} | {'-'*10:>10s}")

    # Sort by IDF weight (descending) for display
    sorted_indices = torch.argsort(idf_weights, descending=True)
    for idx in sorted_indices:
        k = idx.item()
        df = cluster_doc_freq[k].item()
        pct = 100.0 * df / total_images
        idf = idf_weights[k].item()
        print(f"  {k:>8d} | {df:>10d} | {pct:>9.1f}% | {idf:>10.4f}")

    print(f"\n  IDF range: [{idf_weights.min().item():.4f}, {idf_weights.max().item():.4f}]")
    print(f"  Mean IDF:  {idf_weights.mean().item():.4f}")
    print(f"  Std IDF:   {idf_weights.std().item():.4f}")
    print("=" * 70)

    # =========================================================================
    # 7. Save results
    # =========================================================================
    output_dir = f"{workdir}/idf_weights"
    os.makedirs(output_dir, exist_ok=True)

    save_data = {
        'idf_weights': idf_weights,                    # [num_c]
        'cluster_doc_freq': cluster_doc_freq,           # [num_c]
        'total_images': total_images,
        'dataset': dataset_name,
        'domain': domain,
        'vocab_vlad': vocab_vlad,
        'num_clusters': num_c,
        'desc_dim': desc_dim,
        'c_centers_file': c_centers_file,
        'per_image_cluster_presence': per_image_cluster_presence,  # [N_imgs, num_c]
    }

    save_path = f"{output_dir}/{dataset_name}_idf_weights_domain_{domain}.pt"
    torch.save(save_data, save_path)
    print(f"\n[IDF] Results saved to: {save_path}")

    # =========================================================================
    # 8. Visualization (wrapped in try/except due to potential matplotlib issues)
    # =========================================================================
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # --- Bar chart of IDF weights ---
        ax1 = axes[0]
        colors = plt.cm.RdYlGn(idf_weights / idf_weights.max())  # Green=high IDF (rare), Red=low IDF (common)
        bars = ax1.bar(range(num_c), idf_weights.numpy(), color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Cluster Index', fontsize=12)
        ax1.set_ylabel('IDF Weight', fontsize=12)
        ax1.set_title(f'IDF Weights per VLAD Cluster — {dataset_name} (domain={domain})', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(num_c))
        ax1.grid(axis='y', alpha=0.3)

        # --- Bar chart of document frequency ---
        ax2 = axes[1]
        df_pct = 100.0 * cluster_doc_freq.float() / total_images
        colors2 = plt.cm.RdYlGn_r(df_pct / 100.0)  # Red=high freq (common), Green=low freq (rare)
        ax2.bar(range(num_c), df_pct.numpy(), color=colors2, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Cluster Index', fontsize=12)
        ax2.set_ylabel('% of Images Containing Cluster', fontsize=12)
        ax2.set_title(f'Cluster Document Frequency — {dataset_name} (domain={domain})', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(num_c))
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% (appears in all images)')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig_path = f"{output_dir}/{dataset_name}_idf_visualization_domain_{domain}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[IDF] Visualization saved to: {fig_path}")
    except Exception as e:
        print(f"[IDF] Warning: Could not create visualization: {e}")
        print("[IDF] The IDF data (.pt file) was saved successfully regardless.")

    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute IDF weights for VLAD cluster centers over the database. '
                    'See place_rec_global_config.py for dataset configurations.'
    )
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (e.g., 17places, baidu, pitts)')
    parser.add_argument('--vocab-vlad', required=True,
                        choices=['domain', 'map'],
                        help='Vocabulary choice for VLAD. Options: map, domain.')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for computation (default: cuda)')
    parser.add_argument('--finetuned', action='store_true',
                        help='Use this flag if extracting weights for a fine-tuned DINOv2 model (768-d)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Computing IDF Weights for VLAD Cluster Centers")
    print(f"  Dataset: {args.dataset}")
    print(f"  Vocabulary: {args.vocab_vlad}")
    print(f"  Device: {args.device}")
    print(f"{'='*70}\n")

    result = compute_idf_weights(
        dataset_name=args.dataset,
        vocab_vlad=args.vocab_vlad,
        is_finetuned=args.finetuned,
        device=args.device,
    )

    print(f"\n[DONE] IDF computation complete for {args.dataset}!")
    print(f"  Use the saved .pt file in your retrieval pipeline as segment weights.")
    print(f"  Example usage:")
    print(f"    idf_data = torch.load('<path_to_idf_weights>.pt')")
    print(f"    idf_weights = idf_data['idf_weights']  # shape: [{result['num_clusters']}]")
    print(f"    # For a segment assigned to cluster k:")
    print(f"    #   w_s = idf_weights[k]")
