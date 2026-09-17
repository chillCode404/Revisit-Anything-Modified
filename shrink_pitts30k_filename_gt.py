"""
shrink_pitts30k_filename_gt.py
--------------------------------
Dành cho format Pitts30k kiểu "datasets_vg" (gmberton / deep-visual-geo-
localization-benchmark / CosPlace / EigenPlaces), nơi ground truth (toạ
độ UTM) được nhúng ngay trong TÊN FILE ảnh, dạng:

    @utm_east@utm_north@zone_num@zone_letter@lat@lon@pano_id@tile@...@.jpg

Vì vậy KHÔNG cần file index/gt riêng: chỉ cần chọn đúng tập con ảnh
(database + queries) sao cho mỗi query còn ít nhất 1 positive trong
database đã giữ, rồi copy sang thư mục mới với tên file giữ nguyên.
Loader gốc sẽ tự tính lại ground truth từ tên file như bình thường.

Cách dùng (chạy riêng cho từng split: train / val / test):

    python3 shrink_pitts30k_filename_gt.py \
        --db-dir  images/val/database \
        --q-dir   images/val/queries \
        --out-dir pitts30k_small/images/val \
        --ratio 0.1 \
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def list_jpgs(dir_path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in exts
    )


def parse_utm(filename):
    """Trích utm_east, utm_north từ tên file dạng datasets_vg."""
    parts = filename.split("@")
    # parts[0] thường rỗng (path bắt đầu bằng '@')
    utm_east = float(parts[1])
    utm_north = float(parts[2])
    return utm_east, utm_north


def build_utm_array(files):
    coords = np.array([parse_utm(f) for f in files], dtype=np.float64)
    return coords


def shrink(db_files, q_files, ratio, pos_dist_thr, seed=0):
    rng = np.random.default_rng(seed)

    n_db, n_q = len(db_files), len(q_files)
    utm_db = build_utm_array(db_files)
    utm_q = build_utm_array(q_files)

    # 1) giữ ~ratio database, random uniform
    keep_db_idx = rng.choice(n_db, size=max(1, int(n_db * ratio)), replace=False)
    keep_db_idx.sort()
    utm_db_sub = utm_db[keep_db_idx]

    # 2) query nào còn >=1 positive trong db đã giữ mới hợp lệ
    nbrs = NearestNeighbors(radius=pos_dist_thr).fit(utm_db_sub)
    _, neighbors = nbrs.radius_neighbors(utm_q)
    valid_q_idx = np.array([i for i in range(n_q) if len(neighbors[i]) > 0])

    print(f"  Database: {n_db} -> giữ {len(keep_db_idx)} ({len(keep_db_idx)/n_db:.1%})")
    print(f"  Query hợp lệ sau khi giảm database: {len(valid_q_idx)}/{n_q}")

    # 3) trong query hợp lệ, random giữ tiếp để đạt ~ratio so với n_q gốc
    target_q = max(1, int(n_q * ratio))
    if len(valid_q_idx) > target_q:
        sel = rng.choice(valid_q_idx, size=target_q, replace=False)
        sel.sort()
    else:
        print(f"  [!] Chỉ có {len(valid_q_idx)} query hợp lệ, ít hơn target "
              f"{target_q}. Giữ hết {len(valid_q_idx)} query hợp lệ.")
        sel = valid_q_idx

    print(f"  Query: {n_q} -> giữ {len(sel)} ({len(sel)/n_q:.1%})")
    return keep_db_idx, sel


def copy_files(files, keep_idx, src_dir, dst_dir):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i in keep_idx:
        fname = files[i]
        shutil.copy2(Path(src_dir) / fname, dst_dir / fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--q-dir", required=True)
    ap.add_argument("--out-dir", required=True,
                     help="Thư mục output cho split này, sẽ tạo out-dir/database và out-dir/queries")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--pos-dist-thr", type=float, default=25.0,
                     help="Ngưỡng khoảng cách (m) để coi là positive, mặc định 25m (chuẩn Pitts30k)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Đang đọc danh sách file...")
    db_files = list_jpgs(args.db_dir)
    q_files = list_jpgs(args.q_dir)
    print(f"  database: {len(db_files)} ảnh | queries: {len(q_files)} ảnh")

    keep_db_idx, keep_q_idx = shrink(
        db_files, q_files, args.ratio, args.pos_dist_thr, seed=args.seed
    )

    out_db_dir = Path(args.out_dir) / "database"
    out_q_dir = Path(args.out_dir) / "queries"

    print("Đang copy database...")
    copy_files(db_files, keep_db_idx, args.db_dir, out_db_dir)
    print("Đang copy queries...")
    copy_files(q_files, keep_q_idx, args.q_dir, out_q_dir)

    print(f"Xong. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
