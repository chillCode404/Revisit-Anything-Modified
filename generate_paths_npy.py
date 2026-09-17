"""
generate_paths_npy.py
-----------------------
Tạo lại file .npy danh sách đường dẫn ảnh (giống định dạng bạn đã có:
mảng string chứa full path, dtype '<U...'), dùng cho dataset Pitts30k
đã bị cắt giảm.

Cách dùng:
    python3 generate_paths_npy.py \
        --img-dir pitts30k_small/images/val/database \
        --out-npy pitts30k_small/images/val/database.npy
"""

import argparse
import os
from pathlib import Path

import numpy as np


def list_images(dir_path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in exts
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="Thư mục chứa ảnh (database hoặc queries)")
    ap.add_argument("--out-npy", required=True, help="Đường dẫn lưu file .npy output")
    ap.add_argument("--relative", action="store_true",
                     help="Lưu path dạng tương đối so với --img-dir thay vì full path tuyệt đối")
    args = ap.parse_args()

    files = list_images(args.img_dir)
    print(f"Tìm thấy {len(files)} ảnh trong {args.img_dir}")

    if args.relative:
        paths = files
    else:
        base = str(Path(args.img_dir).resolve())
        paths = [f"{base}/{f}" for f in files]

    arr = np.array(paths, dtype=str)
    Path(args.out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, arr)

    print(f"Đã lưu {len(arr)} path vào {args.out_npy}")
    print(f"Ví dụ 2 dòng đầu:\n{arr[:2]}")


if __name__ == "__main__":
    main()
