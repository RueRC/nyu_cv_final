#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export VGGT-Long outputs to COLMAP sparse model (bin), aligned to metacam2colmap.

Key compat points:
- Same coordinate corrections as metacam2colmap:
    1) R = R @ diag(1,-1,-1)
    2) T = GLOBAL_TRANS @ T
    3) T = Y_ROT_180 @ T
- Same binary layout for cameras.bin/images.bin/points3D.bin
- Single PINHOLE camera (id=1). Intrinsics read from INTR_TXT:
    * If multiple lines (per-frame), use the FIRST line as the shared camera
      (COLMAP minimal model with single camera; this matches "学长"最小导出).
    * If one line, use it directly.
- camera_poses.txt by VGGT-Long: each line = 16 floats of 4x4 C2W (row-major).
  We map them to images by sorted filename order of IMAGES_DIR.

Usage:
  python3 export_to_colmap.py \
    --images_dir <dir> \
    --pcd_ply <combined_pcd.ply> \
    --poses_txt <camera_poses.txt> \
    --intr_txt <intrinsic.txt> \
    --output_sparse_dir <.../sparse>

Or edit the defaults below.
"""

import os
import sys
import struct
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial.transform import Rotation as SciRot

# -------------------------
# Defaults (can be overridden by CLI)
# -------------------------
DEFAULT_IMAGES_DIR = "/media/huge/Huge/lab/processed_data/data3/images"
DEFAULT_PCD_PLY    = "/media/huge/Huge/lab/VGGT-Long/exps/_media_huge_Huge_lab_processed_data_data3_images/2025-10-27-17-56-52/pcd/combined_pcd.ply"
DEFAULT_POSES_TXT  = "/media/huge/Huge/lab/VGGT-Long/exps/_media_huge_Huge_lab_processed_data_data3_images/2025-10-27-17-56-52/camera_poses.txt"
DEFAULT_INTR_TXT   = "/media/huge/Huge/lab/VGGT-Long/exps/_media_huge_Huge_lab_processed_data_data3_images/2025-10-27-17-56-52/intrinsic.txt"
DEFAULT_OUTPUT_SPARSE_DIR = "/media/huge/Huge/lab/VGGT-Long/colmap-output/scene3/sparse"

# -------------------------
# COLMAP binary writers (aligned with metacam2colmap)
# -------------------------

def write_next_bytes(fid, data, fmt, endian="<"):
    if isinstance(data, (list, tuple, np.ndarray)):
        fid.write(struct.pack(endian + fmt, *data))
    else:
        fid.write(struct.pack(endian + fmt, data))

def write_cameras_binary(cameras: Dict[int, dict], path: Path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            # metacam2colmap uses model table where PINHOLE has model_id=1
            model_id = 1  # PINHOLE
            write_next_bytes(fid, [cam["id"], model_id, cam["width"], cam["height"]], "iiQQ")
            for p in cam["params"]:
                write_next_bytes(fid, float(p), "d")

def write_images_binary(images: Dict[int, dict], path: Path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img["id"], "i")
            write_next_bytes(fid, img["qvec"], "dddd")  # qw qx qy qz
            write_next_bytes(fid, img["tvec"], "ddd")
            write_next_bytes(fid, img["camera_id"], "i")
            for ch in img["name"]:
                write_next_bytes(fid, ch.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            # minimal model: no 2D points
            write_next_bytes(fid, 0, "Q")

def write_points3D_binary(points3D: Dict[int, dict], path: Path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt["id"], "Q")
            write_next_bytes(fid, pt["xyz"], "ddd")
            write_next_bytes(fid, pt["rgb"], "BBB")
            write_next_bytes(fid, pt["error"], "d")
            # empty track
            write_next_bytes(fid, 0, "Q")

# -------------------------
# Coordinate corrections (exactly like metacam2colmap)
# -------------------------

GLOBAL_ROT = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
], dtype=np.float64)

GLOBAL_TRANS = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

Y_ROT_180 = np.array([
    [ 1.0, 0.0,  0.0, 0.0],
    [ 0.0,-1.0,  0.0, 0.0],
    [ 0.0, 0.0, -1.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
], dtype=np.float64)

# def apply_coordinate_corrections(T_c2w: np.ndarray) -> np.ndarray:
#     # T = T_c2w.copy()
#     # T[:3, :3] = T[:3, :3] @ GLOBAL_ROT
#     # T = GLOBAL_TRANS @ T
#     # T = Y_ROT_180 @ T
#     # return T
#
#     # T = T_c2w.copy()
#     # T[:3,:3] = T[:3,:3] @ GLOBAL_ROT
#     # T = GLOBAL_TRANS @ T
#     # return T
def apply_coordinate_corrections(T_c2w):
    # 对 VGGT-Long：不要任何额外修正
    return T_c2w


# -------------------------
# IO helpers for VGGT-Long outputs
# -------------------------

def list_images(images_dir: Path) -> List[str]:
    exts = ("*.png","*.jpg","*.jpeg","*.JPG","*.PNG")
    paths = []
    for e in exts:
        paths.extend(images_dir.glob(e))
    paths = sorted(paths)
    return [p.name for p in paths]

def read_poses_vggt(path_txt: Path) -> List[np.ndarray]:
    """
    VGGT-Long's camera_poses.txt:
      Each line is 16 floats (row-major 4x4) of C2W. No image name.
    Return: list of 4x4 T_c2w, in file order.
    """
    poses = []
    with open(path_txt, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.replace(",", " ").split()
            try:
                vals = [float(x) for x in toks]
            except Exception:
                print(f"[WARN] Line {ln} not numeric, skip.")
                continue
            if len(vals) != 16:
                print(f"[WARN] Line {ln}: expect 16 floats, got {len(vals)}; skip.")
                continue
            T = np.array(vals, dtype=np.float64).reshape(4,4)
            poses.append(T)
    if not poses:
        raise ValueError(f"No valid 4x4 poses parsed from {path_txt}")
    return poses

def read_intrinsics_vggt(path_txt: Path) -> Tuple[float,float,float,float]:
    """
    VGGT-Long's intrinsic.txt:
      One line per image: 'fx fy cx cy'
      We use the FIRST line to define the shared COLMAP camera, matching
      the minimal 'single camera' export style.
    """
    with open(path_txt, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.replace(",", " ").split()
            vals = [float(x) for x in toks if _is_number(x)]
            if len(vals) >= 4:
                fx, fy, cx, cy = vals[:4]
                return float(fx), float(fy), float(cx), float(cy)
            else:
                print(f"[WARN] intrinsic line {ln}: need 4 numbers, got {len(vals)}; skip.")
                continue
    # fallback if file empty
    raise ValueError(f"Failed to read intrinsics from {path_txt}")

def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False

def get_image_size(images_dir: Path, sample_name: str) -> Tuple[int,int]:
    """
    Read size from file header if possible, else fall back to 1600x1600 to
    match metacam2colmap target.
    """
    import cv2
    p = str(images_dir / sample_name)
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Could not read image: {p}; fallback to 1600x1600")
        return 1600, 1600
    h, w = img.shape[:2]
    return int(w), int(h)

# -------------------------
# PLY loader (open3d -> plyfile)
# -------------------------

def load_ply_points_colors(ply_path: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    if not ply_path.exists():
        print(f"[INFO] PLY not found: {ply_path}")
        return None
    # Try open3d
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        cols = np.asarray(pcd.colors, dtype=np.float64) if pcd.colors else None
        if cols is not None and cols.size > 0:
            cols = np.clip((cols * 255.0).round().astype(np.uint8), 0, 255)
        else:
            cols = None
        return pts, cols
    except Exception as e:
        print(f"[INFO] open3d read failed ({e}); fallback to plyfile...")
    # Try plyfile
    try:
        from plyfile import PlyData
        with open(ply_path, 'rb') as f:
            ply = PlyData.read(f)
        v = ply['vertex']
        x = np.array(v['x'], dtype=np.float64)
        y = np.array(v['y'], dtype=np.float64)
        z = np.array(v['z'], dtype=np.float64)
        pts = np.stack([x,y,z], axis=1)
        cols = None
        has_rgb = all(k in v._property_lookup for k in ('red','green','blue'))
        if has_rgb:
            r = np.array(v['red'])
            g = np.array(v['green'])
            b = np.array(v['blue'])
            max_val = float(max(r.max(initial=0), g.max(initial=0), b.max(initial=0)))
            if max_val > 255.0 and max_val > 0:
                scale = 255.0 / max_val
                r = (r * scale).astype(np.uint8)
                g = (g * scale).astype(np.uint8)
                b = (b * scale).astype(np.uint8)
            else:
                r = r.astype(np.uint8)
                g = g.astype(np.uint8)
                b = b.astype(np.uint8)
            cols = np.stack([r,g,b], axis=1)
        return pts, cols
    except Exception as e:
        print(f"[WARN] plyfile read failed: {e}")
        return None

# -------------------------
# Math helpers
# -------------------------

def rotmat_to_qwxyz(R: np.ndarray) -> Tuple[float,float,float,float]:
    q_xyzw = SciRot.from_matrix(R).as_quat()
    return float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Export VGGT-Long outputs to COLMAP (metacam2colmap-aligned)")
    ap.add_argument("--images_dir", default=DEFAULT_IMAGES_DIR, type=str)
    ap.add_argument("--pcd_ply", default=DEFAULT_PCD_PLY, type=str)
    ap.add_argument("--poses_txt", default=DEFAULT_POSES_TXT, type=str)
    ap.add_argument("--intr_txt", default=DEFAULT_INTR_TXT, type=str)
    ap.add_argument("--output_sparse_dir", default=DEFAULT_OUTPUT_SPARSE_DIR, type=str)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    ply_path   = Path(args.pcd_ply)
    poses_txt  = Path(args.poses_txt)
    intr_txt   = Path(args.intr_txt)
    out_root   = Path(args.output_sparse_dir) / "0"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Files & lists
    image_names = list_images(images_dir)
    if not image_names:
        print(f"[ERROR] No images in {images_dir}")
        sys.exit(1)

    poses_list = read_poses_vggt(poses_txt)  # list of 4x4 c2w
    if len(poses_list) != len(image_names):
        n = min(len(poses_list), len(image_names))
        print(f"[WARN] #poses ({len(poses_list)}) != #images ({len(image_names)}); "
              f"using first {n} pairs.")
        image_names = image_names[:n]
        poses_list  = poses_list[:n]

    fx, fy, cx, cy = read_intrinsics_vggt(intr_txt)

    # Image size for camera width/height
    sample_w, sample_h = get_image_size(images_dir, image_names[0])

    # 2) Camera dict (single camera id=1; PINHOLE; params [fx, fy, cx, cy])
    cameras = {
        1: {
            "id": 1,
            "model": "PINHOLE",
            "width": int(sample_w),
            "height": int(sample_h),
            "params": np.array([fx, fy, cx, cy], dtype=np.float64)
        }
    }

    # 3) Images dict (apply metacam2colmap corrections, then invert to w2c)
    images = {}
    img_id = 1
    for name, T_c2w in zip(image_names, poses_list):
        T_corr = apply_coordinate_corrections(T_c2w)
        T_w2c = np.linalg.inv(T_corr)
        R = T_w2c[:3, :3]
        t = T_w2c[:3, 3]
        q_wxyz = rotmat_to_qwxyz(R)
        images[img_id] = {
            "id": img_id,
            "qvec": q_wxyz,
            "tvec": (float(t[0]), float(t[1]), float(t[2])),
            "camera_id": 1,
            "name": name
        }
        img_id += 1

    # 4) points3D from PLY (optional) — apply the SAME corrections
    points3D = {}
    pts_cols = load_ply_points_colors(ply_path)
    if pts_cols is None:
        print("[INFO] No points exported (points3D.bin will be empty).")
    else:
        pts, cols = pts_cols
        N = pts.shape[0]
        homog = np.hstack([pts, np.ones((N,1), dtype=np.float64)])
        # (1) rot Y/Z flip on points
        homog[:, :3] = homog[:, :3] @ GLOBAL_ROT
        # (2) axis reorder
        homog = (GLOBAL_TRANS @ homog.T).T
        # (3) extra 180° rotation (same as metacam2colmap)
        homog = (Y_ROT_180 @ homog.T).T
        pts_corr = homog[:, :3]

        if cols is None:
            rgb = np.full((N,3), 255, dtype=np.uint8)
        else:
            rgb = np.clip(cols, 0, 255).astype(np.uint8)

        for i in range(N):
            points3D[i+1] = {
                "id": i+1,
                "xyz": (float(pts_corr[i,0]), float(pts_corr[i,1]), float(pts_corr[i,2])),
                "rgb": (int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2])),
                "error": 0.0
            }
        print(f"[INFO] points3D: exported {N} points")

    # 5) Write COLMAP bin files
    write_cameras_binary(cameras, out_root / "cameras.bin")
    write_images_binary(images, out_root / "images.bin")
    write_points3D_binary(points3D, out_root / "points3D.bin")

    print(f"[OK] Wrote COLMAP sparse model to: {out_root}")
    print(f"     cameras.bin: 1 camera; images.bin: {len(images)} images; points3D.bin: {len(points3D)} points")

if __name__ == "__main__":
    main()