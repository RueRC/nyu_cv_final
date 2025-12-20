import os
import numpy as np
from scipy.spatial.transform import Rotation as R

base_output = "/scratch/yz10442/must3r/output"
base_images = "/wanderland_eval"

# 遍历所有 output 子目录
for dataset in sorted(os.listdir(base_output)):
    npz_path = os.path.join(base_output, dataset, "all_poses.npz")
    image_dir = os.path.join(base_images, dataset, "images")
    output_dir = os.path.join(base_output, dataset)

    # 检查文件存在性
    if not os.path.exists(npz_path):
        print(f"⏭️ Skipping {dataset}: no all_poses.npz")
        continue
    if not os.path.isdir(image_dir):
        print(f"⚠️ Skipping {dataset}: image folder not found ({image_dir})")
        continue

    print(f"\n📂 Processing dataset: {dataset}")

    # ==== 加载 npz ====
    data = np.load(npz_path, allow_pickle=True)
    poses = data["poses"].astype(np.float64)
    focal_obj = data["focal"].item()

    if isinstance(focal_obj, dict):
        focal = list(focal_obj.values())[0]
    else:
        focal = float(focal_obj)
    focal = float(focal)

    imgHWs = data["imgHWs"]
    height, width = np.mean(imgHWs, axis=0).astype(int)

    # ==== 读取真实文件名 ====
    img_names = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"  Found {len(img_names)} images, {len(poses)} poses")

    if len(img_names) != len(poses):
        print(f"  ⚠️ Mismatch: {len(img_names)} images but {len(poses)} poses!")

    # ==== 写 cameras.txt ====
    cam_txt = os.path.join(output_dir, "cameras.txt")
    with open(cam_txt, "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fx = fy = focal
        cx, cy = width / 2, height / 2
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    # ==== 写 images.txt ====
    img_txt = os.path.join(output_dir, "images.txt")
    with open(img_txt, "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i, pose in enumerate(poses):
            R_c2w = pose[:3, :3]
            t_c2w = pose[:3, 3]

            R_w2c = R_c2w.T
            t_w2c = -R_c2w.T @ t_c2w

            q = R.from_matrix(R_w2c).as_quat()
            qx, qy, qz, qw = q

            img_name = img_names[i] if i < len(img_names) else f"{i:06d}.png"
            f.write(f"{i+1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{t_w2c[0]:.8f} {t_w2c[1]:.8f} {t_w2c[2]:.8f} 1 {img_name}\n\n")

    print(f"✅ Saved COLMAP files → {output_dir}/(cameras.txt, images.txt)")
