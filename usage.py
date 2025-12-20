import os
from glob import glob
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# === 你的保存函数直接复用 ===
from usage import save_colmap_model_with_points  # 如果定义在同文件里就不用导入

# === 主要参数 ===
ROOT_DIR = "/local_data/xl3136/DATA/wanderland_eval"
OUTPUT_ROOT = "/local_data/yz10442/dust3r/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

device = torch.device("cuda")
batch_size = 1
schedule = "cosine"
lr = 0.01
niter = 300
model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_224_dpt"
model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

# === 遍历每个子文件夹 ===
all_folders = sorted(os.listdir(ROOT_DIR))
print(f"发现 {len(all_folders)} 个文件夹，开始批量处理...")

start_idx = 177  # 从第三个文件夹开始（Python 从 0 开始计数）

for idx, folder in enumerate(all_folders[start_idx:], start=start_idx):
    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n🚀 [{idx+1}/{len(all_folders)}] 处理: {folder}")
    image_dir = os.path.join(folder_path, "images")  # 假设图片直接在子文件夹里
    all_images = sorted(glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg")))
    image_paths = all_images[:43]
    print(f"Loading {len(image_paths)} images (first 50 of total {len(all_images)})")
    if len(all_images) < 2:
        print(f"⚠️ 文件夹 {folder} 图片数不足 2，跳过。")
        continue

    # === 载入图片 ===
    images = load_images(image_paths, size=512)
    for im_dict in images:
        im_dict["img"] = im_dict["img"].to(device)
    pairs = make_pairs(images, scene_graph='swin-30', prefilter=None, symmetrize=False)
    output = inference(pairs, model, device, batch_size=batch_size)

    # === 全局配准 ===
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    _ = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # === 导出结果 ===
    out_dir = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)
    poses = scene.get_im_poses()
    focals = scene.get_focals()
    imgs = scene.imgs
    pts3d = scene.get_pts3d()

    np.save(os.path.join(out_dir, "poses2.npy"), poses.detach().cpu().numpy())
    save_colmap_model_with_points(
        out_dir=out_dir,
        poses_npy_path=os.path.join(out_dir, "poses2.npy"),
        image_paths=image_paths,
        pts3d=pts3d,
        focals=focals.detach().cpu().numpy(),
        wh=(512, 384),
        camera_model="PINHOLE",
        fix_axes=False,
        color_from_imgs=imgs
    )

    print(f"✅ 完成 {folder}，结果保存至 {out_dir}")

    # === 清理显存 ===
    del images, pairs, output, scene, poses, focals, imgs, pts3d
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("\n🎉 所有文件夹处理完成！")
