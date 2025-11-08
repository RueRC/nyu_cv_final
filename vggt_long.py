# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-


import os
import sys
import gc
import glob
import argparse
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
from PIL import Image

# ========= 你项目里的依赖 =========
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from loop_utils.config_utils import load_config
    from loop_utils.sim3utils import (
        weighted_align_point_maps,
        accumulate_sim3_transforms,
        compute_sim3_ab,
        process_loop_list,
    )
    from loop_utils.sim3loop import Sim3LoopOptimizer
    from LoopModels.LoopModel import LoopDetector
    from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
except Exception as e:
    print("[ERROR] Import failed:", e)
    raise

# 可选：onnxruntime（本脚本不使用）
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass


def remove_duplicates(data_list):
    """
    data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    去重：只保留 (i, j) 成对的第一次出现，且排除 i == j 的情况
    """
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen:
            seen[key] = True
            result.append(item)
    return result


class VGGT_Long:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            cc_major = torch.cuda.get_device_capability()[0]
        except Exception:
            cc_major = 0
        self.dtype = torch.bfloat16 if cc_major >= 8 else torch.float16

        self.useDBoW = self.config["Model"]["useDBoW"]
        self.loop_enable = self.config["Model"]["loop_enable"]

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        # === 恢复与原版一致的临时目录（仅流程一致；最后会清理） ===
        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir   = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir      = os.path.join(save_dir, "_tmp_results_loop")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)

        # 不导出点云
        self.pcd_dir = None

        self.all_camera_poses = []       # [(chunk_range, extrinsics)]
        self.all_camera_intrinsics = []  # [(chunk_range, intrinsics)]

        # 处理完自动删除临时文件
        self.delete_temp_files = True

        print("Loading model...")
        self.model = VGGT()
        weights_path = self.config["Weights"]["VGGT"]
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.device)
        print("Model loaded.")

        # Loop closure 组件（可选）
        self.retrieval = None
        self.loop_detector = None
        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                # 不在结果目录写任何 loop 文件
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output="/dev/null",
                    config=self.config,
                )

        self.chunk_indices = None
        self.loop_list = []          # 帧级或块级 loop 候选
        self.sim3_list = []          # 串行相邻块间 Sim(3)
        self.loop_sim3_list = []     # 闭环约束
        self.loop_predict_list = []  # 用于闭环的拼块预测（存内存 & .npy）

        print("Init done.")

    def get_loop_pairs(self):
        if self.useDBoW:
            # DBoW2：逐帧检索候选
            for frame_id, img_path in tqdm(enumerate(self.img_list), total=len(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if image_ori.ndim == 2:
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)
                frame = cv2.resize(image_ori, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(
                    thresh=self.config["Loop"]["DBoW"]["thresh"],
                    num_repeat=self.config["Loop"]["DBoW"]["num_repeat"],
                )
                if cands is not None:
                    (i, j) = cands
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)
                # 内部如有持久化，不写入 save_dir
                self.retrieval.save_up_to(frame_id)
        else:
            # DNIO v2
            self.loop_detector.run()  # output="/dev/null"
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start2, end2 = range_2
            chunk_image_paths += self.img_list[start2:end2]

        images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        assert images.ndim == 4 and images.shape[1] == 3, "images must be [B, 3, H, W]"

        # 推理
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                if self.config["Model"].get("reference_frame_mid", False):
                    mid_idx = len(images) // 2
                    images = torch.cat(
                        [images[mid_idx:mid_idx + 1], images[:mid_idx], images[mid_idx + 1:]],
                        dim=0,
                    )
                    predictions = self.model(images)

                    def _reorder(x):
                        return torch.cat([x[:, 1:mid_idx + 1], x[:, :1], x[:, mid_idx + 1:]], dim=1)

                    for k in ["depth", "depth_conf", "world_points", "world_points_conf", "pose_enc", "images"]:
                        predictions[k] = _reorder(predictions[k])
                else:
                    predictions = self.model(images)

        torch.cuda.empty_cache()

        # 编码 → 外参/内参
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # 转 numpy
        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].detach().cpu().numpy().squeeze(0)
        predictions["depth"] = np.squeeze(predictions["depth"])

        # === 与原版一致：保存为 .npy，供后续对齐/闭环 ===
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, predictions)

        # 记录相机参数（用于最终写 txt）
        if not is_loop and range_2 is None:
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, predictions["extrinsic"]))
            self.all_camera_intrinsics.append((chunk_range, predictions["intrinsic"]))

        # loop 或双段拼块：返回 predictions
        return predictions if (is_loop or range_2 is not None) else None

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] overlap({self.overlap}) must be < chunk_size({self.chunk_size})")

        n = len(self.img_list)
        if n <= self.chunk_size:
            self.chunk_indices = [(0, n)]
        else:
            step = self.chunk_size - self.overlap
            self.chunk_indices = []
            i = 0
            while True:
                s = i * step
                e = min(s + self.chunk_size, n)
                self.chunk_indices.append((s, e))
                if e == n:
                    break
                i += 1

        # 闭环候选（帧级→块级）
        if self.loop_enable:
            print("Loop SIM(3) estimating...")
            loop_results = process_loop_list(
                self.chunk_indices, self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2)
            )
            loop_results = remove_duplicates(loop_results)
            for item in loop_results:
                # item: (chunk_i, (a0,a1), chunk_j, (b0,b1))
                preds = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)
                self.loop_predict_list.append((item, preds))

            # 释放 loop 资源
            if self.useDBoW and self.retrieval is not None:
                try:
                    self.retrieval.close()
                except Exception:
                    pass
                self.retrieval = None
            if (not self.useDBoW) and self.loop_detector is not None:
                try:
                    del self.loop_detector
                except Exception:
                    pass
                self.loop_detector = None
            gc.collect()
            torch.cuda.empty_cache()

        # 主序：逐块推理
        print(f"Processing {len(self.img_list)} images in {len(self.chunk_indices)} chunks: "
              f"chunk_size={self.chunk_size}, overlap={self.overlap}")
        for ci, rng in enumerate(self.chunk_indices):
            print(f"[Chunk] {ci+1}/{len(self.chunk_indices)} range={rng}")
            self.process_single_chunk(rng, chunk_idx=ci)
            torch.cuda.empty_cache()

        # 对齐不再需要网络，释放显存
        del self.model
        torch.cuda.empty_cache()

        # === 相邻块对齐：严格按原版，从磁盘 .npy 读取 ===
        print("Aligning all the chunks...")
        self.sim3_list = []
        for i in range(len(self.chunk_indices) - 1):
            print(f"Aligning {i} and {i+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{i}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{i+1}.npy"), allow_pickle=True).item()

            point_map1 = chunk_data1["world_points"][-self.overlap:]
            point_map2 = chunk_data2["world_points"][:self.overlap]
            conf1 = chunk_data1["world_points_conf"][-self.overlap:]
            conf2 = chunk_data2["world_points_conf"][:self.overlap]

            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            s, R, t = weighted_align_point_maps(
                point_map1, conf1, point_map2, conf2,
                conf_threshold=conf_threshold, config=self.config
            )
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)
            self.sim3_list.append((s, R, t))

        # === 闭环约束：与原版完全一致的实现 ===
        if self.loop_enable and len(self.loop_predict_list) > 0:
            self.loop_sim3_list = []
            for item, preds in self.loop_predict_list:
                # item: (chunk_idx_a, (a0,a1), chunk_idx_b, (b0,b1))
                chunk_idx_a, chunk_a_range, chunk_idx_b, chunk_b_range = item

                # A 段：loop 拼块的前半段 ↔ chunk_a 对应片段
                len_a = chunk_a_range[1] - chunk_a_range[0]
                point_map_loop_a = preds["world_points"][:len_a]
                conf_loop_a      = preds["world_points_conf"][:len_a]

                chunk_data_a = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                    allow_pickle=True
                ).item()
                a_rel_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                a_rel_end   = a_rel_begin + len_a
                point_map_a = chunk_data_a["world_points"][a_rel_begin:a_rel_end]
                conf_a      = chunk_data_a["world_points_conf"][a_rel_begin:a_rel_end]

                conf_thr_a = min(np.median(conf_a), np.median(conf_loop_a)) * 0.1
                s_a, R_a, t_a = weighted_align_point_maps(
                    point_map_a, conf_a, point_map_loop_a, conf_loop_a,
                    conf_threshold=conf_thr_a, config=self.config
                )

                # B 段：loop 拼块的后半段 ↔ chunk_b 对应片段
                len_b = chunk_b_range[1] - chunk_b_range[0]
                point_map_loop_b = preds["world_points"][-len_b:]
                conf_loop_b      = preds["world_points_conf"][-len_b:]

                chunk_data_b = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                    allow_pickle=True
                ).item()
                b_rel_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                b_rel_end   = b_rel_begin + len_b
                point_map_b = chunk_data_b["world_points"][b_rel_begin:b_rel_end]
                conf_b      = chunk_data_b["world_points_conf"][b_rel_begin:b_rel_end]

                conf_thr_b = min(np.median(conf_b), np.median(conf_loop_b)) * 0.1
                s_b, R_b, t_b = weighted_align_point_maps(
                    point_map_b, conf_b, point_map_loop_b, conf_loop_b,
                    conf_threshold=conf_thr_b, config=self.config
                )

                # 计算 a -> b 的 Sim(3)
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

            # 与原版一致：优化器流程（不保存可视化）
            loop_opt = Sim3LoopOptimizer(self.config)
            _ = loop_opt.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = loop_opt.optimize(self.sim3_list, self.loop_sim3_list)
            _ = loop_opt.sequential_to_absolute_poses(self.sim3_list)

        # 串联到第一块坐标系
        print("Apply alignment to global (accumulate sim3)...")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)

        # 仅写出最终相机位姿与内参
        self.save_camera_poses()
        print("Done.")

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()
            # 释放 loop 资源
            if self.useDBoW and self.retrieval is not None:
                try:
                    self.retrieval.close()
                except Exception:
                    pass
                self.retrieval = None
            if (not self.useDBoW) and self.loop_detector is not None:
                try:
                    del self.loop_detector
                except Exception:
                    pass
                self.loop_detector = None
            gc.collect()
            torch.cuda.empty_cache()

        self.process_long_sequence()

    def save_camera_poses(self):
        """
        仅保存两类文件：
          - camera_poses.txt：每行 4x4 C2W（按行拍平 16 个数）
          - intrinsic.txt：每行 fx fy cx cy
        不再输出任何 ply/png/npy/点云。
        """
        print("Saving camera poses & intrinsics ...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        # 第一块：直接放
        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i]
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]

        # 后续块：左乘累计后的 Sim(3)
        for k in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[k]
            _, chunk_intrinsics = self.all_camera_intrinsics[k]
            s, R, t = self.sim3_list[k - 1]  # 已经是相对第一块

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i]
                c2w = np.linalg.inv(w2c)
                c2w_global = S @ c2w  # 左乘
                all_poses[idx] = c2w_global
                all_intrinsics[idx] = chunk_intrinsics[i]

        # 写文件
        os.makedirs(self.output_dir, exist_ok=True)
        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        intri_path = os.path.join(self.output_dir, "intrinsic.txt")

        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat = pose.flatten()
                f.write(" ".join(map(str, flat)) + "\n")
        print(f"[SAVE] {poses_path}")

        with open(intri_path, "w") as f:
            for K in all_intrinsics:
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                f.write(f"{fx} {fy} {cx} {cy}\n")
        print(f"[SAVE] {intri_path}")

    def close(self):
        """
        清理 _tmp_results_unaligned / _tmp_results_aligned / _tmp_results_loop
        仅删除临时 .npy，保留 camera_poses.txt / intrinsic.txt
        """
        if not getattr(self, "delete_temp_files", False):
            return
        total_space = 0
        for d in [self.result_unaligned_dir, self.result_aligned_dir, self.result_loop_dir]:
            if not d or (not os.path.isdir(d)):
                continue
            print(f"[CLEAN] Deleting temp files under {d}")
            for fn in os.listdir(d):
                fp = os.path.join(d, fn)
                if os.path.isfile(fp):
                    try:
                        total_space += os.path.getsize(fp)
                        os.remove(fp)
                    except Exception:
                        pass
            try:
                os.rmdir(d)
            except Exception:
                pass
        print(f"[CLEAN] Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


def main():
    parser = argparse.ArgumentParser(description="VGGT-Long (only poses & intrinsics)")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images folder")
    parser.add_argument("--config", type=str, default="./configs/base_config.yaml", help="Config yaml")
    parser.add_argument("--overwrite", action="store_true", help="Kept for CLI compatibility (ignored here)")
    parser.add_argument('--exp_root', type=str, default='/media/huge/Game/test/exp',
                        help='Root directory to store exp outputs')
    args = parser.parse_args()

    # # 固定输出根：/media/huge/Game/test/exp/<scene_name>
    # EXP_ROOT = "/media/huge/Game/test/exp"
    # scene_name = os.path.basename(os.path.dirname(args.image_dir.rstrip("/")))
    # save_dir = os.path.join(EXP_ROOT, scene_name)
    # os.makedirs(save_dir, exist_ok=True)

    exp_root = args.exp_root.rstrip("/")
    scene_name = os.path.basename(os.path.dirname(args.image_dir.rstrip("/")))
    save_dir = os.path.join(exp_root, scene_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving Stage-1 outputs to: {save_dir}")

    config = load_config(args.config)

    vggt_long = VGGT_Long(args.image_dir, save_dir, config)
    vggt_long.run()
    vggt_long.close()  # 清理临时 .npy

    # 不做合并点云/PLY等任何额外导出
    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()
    print("VGGT Long done.")


if __name__ == "__main__":
    main()
