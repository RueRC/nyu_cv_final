### Environment Setup

#### using setup.py
```bash
micromamba create -n must3r python=3.11 cmake=3.14.0
micromamba activate must3r 
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126 # use the correct version of cuda for your system

# (recommended) if you can, install xFormers for memory-efficient attention
pip3 install -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126
pip3 install must3r@git+https://github.com/naver/must3r.git
# pip3 install must3r[optional]@git+https://github.com/naver/must3r.git # adds pillow-heif
# pip3 install --no-build-isolation must3r[curope]@git+https://github.com/naver/must3r.git # adds curope
# pip3 install --no-build-isolation must3r[all]@git+https://github.com/naver/must3r.git # adds all optional dependencies
```

#### development (no installation)

```bash
micromamba create -n must3r python=3.11 cmake=3.14.0
micromamba activate must3r 
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126 # use the correct version of cuda for your system

# (recommended) if you can, install xFormers for memory-efficient attention
pip3 install -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126

git clone --recursive https://github.com/naver/must3r.git
cd must3r
# if you have already cloned must3r:
# git submodule update --init --recursive

pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt
pip install -r requirements.txt

# install asmk
pip install faiss-cpu  # or the officially supported way (not tested): micromamba install -c pytorch faiss-cpu=1.11.0  # faiss-gpu=1.11.0 
mkdir build
cd build
git clone https://github.com/jenicek/asmk.git
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .
cd ../..

# Optional step: MUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dust3r/croco/models/curope/
pip install .
cd ../../../../
```

### Checkpoints
We provide several pre-trained models. For these checkpoints, make sure to agree to the license of all the training datasets we used, in addition to [MUSt3R License](LICENSE). For more information, check [NOTICE](NOTICE).

| [`MUSt3R_512.pth`](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |

### Evaluation

> [!NOTE]
> `slam.py` is installed as `must3r_slam` (or `must3r_slam.exe`) when must3r is installed to `site-packages`. 

```bash
# examples
# slam demo from a webcam (512 model)
python slam.py --chkpt /path/to/MUSt3R_512.pth --res 512 --subsamp 4 --gui --input cam:0 

# slam demo from a directory of images (224 model)
python slam.py \
	--chkpt "/path/to/MUSt3R_224_cvpr.pth" \
	--res 224 \
	--subsamp 2 \
  --keyframe_overlap_thr 0.05 \
  --min_conf_keyframe 1.5 \
  --overlap_percentile 85 \
	--input "/path_to/TUM_RGBD/rgbd_dataset_freiburg1_xyz/rgb" \  # can be a folder of video frames, or a webcam: cam:0
	--gui

# slam demo without a gui (it will write final memory state <memory.pkl> and camera trajectory <all_poses.npz>, optionally rerendered with --rerender)
python slam.py --chkpt /path/to/MUSt3R_512.pth --res 512 --subsamp 4 --input /path/to/video.mp4 --output /path/to/export
```

Hit the start toggle on the top right

```bibtex
@inproceedings{must3r_cvpr25,
      title={MUSt3R: Multi-view Network for Stereo 3D Reconstruction}, 
      author={Yohann Cabon and Lucas Stoffl and Leonid Antsfeld and Gabriela Csurka and Boris Chidlovskii and Jerome Revaud and Vincent Leroy},
      booktitle = {CVPR},
      year = {2025}
}

@misc{must3r_arxiv25,
      title={MUSt3R: Multi-view Network for Stereo 3D Reconstruction}, 
      author={Yohann Cabon and Lucas Stoffl and Leonid Antsfeld and Gabriela Csurka and Boris Chidlovskii and Jerome Revaud and Vincent Leroy},
      year={2025},
      eprint={2503.01661},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
