## Implicit Neural Image Stitching With Enhanced and Blended Feature Reconstruction

This repository contains the official implementation of NIS, 24' WACV: https://arxiv.org/abs/2309.01409

## Requirement
1) Python packages
```
conda env create --file environment.yaml
conda activate nis
```
2) [pysrwarp](https://github.com/sanghyun-son/srwarp)
: Follow the guidelines in the repository for more details and compile debugging.
```
git clone https://github.com/sanghyun-son/pysrwarp
cd pysrwarp
make
```

### Pixi environment (reproducible alternative)
`pixi.toml` pins Python 3.9, PyTorch 1.10.2+cu113 and the inference dependencies.
```
pixi install        # create .pixi/envs/default
pixi run setup      # build pysrwarp + download the pretrained weights
pixi run test-e2e   # end-to-end pytest on left.jpg / right.jpg
```
- `pysrwarp` is cloned to `.cache/pysrwarp` at a pinned commit and built with the host CUDA toolkit (`CUDA_HOME` / `NVCC`, `PYSRWARP_CUDA_ARCH` to override the GPU arch).
- `scripts/download_pretrained.sh` fetches the official Google Drive archive with gdown, verifies SHA-256 and extracts it into `pretrained/`.
- `pysrwarp` defines a `NUM_THREADS` macro that collides with CUB template parameters on CUDA >= 12; the setup script renames it.
- The e2e test resizes the samples to fit an 8 GiB GPU. Use `pixi run test-e2e-full` for the original 1008x756 images (needs more VRAM).
- `models/ihn/network.py` uses `kornia.geometry.transform.get_perspective_transform` instead of the unmaintained `torchgeometry` (same DLT algorithm, already used in `models/ihn/utils.py`).

### Gradio demo (optional)
`gradio` lives in a separate pixi environment (`-e gradio`) so the default environment stays lightweight.
```
pixi run -e gradio start   # first time: build pysrwarp + download weights + launch (http://127.0.0.1:7860)
pixi run -e gradio app     # launch only (after the first `start`)
pixi run -e gradio test-app
```
- Starting the app from another environment (`pixi run app`) prints a hint to use `pixi run -e gradio start`.
- Both inputs are automatically resized so their longest side fits the "処理の最大辺" slider (default 512 px, 8 GiB GPU friendly); the processed size is shown before stitching.
- Results are served and downloaded as PNG (`gr.Image(format="png")`).
- `pydantic` is pinned `<2.10` in the gradio feature: newer versions emit boolean `additionalProperties` JSON schemas that `gradio-client` 1.3 (bundled with gradio 4.44) cannot parse.

## Dataset
- [UDIS-D](https://github.com/nie-lang/UnsupervisedDeepImageStitching)
- [MS-COCO 2017v](https://cocodataset.org/#download)

## Pretrained Models
You can download below models on this [link](https://drive.google.com/file/d/1sdfquwxhKLq2aBGGdtiu8_SM-g-aDUtM/view?usp=share_link).
1. NIS_enhancing.pth: Pretrained on Enhanced Stitching (Stage 1),
2. NIS_blending.pth: Pretrained on Enhanced & Blended Stitching (Stage 1 & 2),
3. ihn.pth: Our reproduced Homography Estimator used in the second stage training.

## Train & Evaluation
```
bash scripts/train.sh 0
bash scripts/eval.sh 0
```

## Stitching Example
Note that stitching large-sized images may cause the GPU out-of-memory due to the consumption of the backbone.
```
bash scripts/stitch.sh left.jpg right.jpg
```

## Acknowlegment
This work is mainly based on [LTEW](https://github.com/jaewon-lee-b/ltew) and [IHN](https://github.com/imdumpl78/IHN), we thank the authors for the contribution.
