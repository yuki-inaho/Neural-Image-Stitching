"""Gradio demo for NIS (WACV 2024) neural image stitching.

Run with the optional gradio environment:

    pixi run -e gradio setup   # build pysrwarp + download the pretrained weights
    pixi run -e gradio app     # http://127.0.0.1:7860

The app loads ``pretrained/ihn.pth`` (homography estimator) and
``pretrained/NIS_blending.pth`` (enhanced & blended stitching) and stitches the
two input images. Layout is inspired by the kornia Image-Stitching space
(https://huggingface.co/spaces/kornia/Image-Stitching).
"""
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import stitch  # noqa: E402  (imported after cwd/sys.path setup)

OUTPUT_IMAGE = REPO_ROOT / "results.png"
DEFAULT_MAX_SIDE = 512
MIN_MAX_SIDE = 256
MAX_MAX_SIDE = 1608
LARGE_INPUT_HINT = 768  # above this, an 8 GiB GPU may run out of memory

_LOCK = threading.Lock()
_MODELS = None


def get_models():
    global _MODELS
    if _MODELS is None:
        with open(REPO_ROOT / "configs/test/NIS_blending.yaml") as f:
            import yaml

            config = yaml.safe_load(f)
        stitch.config = config
        _MODELS = stitch.prepare_validation(config)
    return _MODELS


def effective_scale(ref_img, tgt_img, max_side):
    """Single scale factor so that the longest side of both inputs fits max_side."""
    longest = max(ref_img.width, ref_img.height, tgt_img.width, tgt_img.height)
    return min(1.0, float(max_side) / float(longest))


def resize_input(image, scale):
    if scale == 1.0:
        return image
    size = (max(16, round(image.width * scale)), max(16, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def describe_processing(left, right, max_side):
    if left is None or right is None:
        return "**処理サイズ**: 左右の画像を入力してください。"
    scale = effective_scale(left, right, max_side)
    left_size = (max(16, round(left.width * scale)), max(16, round(left.height * scale)))
    right_size = (max(16, round(right.width * scale)), max(16, round(right.height * scale)))
    note = ""
    if int(max_side) > LARGE_INPUT_HINT:
        note = "\n\n⚠️ 最大辺が大きいため、8 GiB GPU では Out-of-Memory になる可能性があります。"
    return (
        f"**処理サイズ**: left {left_size[0]}×{left_size[1]} / "
        f"right {right_size[0]}×{right_size[1]} (scale {scale:.2f}){note}"
    )


def mark_stale():
    return None, "入力または処理サイズが変更されました。再度 **Stitch Images** で再合成してください。"


def stitch_images(left, right, max_side):
    """Stitch a left/right image pair and return (panorama, status)."""
    if left is None or right is None:
        raise gr.Error("left / right の両方の画像を入力してください。")

    if not torch.cuda.is_available():
        raise gr.Error("CUDA GPU が利用できません。")

    scale = effective_scale(left, right, max_side)
    ref_img = resize_input(left.convert("RGB"), scale)
    tgt_img = resize_input(right.convert("RGB"), scale)

    device = torch.cuda.get_device_name(0)
    started = time.time()
    try:
        with _LOCK:
            model, h_model = get_models()
            ref = transforms.ToTensor()(ref_img).cuda().unsqueeze(0)
            tgt = transforms.ToTensor()(tgt_img).cuda().unsqueeze(0)
            with torch.no_grad():
                stitch.valid(model, h_model, ref, tgt)
            stitched = Image.open(OUTPUT_IMAGE).convert("RGB")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise gr.Error(
                "CUDA out of memory: 最大辺を下げるか、より大きな VRAM の GPU を使用してください。"
            )
        raise
    finally:
        torch.cuda.empty_cache()

    elapsed = time.time() - started
    status = (
        f"**stitched**: {stitched.width}×{stitched.height}  |  "
        f"processed input: {ref_img.width}×{ref_img.height} "
        f"(scale {scale:.2f}, max side {int(max_side)}px)  |  "
        f"{elapsed:.1f} s on {device}"
    )
    return stitched, status


EXAMPLES = [[str(REPO_ROOT / "left.jpg"), str(REPO_ROOT / "right.jpg"), DEFAULT_MAX_SIDE]]

with gr.Blocks(title="NIS: Neural Image Stitching") as demo:
    gr.Markdown(
        "# NIS: Neural Image Stitching\n"
        "左右の画像を選び、**Stitch Images** を押すと合成します（結果は PNG で表示・保存できます）。"
    )

    input_left = gr.Image(label="Left image (ref)", type="pil", height=260, render=False)
    input_right = gr.Image(label="Right image (tgt)", type="pil", height=260, render=False)
    max_side = gr.Slider(
        MIN_MAX_SIDE,
        MAX_MAX_SIDE,
        value=DEFAULT_MAX_SIDE,
        step=64,
        label="処理の最大辺 (px)",
        info="画像の長辺をこのサイズに収まるよう自動縮小します",
        render=False,
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_left, input_right, max_side],
        label="サンプルで試す",
    )

    with gr.Row():
        input_left.render()
        input_right.render()

    size_info = gr.Markdown(describe_processing(None, None, DEFAULT_MAX_SIDE))

    max_side.render()
    stitch_button = gr.Button("Stitch Images", variant="primary")

    output_image = gr.Image(label="Stitched output (PNG)", type="pil", format="png")
    status_text = gr.Markdown()

    with gr.Accordion("モデル・実行環境について", open=False):
        gr.Markdown(
            "- WACV'24 *Implicit Neural Image Stitching With Enhanced and Blended Feature "
            "Reconstruction* ([paper](https://arxiv.org/abs/2309.01409))\n"
            "- `pretrained/ihn.pth` でホモグラフィを推定し、`pretrained/NIS_blending.pth` で合成\n"
            "- 8 GiB GPU では処理の最大辺 512px 程度を推奨（原寸相当は 11 GiB+ が必要）\n"
            "- 左右の入力サイズは自動縮小後に揃えられます"
        )

    stitch_button.click(
        fn=stitch_images,
        inputs=[input_left, input_right, max_side],
        outputs=[output_image, status_text],
    )
    for component in (input_left, input_right, max_side):
        component.change(
            fn=describe_processing,
            inputs=[input_left, input_right, max_side],
            outputs=[size_info],
        )
        component.change(fn=mark_stale, inputs=None, outputs=[output_image, status_text])

if __name__ == "__main__":
    launch_kwargs = {"show_error": True}
    if os.environ.get("GRADIO_SHARE") == "1":
        launch_kwargs["share"] = True
    if os.environ.get("SPACE_ID"):  # Hugging Face Spaces
        launch_kwargs.update(server_name="0.0.0.0", server_port=7860)
    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
