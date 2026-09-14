import pytest

pytest.importorskip("gradio")

from PIL import Image  # noqa: E402

import app  # noqa: E402


def test_output_component_serves_png():
    assert app.output_image.format == "png"


def test_api_info_generation():
    info = app.demo.get_api_info()
    assert info


def test_effective_scale_fits_max_side():
    left = Image.new("RGB", (1008, 756))
    right = Image.new("RGB", (1008, 756))
    scale = app.effective_scale(left, right, 512)
    assert round(left.width * scale) == 512
    assert round(left.height * scale) == 384


def test_effective_scale_for_high_resolution_input():
    left = Image.new("RGB", (6000, 4000))
    right = Image.new("RGB", (6000, 4000))
    scale = app.effective_scale(left, right, 512)
    assert round(6000 * scale) == 512
    assert round(4000 * scale) == 341


def test_effective_scale_never_upscales():
    left = Image.new("RGB", (320, 240))
    right = Image.new("RGB", (320, 240))
    assert app.effective_scale(left, right, 512) == 1.0


def test_describe_processing_shows_size():
    left = Image.new("RGB", (1008, 756))
    right = Image.new("RGB", (1008, 756))
    text = app.describe_processing(left, right, 512)
    assert "512×384" in text
    assert "scale 0.51" in text
    assert "Out-of-Memory" not in text


def test_describe_processing_warns_for_large_max_side():
    left = Image.new("RGB", (1008, 756))
    right = Image.new("RGB", (1008, 756))
    text = app.describe_processing(left, right, 1024)
    assert "Out-of-Memory" in text


def test_describe_processing_without_inputs():
    assert "入力してください" in app.describe_processing(None, None, 512)


def test_mark_stale_clears_result():
    image, message = app.mark_stale()
    assert image is None
    assert "再合成" in message
