import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = REPO_ROOT / "pretrained"
SAMPLE_REF = REPO_ROOT / "left.jpg"
SAMPLE_TGT = REPO_ROOT / "right.jpg"
RESULT_IMAGE = REPO_ROOT / "results.png"
CONFIG = REPO_ROOT / "configs" / "test" / "NIS_blending.yaml"

REQUIRED_WEIGHTS = ("NIS_blending.pth", "ihn.pth")

# Full-resolution stitching needs more VRAM than an 8 GiB GPU has. The samples
# are resized by default; set NIS_E2E_SCALE=1 to run at the original 1008x756.
E2E_SCALE = float(os.environ.get("NIS_E2E_SCALE", "0.5"))


@pytest.fixture(scope="module")
def require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available")
    return torch


@pytest.fixture(scope="module")
def require_assets():
    missing = [
        name for name in REQUIRED_WEIGHTS if not (PRETRAINED_DIR / name).is_file()
    ]
    if missing:
        pytest.skip(
            "missing pretrained weights: "
            + ", ".join(missing)
            + " (run `pixi run download-pretrained`)"
        )
    for path in (SAMPLE_REF, SAMPLE_TGT):
        assert path.is_file(), f"missing sample image: {path}"


def _make_inputs(tmp_path):
    if E2E_SCALE == 1.0:
        return SAMPLE_REF, SAMPLE_TGT

    paths = []
    for src in (SAMPLE_REF, SAMPLE_TGT):
        image = Image.open(src).convert("RGB")
        size = (round(image.width * E2E_SCALE), round(image.height * E2E_SCALE))
        path = tmp_path / src.name
        image.resize(size, Image.LANCZOS).save(path, quality=95)
        paths.append(path)
    return paths[0], paths[1]


@pytest.mark.e2e
def test_stitch_e2e_with_sample_images(require_cuda, require_assets, tmp_path):
    """Run the full NIS blending pipeline end-to-end on the bundled sample pair."""
    for path in (CONFIG,):
        assert path.is_file(), f"missing input: {path}"

    ref_path, tgt_path = _make_inputs(tmp_path)
    ref_w, ref_h = Image.open(ref_path).size

    RESULT_IMAGE.unlink(missing_ok=True)
    started_at = time.time()

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128"
    )

    proc = subprocess.run(
        [
            sys.executable,
            "stitch.py",
            f"--config={CONFIG}",
            "--ref",
            str(ref_path),
            "--tgt",
            str(tgt_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"stitch.py failed:\n{output}"

    assert RESULT_IMAGE.is_file(), f"results.png was not created:\n{output}"
    assert RESULT_IMAGE.stat().st_mtime >= started_at

    image = Image.open(RESULT_IMAGE).convert("RGB")
    width, height = image.size
    assert width >= ref_w * 0.98, output
    assert height >= ref_h * 0.98, output

    pixels = np.asarray(image).astype(np.float32)
    assert pixels.std() > 10.0, "stitched output looks flat"
    assert pixels.mean() > 10.0, "stitched output is almost black"
    assert pixels.mean() < 250.0, "stitched output is almost white"
    lit = (pixels.max(axis=-1) > 16).mean()
    assert lit > 0.5, "more than half of the stitched output is black"
