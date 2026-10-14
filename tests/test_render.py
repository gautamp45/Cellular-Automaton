import numpy as np
import pytest
from PIL import Image

from cellauto import render


def test_binary_to_image_size_and_colours() -> None:
    g = np.array([[0, 1], [1, 0]])
    img = render.binary_to_image(g, scale=3)
    assert img.size == (6, 6) and img.mode == "RGB"
    off, on = render.PALETTES["dark"]
    assert img.getpixel((0, 0)) == off
    assert img.getpixel((3, 0)) == on
    with pytest.raises(ValueError):
        render.binary_to_image(g, palette="nope")
    with pytest.raises(ValueError):
        render.binary_to_image(np.zeros(3))


def test_scalar_luts() -> None:
    a = np.array([[0.0, 0.5, 1.0, 2.0]])
    img = render.scalar_to_image(a, scale=1)
    assert img.getpixel((0, 0)) == (0, 0, 4)  # magma(0)
    assert img.getpixel((2, 0)) == (252, 253, 191)  # magma(1)
    assert img.getpixel((3, 0)) == img.getpixel((2, 0))  # clipped
    gray = render.scalar_to_image(a, scale=1, cmap="gray")
    assert gray.getpixel((1, 0)) == (128, 128, 128)
    with pytest.raises(ValueError):
        render.scalar_to_image(a, cmap="viridis")


def test_rgb_layouts_agree() -> None:
    chw = np.random.default_rng(0).random((3, 5, 7))
    a = render.rgb_to_image(chw, scale=2)
    b = render.rgb_to_image(chw.transpose(1, 2, 0), scale=2)
    assert a.size == (14, 10) and np.array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError):
        render.rgb_to_image(np.zeros((4, 4, 4)))


def test_save_png_and_gif(tmp_path) -> None:
    frames = [render.binary_to_image(np.eye(4) * (i % 2), 2) for i in range(5)]
    render.save_png(frames[1], tmp_path / "sub" / "a.png")
    assert Image.open(tmp_path / "sub" / "a.png").size == (8, 8)
    render.save_gif(frames, tmp_path / "a.gif", fps=10)
    with Image.open(tmp_path / "a.gif") as gif:
        assert gif.n_frames == 5
        assert gif.info["duration"] == 100
    with pytest.raises(ValueError):
        render.save_gif([], tmp_path / "b.gif")
