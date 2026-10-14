import numpy as np
import pytest
from PIL import Image

from cellauto.cli import main


def test_elementary_png_size(tmp_path) -> None:
    out = tmp_path / "r30.png"
    assert main(["elementary", "--rule", "30", "--width", "51", "--steps", "20", "--scale", "3",
                 "--out", str(out)]) == 0  # fmt: skip
    assert Image.open(out).size == (51 * 3, 21 * 3)


def test_elementary_random_is_seeded(tmp_path) -> None:
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for out in (a, b):
        args = ["elementary", "--rule", "110", "--init", "random", "--seed", "4", "--out", str(out)]
        assert main(args) == 0
    assert a.read_bytes() == b.read_bytes()


def test_life_gif_frames(tmp_path) -> None:
    out = tmp_path / "life.gif"
    args = ["life", "--pattern", "glider", "--size", "20x20", "--steps", "12", "--scale", "2",
            "--boundary", "wrap", "--out", str(out)]  # fmt: skip
    assert main(args) == 0
    with Image.open(out) as gif:
        assert gif.n_frames == 13


def test_life_preset_and_random(tmp_path) -> None:
    out = tmp_path / "hl.gif"
    assert main(["life", "--preset", "highlife", "--random", "0.3", "--seed", "1", "--size",
                 "16x16", "--steps", "5", "--out", str(out)]) == 0  # fmt: skip
    with Image.open(out) as gif:
        assert gif.n_frames == 6


def test_lenia_gif_frames(tmp_path) -> None:
    out = tmp_path / "orbium.gif"
    assert main(["lenia", "--preset", "O2u", "--size", "64x64", "--steps", "6", "--scale", "2",
                 "--out", str(out)]) == 0  # fmt: skip
    with Image.open(out) as gif:
        assert gif.n_frames == 7


def test_nca_gif_frames_and_damage(tmp_path) -> None:
    out = tmp_path / "chest.gif"
    args = ["nca", "run", "--model", "chest-l2-sobel", "--steps", "15", "--damage-at", "10",
            "--scale", "2", "--out", str(out)]  # fmt: skip
    assert main(args) == 0
    with Image.open(out) as gif:
        assert gif.n_frames == 16
        assert gif.size == (56, 56)
    np.testing.assert_equal(main(["nca", "run", "--model", "nope", "--out", str(out)]), 2)


def test_list_commands(capsys) -> None:
    assert main(["list", "patterns"]) == 0
    assert "glider" in capsys.readouterr().out
    for what, needle in [("elementary", "110"), ("lifelike", "B3/S23"), ("lenia", "o2u"),
                         ("models", "chest-l2-sobel-damage")]:  # fmt: skip
        assert main(["list", what]) == 0
        assert needle in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["elementary", "--rule", "256", "--out", "x.png"],
        ["life", "--rule", "B9/S23", "--out", "x.gif"],
        ["life", "--preset", "nope", "--out", "x.gif"],
        ["life", "--pattern", "nope", "--out", "x.gif"],
        ["life", "--size", "banana", "--out", "x.gif"],
        ["lenia", "--preset", "life_as_lenia", "--out", "x.gif"],
        ["lenia", "--preset", "nope", "--out", "x.gif"],
        ["bogus"],
        [],
    ],
)
def test_user_errors_return_2(argv, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(argv) == 2
    assert not list(tmp_path.iterdir())  # nothing written on error


def test_help_returns_zero(capsys) -> None:
    assert main(["--help"]) == 0
