# P06: Renderer and CLI

**Tier:** Must · **Prereqs:** P02–P05

## `src/cellauto/render.py` (Pillow only; no matplotlib/pygame)

```python
PALETTES: dict[str, tuple[tuple[int,int,int], tuple[int,int,int]]]  # e.g. "dark": ((12,12,16),(240,240,240))
def binary_to_image(grid, scale=4, palette="dark") -> Image.Image
def scalar_to_image(a, scale=4, cmap="magma") -> Image.Image   # Lenia: 256-entry LUT; define 'magma' & 'gray' inline as uint8 arrays (no matplotlib)
def rgb_to_image(rgb, scale=4) -> Image.Image                   # (3,H,W) or (H,W,3) float in [0,1]
def save_png(img, path) -> None
def save_gif(frames: Iterable[Image.Image], path, fps=20) -> None   # save_all=True, duration=1000//fps, loop=0
```
Scale with `Image.Resampling.NEAREST`. For the magma LUT, generate a 256×3 uint8 table **once** in a helper script and paste it as a constant. Don't add matplotlib as a dependency.

## `src/cellauto/cli.py` (argparse subcommands)

```
cellauto elementary --rule 30 [--width 201] [--steps 100] [--init center|random] [--density 0.5] [--seed N]
                    [--boundary wrap|dead] [--scale 4] --out rule30.png
cellauto life [--rule B3/S23 | --preset highlife] [--pattern glider | --random 0.3] [--seed N]
              [--size 64x64] [--steps 200] [--boundary wrap|dead] [--fps 20] [--scale 6] --out life.gif
cellauto lenia [--preset O2u] [--size 128x128] [--steps 300] [--scale 3] --out orbium.gif
cellauto nca run --model chest-l2-sobel [--steps 200] [--fire-rate 0.5] [--seed 0] [--damage-at 100 ...] --out chest.gif
cellauto nca train --config configs/nca/chest-l2-sobel-damage.yaml          # P08, needs [train]
cellauto list {elementary|lifelike|patterns|lenia|models}
cellauto export-web [--out web/public/data] [--check]                          # P07
cellauto analyze {eca-atlas|lifelike-atlas|nca-eval|inference-study} [--quick]  # P09–P12
cellauto experiment learnability [--quick]                                     # P13
```
`main(argv=None) -> int`. User errors print to stderr and return exit code 2. Missing torch prints `pip install -e ".[train]"` and returns 2.

## Tests: `tests/test_cli.py`, `tests/test_render.py`
- elementary PNG has size `(width*scale, (steps+1)*scale)`.
- life/lenia/nca GIFs exist and have `n_frames == steps + 1` (open with PIL and read `n_frames`).
- `list patterns` stdout contains `glider`. An invalid rule returns 2.

**Commit:** `feat: pillow renderer and cellauto CLI`
