# P05: Neural CA, torch-free inference, checkpoint conversion, manifest

**Tier:** Must · **Prereqs:** P01

## 1. Move assets (use `git mv`)

| from | to |
|---|---|
| `neural-cellular-automata/data/{chest,blood,retina}.png` | `assets/nca/targets/` |
| `neural-cellular-automata/models/<t>/<Loss>/<t>_<filter>[_damage].pt` | `models/nca/<id>.pt` |
| `neural-cellular-automata/animations/<t>/<Loss>/<t>_<filter>[_damage].gif` | `assets/nca/animations/<id>.gif` |

`id = f"{target}-{loss.lower()}-{filter}" + ("-damage" if damage else "")`, e.g. `chest-l2-sobel-damage`. There are 25 ids. Use a throwaway loop; don't commit it. Leave `neural-cellular-automata/src` in place until P08.

## 2. `scripts/convert_checkpoints.py`

For each `models/nca/*.pt`, write `models/nca/<id>.npz` with `w1 (128,48)`, `b1 (128,)`, `w2 (16,128)` float32. These are the squeezed 1×1 conv weights, e.g. `w1 = sd["update_module.0.weight"][:, :, 0, 0]`. Use the **torch-free reader** below (verified on all 25). If torch is importable, also assert equality with `torch.load(p, map_location="cpu", weights_only=True)`. With `--manifest`, also write `models/nca/manifest.json` (below).

```python
import collections, pickle, zipfile
import numpy as np

def load_pt_state_dict(path) -> dict[str, np.ndarray]:
    """One-off conversion of TRUSTED repo files only. Never use on untrusted input (pickle)."""
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    def rebuild(storage, offset, size, stride, *_):
        return np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                               strides=[s * 4 for s in stride]).copy()
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if name == "OrderedDict": return collections.OrderedDict
            return lambda *a, **k: None
        def persistent_load(self, pid):  # ('storage', type, key, location, numel)
            return np.frombuffer(z.read(f"{root}/data/{pid[2]}"), dtype="<f4")
    return dict(Unpickler(z.open(f"{root}/data.pkl")).load())
```

## 3. `models/nca/manifest.json`

A list sorted by id:
```json
{"id":"chest-l2-sobel-damage","target":"chest","target_image":"assets/nca/targets/chest.png",
 "loss":"l2","filter":"sobel","trained_with_damage":true,
 "n_channels":16,"hidden_channels":128,"fire_rate":0.5,"grid_size":28,"padding":0,
 "weights":"models/nca/chest-l2-sobel-damage.npz","torch_checkpoint":"models/nca/chest-l2-sobel-damage.pt",
 "animation":"assets/nca/animations/chest-l2-sobel-damage.gif",
 "provenance":"trained 2023 with neural-cellular-automata/src/train.py (1000 iterations, lr 2e-3, batch 8, pool 1024)"}
```

## 4. Library modules

**`nca/spec.py`**
```python
FILTERS = {
  "sobel":     ([[-1,0,1],[-2,0,2],[-1,0,1]], 8.0),
  "scharr":    ([[-3,0,3],[-10,0,10],[-3,0,3]], 16.0),
  "gaussian":  ([[1,2,1],[2,4,2],[1,2,1]], 16.0),
  "laplacian": ([[0,1,0],[1,-4,1],[0,1,0]], 8.0),
  "mean":      ([[1,1,1],[1,1,1],[1,1,1]], 9.0),
}
ALPHA_CHANNEL = 3
ALIVE_THRESHOLD = 0.1
def perception_kernels(name) -> FloatArray   # (3,3,3) [identity, dx=k/s, dy=(k/s).T]
def is_isotropic(name) -> bool                # dx == dy  (gaussian, laplacian, mean)
@dataclass(frozen=True)
class NCAConfig: n_channels: int = 16; hidden_channels: int = 128; filter: str = "sobel"; fire_rate: float = 0.5
```

**`nca/image.py`** (numpy, channel-first float32): `load_target(path, size)` (RGBA, RGB premultiplied by alpha, `Image.Resampling.LANCZOS`, which fixes N1), `rgba_to_rgb(x) = clip(1 − clip(a,0,1) + rgb, 0, 1)`, `make_seed(size, n_channels=16, padding=0)` (channels 3: = 1 at the centre), `circle_mask(size, rng)` (centre ~ U(−0.5,0.5)², r ~ U(0.1,0.4) on a [−1,1] grid, mask = inside), `pad(x, p)`.

**`nca/numpy_model.py`**: the fast reference step (verified: ~1 ms per 28² step; 1000 steps ≈ 1 s):
```python
class NumpyNCA:
    def __init__(self, w1, b1, w2, config: NCAConfig): ...   # store kernels = perception_kernels(config.filter)
    @classmethod
    def from_npz(cls, path, config) -> "NumpyNCA": ...
    def alive(self, alpha):  # (H,W) → bool; 3x3 max-pool with -inf padding, > ALIVE_THRESHOLD
        p = np.pad(alpha, 1, constant_values=-np.inf); H, W = alpha.shape
        return np.max([p[i:i+H, j:j+W] for i in range(3) for j in range(3)], axis=0) > ALIVE_THRESHOLD
    def perceive(self, x):   # (C,H,W) → (3C, H*W), index 3c+k, zero padding, cross-correlation
        C, H, W = x.shape; p = np.pad(x, ((0,0),(1,1),(1,1)))
        sh = [p[:, i:i+H, j:j+W] for i in range(3) for j in range(3)]
        K = self.kernels
        per = [sum(K[k,i,j] * sh[3*i+j] for i in range(3) for j in range(3) if K[k,i,j] != 0) for k in range(3)]
        return np.stack(per, axis=1).reshape(3*C, H*W)
    def step(self, x, rng=None, fire_rate=None, return_dx=False):
        C, H, W = x.shape; fr = self.config.fire_rate if fire_rate is None else fire_rate
        hidden = np.maximum(self.w1 @ self.perceive(x) + self.b1[:, None], 0.0)
        dx = (self.w2 @ hidden).reshape(C, H, W)
        if fr < 1.0: dx = dx * (rng.random((H, W)) <= fr)        # fr == 1.0 must not consume RNG
        new = x + dx
        return (new * (self.alive(x[ALPHA_CHANNEL]) & self.alive(new[ALPHA_CHANNEL]))).astype(np.float32)
    def run(self, x, steps, rng=None, damage_at=(), damage_rng=None) -> Iterator[FloatArray]: ...
```
Also expose `hidden_activations(x)` for the frontend's "inside the cell" visualisation, which P07 exports in a fixture.

**`nca/checkpoints.py`**: `find_repo_root()` walks up from the CWD to the directory containing `models/nca/manifest.json`, or raises FileNotFoundError with a hint. Also `load_manifest()` and `load_model(id) -> NumpyNCA`. Models are repo data and are not shipped in the wheel.

## 5. Tests: `tests/nca/test_numpy_model.py`

- 25 manifest entries; every `.npz` loads with the expected shapes.
- `perception_kernels("sobel")[1] == sobel/8` and `[2] == ([1]).T`; `is_isotropic` is True exactly for gaussian, laplacian, mean.
- fire_rate 1.0: two runs are bit-identical and don't touch the RNG (pass `rng=None`).
- After 1 step from the seed, `x[3]` is nonzero only within the 3×3 block around the centre.
- **Convergence regression** (`slow`): every model with filter ∈ {sobel, scharr} and loss ≠ hinge (13 models), 200 steps, `default_rng(0)`, fire 0.5: MSE(RGBA vs target) < 0.005. The verified worst case is 0.0006.
- **Ablation sanity** (`slow`): every isotropic-filter model has MSE@200 > 0.005 (verified ≥ 0.0095). This guards the finding the README reports.

**Commit:** `feat(nca): torch-free numpy inference, checkpoint conversion, model manifest`
