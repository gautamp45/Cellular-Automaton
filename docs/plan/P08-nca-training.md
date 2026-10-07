# P08: Port torch NCA training (with fixes)

**Tier:** Must · **Prereqs:** P05 · **Extra:** `[train]`

Port `neural-cellular-automata/src/{model,helper,train}.py` into `src/cellauto/nca/training/{model.py,losses.py,train.py}`. Make **only** these changes; everything else (pool sampling, worst-sample reseed, damaging the best 3) stays identical:

| Fix | Change |
|---|---|
| N2 | Honour `fire_rate`. |
| N3 | `self.register_buffer("kernel", k, persistent=False)`, built from `spec.perception_kernels`, so filter definitions live in exactly one place. `persistent=False` is **required** for old checkpoints to load with `strict=True`. |
| N4 | `torch.rand(x[:, :1].shape, device=x.device, generator=self.generator)`; optional `generator` constructor argument. |
| N6 | `LOSSES = {"l1","l2","manhattan","hinge"}`, case-insensitive, ValueError on unknown keys; formulas byte-for-byte identical. |
| N7 | The damage mask uses the seed's padded spatial size. |
| N8 | No medmnist. Targets come from `cellauto.nca.image.load_target` → `torch.from_numpy`. Delete `medmnist_loader.py`. |
| N9 | `device: null` means auto (cuda → mps → cpu). |
| N10 | No `plt.show()`. Write `runs/<id>/losses.csv` and `runs/<id>/config.json`. |
| N5 | `train(config: dict) -> Path`, CLI `cellauto nca train --config PATH`. Config paths are relative to the CWD (document "run from repo root"). |
| new | Seed control: `seed` in the config seeds `torch.manual_seed`, numpy's `default_rng`, and the generator. |
| new | At the end, save `models/nca/<id>.pt` (state_dict) **and** `<id>.npz`, and append or replace the manifest entry. |
| new | Optional config flags (default off, so the defaults reproduce the 2023 training): `grad_norm: bool` (Distill per-parameter normalisation `g/(‖g‖+1e-8)`) and `lr_schedule: {milestones:[...], gamma}`. P14 ablates them. |

Also add `TorchNCA.load(path_pt_or_npz, config)`, and create `configs/nca/chest-l2-sobel-damage.yaml` from the old config with new paths, `device: null`, `seed: 0`, and `id`.

## Tests: `tests/nca/test_torch.py` (`pytest.importorskip("torch")`, mark `torch`)
- All 25 `.pt` load into `TorchNCA` with `strict=True`.
- Parity: for `chest-l2-sobel`, from the seed at fire_rate 1.0, 10 steps in torch vs `NumpyNCA` agree to atol 1e-4.
- Smoke train: `iterations=2, pool_size=8, batch_size=2` writes a `.pt` and `.npz` into `tmp_path` (point the config there).

## Finish
`git rm -r neural-cellular-automata` (only `src/` and configs should remain by now); remove it from the ruff excludes.

**Commit:** `refactor(nca): torch training in cellauto.nca.training with fixes and seed control`
