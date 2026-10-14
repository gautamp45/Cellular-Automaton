"""PyTorch NCA, architecturally identical to the 2023 model (all 25 checkpoints load strictly).

Changes from the original: ``fire_rate`` is honoured (N2); the perception kernel is a
non-persistent buffer built from :func:`cellauto.nca.spec.perception_kernels` (N3); the fire mask
can use an explicit ``torch.Generator`` (N4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from cellauto.nca.spec import ALIVE_THRESHOLD, ALPHA_CHANNEL, NCAConfig, perception_kernels


class TorchNCA(nn.Module):
    kernel: torch.Tensor

    def __init__(
        self,
        config: NCAConfig | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NCAConfig()
        c, hidden = self.config.n_channels, self.config.hidden_channels
        self.n_channels = c
        self.fire_rate = self.config.fire_rate
        self.generator = generator

        # (3C, 1, 3, 3): output index 3*c + k for k in (identity, dx, dy), applied depthwise.
        # persistent=False keeps state_dict() identical to the original checkpoints.
        kernel = torch.from_numpy(perception_kernels(self.config.filter)).repeat((c, 1, 1))
        self.register_buffer("kernel", kernel[:, None, ...], persistent=False)

        self.update_module = nn.Sequential(
            nn.Conv2d(3 * c, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, c, kernel_size=1, bias=False),
        )
        with torch.no_grad():
            self.update_module[2].weight.zero_()
        self.to(torch.device(device) if device is not None else torch.device("cpu"))

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(x, self.kernel, padding=1, groups=self.n_channels)

    @staticmethod
    def get_alive(x: torch.Tensor) -> torch.Tensor:
        a = x[:, ALPHA_CHANNEL : ALPHA_CHANNEL + 1, :, :]
        return nn.functional.max_pool2d(a, kernel_size=3, stride=1, padding=1) > ALIVE_THRESHOLD

    def update(self, x: torch.Tensor) -> torch.Tensor:
        pre_life_mask = self.get_alive(x)
        dx = self.update_module(self.perceive(x))
        mask = (
            torch.rand(x[:, :1, :, :].shape, device=x.device, generator=self.generator)
            <= self.fire_rate
        ).to(torch.float32)
        new_x = x + dx * mask
        post_life_mask = self.get_alive(new_x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        return new_x * life_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.update(x)

    def numpy_weights(self) -> dict[str, np.ndarray]:
        """``w1 (hidden, 3C)``, ``b1 (hidden,)``, ``w2 (C, hidden)`` as float32 numpy arrays."""
        sd = self.update_module.state_dict()
        return {
            "w1": sd["0.weight"][:, :, 0, 0].detach().cpu().numpy().astype(np.float32),
            "b1": sd["0.bias"].detach().cpu().numpy().astype(np.float32),
            "w2": sd["2.weight"][:, :, 0, 0].detach().cpu().numpy().astype(np.float32),
        }

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: NCAConfig | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> TorchNCA:
        """Load a ``.pt`` state dict (strictly) or an ``.npz`` of ``w1, b1, w2``."""
        model = cls(config, device, generator)
        p = Path(path)
        if p.suffix == ".npz":
            with np.load(p) as z:
                w1, b1, w2 = (torch.from_numpy(z[k]) for k in ("w1", "b1", "w2"))
            state = {
                "update_module.0.weight": w1[:, :, None, None],
                "update_module.0.bias": b1,
                "update_module.2.weight": w2[:, :, None, None],
            }
        else:
            state = torch.load(p, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model
