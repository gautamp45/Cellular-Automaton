"""Training losses. Each returns ``(loss_batch, loss)`` for a target and a cell state.

Only the first four channels (RGBA) of the state are compared. The formulas are identical to the
original 2023 implementation; only the lookup is new (case-insensitive, strict).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Loss = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def l1(target: torch.Tensor, cs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss_batch = (torch.abs(target - cs[:, :4, ...])).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def l2(target: torch.Tensor, cs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss_batch = ((target - cs[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def manhattan(target: torch.Tensor, cs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss_batch = (torch.abs(target - cs[:, :4, ...])).sum(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def hinge(target: torch.Tensor, cs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss_batch = torch.max(torch.abs(target - cs[:, :4, ...]) - 0.5, torch.zeros_like(target)).mean(
        dim=[1, 2, 3]
    )
    return loss_batch, loss_batch.mean()


LOSSES: dict[str, Loss] = {"l1": l1, "l2": l2, "manhattan": manhattan, "hinge": hinge}


def get_loss(name: str) -> Loss:
    try:
        return LOSSES[name.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown loss {name!r}; expected one of {sorted(LOSSES)}") from None
