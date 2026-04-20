import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'neural_ca', 'src'))

import torch
import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


# ── Model tests ───────────────────────────────────────────────────────────────

def test_model_all_filters():
    from model import NCA
    for f in ("sobel", "scharr", "gaussian", "laplacian", "mean"):
        try:
            NCA(n_channels=16, filter=f)
            check(f"NCA instantiates with filter='{f}'", True)
        except Exception as ex:
            check(f"NCA instantiates with filter='{f}'", False, str(ex))

def test_model_unknown_filter_raises():
    from model import NCA
    raised = False
    try:
        NCA(filter="nonsense")
    except ValueError:
        raised = True
    check("Unknown filter raises ValueError", raised)

def test_forward_pass_shape():
    from model import NCA
    model = NCA(n_channels=16, filter="sobel")
    x = torch.zeros(1, 16, 28, 28)
    x[:, 3:, 14, 14] = 1.0  # single seed
    out = model(x)
    check("Forward pass output shape matches input",
          out.shape == x.shape, f"got {out.shape}")

def test_forward_pass_batch():
    from model import NCA
    model = NCA(n_channels=16, filter="sobel")
    x = torch.zeros(4, 16, 28, 28)
    out = model(x)
    check("Forward pass works with batch size 4", out.shape == (4, 16, 28, 28))

def test_dead_cells_stay_dead():
    # An all-zero grid has no living cells; output should stay all zero
    from model import NCA
    model = NCA(n_channels=16, filter="sobel")
    x = torch.zeros(1, 16, 28, 28)
    out = model(x)
    check("All-zero input stays all-zero (no living cells)", torch.all(out == 0).item())

def test_get_alive():
    from model import NCA
    model = NCA()
    x = torch.zeros(1, 16, 10, 10)
    x[:, 3:4, 5, 5] = 1.0  # single live cell (alpha > 0.1)
    alive = model.get_alive(x)
    check("get_alive detects live cell neighbourhood",
          alive.any().item())

def test_multiple_forward_steps():
    from model import NCA
    model = NCA(n_channels=16, filter="sobel")
    x = torch.zeros(1, 16, 28, 28)
    x[:, 3:, 14, 14] = 1.0
    for _ in range(10):
        x = model(x)
    check("10 consecutive forward steps run without error", x.shape == (1, 16, 28, 28))


# ── Helper function tests ─────────────────────────────────────────────────────

def test_make_seed_shape():
    from helper import make_seed
    s = make_seed(28, n_channels=16)
    check("make_seed returns correct shape", s.shape == (1, 16, 28, 28))

def test_make_seed_only_centre_active():
    from helper import make_seed
    s = make_seed(28, n_channels=16)
    # All cells except the centre seed should be zero
    mask = torch.ones(1, 16, 28, 28, dtype=torch.bool)
    mask[:, 3:, 14, 14] = False
    check("make_seed: non-seed cells are all zero", torch.all(s[mask] == 0).item())
    check("make_seed: seed channels (3+) at centre are 1", torch.all(s[:, 3:, 14, 14] == 1).item())

def test_make_seed_invalid_channels():
    from helper import make_seed
    raised = False
    try:
        make_seed(28, n_channels=3)
    except ValueError:
        raised = True
    check("make_seed raises ValueError for n_channels < 4", raised)

def test_make_circle_masks_range():
    from helper import make_circle_masks
    m = make_circle_masks(28)
    check("make_circle_masks values are 0 or 1",
          torch.all((m == 0) | (m == 1)).item())
    check("make_circle_masks has both 0s and 1s",
          m.any().item() and not torch.all(m == 1).item())

def test_pad_image():
    from helper import pad_image
    x = torch.ones(1, 16, 10, 10)
    padded = pad_image(x, p=4)
    check("pad_image adds correct padding",
          padded.shape == (1, 16, 18, 18), f"got {padded.shape}")
    # Border should be zero
    check("pad_image border is zero", padded[:, :, 0, :].sum().item() == 0)

def test_rgba_to_rgb_shape():
    from helper import rgba_to_rgb
    x = torch.rand(1, 4, 28, 28)
    rgb = rgba_to_rgb(x)
    check("rgba_to_rgb output shape is (1, 3, 28, 28)",
          rgb.shape == (1, 3, 28, 28), f"got {rgb.shape}")

def test_rgba_to_rgb_range():
    from helper import rgba_to_rgb
    x = torch.rand(1, 4, 28, 28)
    rgb = rgba_to_rgb(x)
    check("rgba_to_rgb output values in [0, 1]",
          rgb.min().item() >= 0 and rgb.max().item() <= 1)


# ── Loss function tests ────────────────────────────────────────────────────────

def test_loss_identical_inputs_near_zero():
    from helper import L1, L2, Manhattan, Hinge
    cs = torch.rand(2, 16, 28, 28)
    target = cs[:, :4, ...].clone()
    for name, fn in [("L1", L1), ("L2", L2), ("Manhattan", Manhattan), ("Hinge", Hinge)]:
        _, loss = fn(target, cs)
        check(f"{name}(target, target) ≈ 0", loss.item() < 1e-5, f"got {loss.item():.2e}")

def test_loss_nonzero_for_different_inputs():
    from helper import L1, L2
    cs = torch.zeros(2, 16, 28, 28)
    target = torch.ones(2, 4, 28, 28)
    _, l1 = L1(target, cs)
    _, l2 = L2(target, cs)
    check("L1 > 0 for different inputs", l1.item() > 0, f"got {l1.item():.4f}")
    check("L2 > 0 for different inputs", l2.item() > 0, f"got {l2.item():.4f}")

def test_loss_batch_shape():
    from helper import L2
    cs = torch.rand(4, 16, 28, 28)
    target = torch.rand(4, 4, 28, 28)
    loss_batch, loss = L2(target, cs)
    check("L2 loss_batch shape is (batch_size,)",
          loss_batch.shape == (4,), f"got {loss_batch.shape}")
    check("L2 scalar loss is a 0-dim tensor", loss.dim() == 0)


# ── Mini training loop ────────────────────────────────────────────────────────

def test_mini_training_loop():
    from model import NCA
    from helper import make_seed, pad_image, load_image, L2
    import pathlib

    model = NCA(n_channels=8, filter="sobel")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Use a small synthetic target instead of loading a file
    target = torch.rand(1, 4, 8, 8)

    seed = make_seed(8, n_channels=8)
    pool = seed.clone().repeat(4, 1, 1, 1)

    try:
        for _ in range(3):
            cs = pool[:2].clone()
            for _ in range(5):
                cs = model(cs)
            _, loss = L2(target.repeat(2, 1, 1, 1), cs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        check("Mini training loop (3 iters) runs without error", True)
    except Exception as ex:
        check("Mini training loop (3 iters) runs without error", False, str(ex))


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Model ===")
    test_model_all_filters()
    test_model_unknown_filter_raises()
    test_forward_pass_shape()
    test_forward_pass_batch()
    test_dead_cells_stay_dead()
    test_get_alive()
    test_multiple_forward_steps()

    print("\n=== Helper Functions ===")
    test_make_seed_shape()
    test_make_seed_only_centre_active()
    test_make_seed_invalid_channels()
    test_make_circle_masks_range()
    test_pad_image()
    test_rgba_to_rgb_shape()
    test_rgba_to_rgb_range()

    print("\n=== Loss Functions ===")
    test_loss_identical_inputs_near_zero()
    test_loss_nonzero_for_different_inputs()
    test_loss_batch_shape()

    print("\n=== Training ===")
    test_mini_training_loop()
    print()
