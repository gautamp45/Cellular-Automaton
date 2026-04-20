import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'classical_ca', 'src'))

import numpy as np
from ca_1d import OneDimensionalCA
from ca_2d import TwoDimensionalCA

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


# ── 1D CA tests ─────────────────────────────────────────────────────────────

def test_1d_shape():
    ca = OneDimensionalCA(rule_number=30, width=51, generations=25)
    ca.initialize(1)
    ca.evolve()
    check("grid shape", ca.grid.shape == (25, 51), f"got {ca.grid.shape}")

def test_1d_rule30_not_trivial():
    # Rule 30 is chaotic — the grid should contain both 0s and 1s
    ca = OneDimensionalCA(rule_number=30, width=101, generations=50)
    ca.initialize(1)
    ca.evolve()
    has_zeros = np.any(ca.grid == 0)
    has_ones  = np.any(ca.grid == 1)
    check("Rule 30 has both 0s and 1s", has_zeros and has_ones)

def test_1d_rule90_symmetry():
    # Rule 90 with a single centre cell produces a symmetric (left-right) pattern
    width = 101
    ca = OneDimensionalCA(rule_number=90, width=width, generations=50)
    ca.initialize(1)
    ca.evolve()
    symmetric = all(np.array_equal(ca.grid[g], ca.grid[g, ::-1]) for g in range(50))
    check("Rule 90 grid is left-right symmetric", symmetric)

def test_1d_rule110_runs():
    # Rule 110 is Turing-complete — just verify it runs and produces output
    ca = OneDimensionalCA(rule_number=110, width=101, generations=50)
    ca.initialize(1)
    final = ca.evolve()
    check("Rule 110 returns a final state string", isinstance(final, str) and len(final) == 101)

def test_1d_rule184_returns_string():
    ca = OneDimensionalCA(rule_number=184, width=51, generations=30)
    ca.initialize(1)
    final = ca.evolve()
    check("Rule 184 final state is a binary string", all(c in '01' for c in final))

def test_1d_all_dead_stays_dead():
    # Rule 30 on an all-zero grid should stay all zero
    ca = OneDimensionalCA(rule_number=30, width=21, generations=10)
    ca.initialize(0)
    ca.evolve()
    # Row 0 is all-zero (init sets centre to 0); rows 1+ should also be zero
    # because rule 30: pattern (0,0,0) → 0
    check("All-zero init stays all-zero (rule 30)", np.all(ca.grid == 0))


# ── 2D CA tests ──────────────────────────────────────────────────────────────

def make_grid(arr, width=7, height=7):
    """Place a numpy pattern in the centre of a blank grid."""
    ca = TwoDimensionalCA(width, height, birth_rule=['3'], survival_rule=['2', '3'])
    h, w = arr.shape
    r = (height - h) // 2
    c = (width  - w) // 2
    ca.grid[r:r+h, c:c+w] = arr
    return ca

def test_2d_block_still_life():
    # 2×2 block is a still life — grid unchanged after any number of steps
    block = np.array([[1, 1],
                      [1, 1]])
    ca = make_grid(block)
    initial = ca.grid.copy()
    for _ in range(5):
        ca.evolve()
    check("Block (still life) unchanged after 5 steps", np.array_equal(ca.grid, initial))

def test_2d_blinker_period2():
    # Horizontal blinker returns to initial state after exactly 2 steps
    blinker = np.array([[1, 1, 1]])
    ca = make_grid(blinker, width=9, height=9)
    initial = ca.grid.copy()
    ca.evolve()
    after1 = ca.grid.copy()
    ca.evolve()
    same_after_2 = np.array_equal(ca.grid, initial)
    different_after_1 = not np.array_equal(after1, initial)
    check("Blinker changes after step 1", different_after_1)
    check("Blinker returns to initial after step 2", same_after_2)

def test_2d_glider_moves():
    # A glider in Conway's Life has period 4 and translates by (1,1) each period.
    # Use a large enough grid to give the glider room.
    ca = TwoDimensionalCA(20, 20, birth_rule=['3'], survival_rule=['2', '3'])
    # Standard glider at top-left area
    glider = [(1,2),(2,3),(3,1),(3,2),(3,3)]
    for r, c in glider:
        ca.grid[r][c] = 1
    initial_sum = ca.grid.sum()
    for _ in range(4):
        ca.evolve()
    # After one period the glider should still have 5 live cells
    check("Glider preserves cell count after 4 steps", ca.grid.sum() == initial_sum,
          f"expected 5 got {ca.grid.sum()}")

def test_2d_empty_stays_empty():
    ca = TwoDimensionalCA(10, 10, birth_rule=['3'], survival_rule=['2', '3'])
    ca.evolve()
    check("Empty grid stays empty", np.all(ca.grid == 0))

def test_2d_highlife_b36s23():
    # HighLife (B36/S23) — just verify it runs and evolves a non-trivial initial state
    ca = TwoDimensionalCA(20, 20, birth_rule=['3', '6'], survival_rule=['2', '3'])
    np.random.seed(0)
    ca.grid = np.random.randint(0, 2, (20, 20))
    initial = ca.grid.copy()
    ca.evolve()
    check("HighLife (B36/S23) grid changes after one step",
          not np.array_equal(ca.grid, initial))

def test_2d_maze_rule():
    # Maze (B3/S12345) — common Life-like rule; verify runs without error
    ca = TwoDimensionalCA(15, 15, birth_rule=['3'], survival_rule=['1','2','3','4','5'])
    np.random.seed(1)
    ca.grid = np.random.randint(0, 2, (15, 15))
    result = ca.evolve()
    check("Maze rule (B3/S12345) returns updated grid", result is not None)


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== 1D CA ===")
    test_1d_shape()
    test_1d_rule30_not_trivial()
    test_1d_rule90_symmetry()
    test_1d_rule110_runs()
    test_1d_rule184_returns_string()
    test_1d_all_dead_stays_dead()

    print("\n=== 2D CA ===")
    test_2d_block_still_life()
    test_2d_blinker_period2()
    test_2d_glider_moves()
    test_2d_empty_stays_empty()
    test_2d_highlife_b36s23()
    test_2d_maze_rule()
    print()
