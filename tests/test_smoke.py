import numpy as np
import pytest

import cellauto
from cellauto._validate import as_binary, check_boundary, check_int_range


def test_version() -> None:
    assert cellauto.__version__ == "0.1.0"


def test_as_binary() -> None:
    src = np.array([0, 1, 1])
    out = as_binary(src, 1)
    assert out.dtype == np.uint8
    out[0] = 1
    assert src[0] == 0  # copy
    with pytest.raises(ValueError):
        as_binary([0, 2], 1)
    with pytest.raises(ValueError):
        as_binary([[0, 1]], 1)


def test_check_boundary_and_range() -> None:
    check_boundary("wrap")
    with pytest.raises(ValueError):
        check_boundary("torus")
    assert check_int_range("x", 3, 0, 5) == 3
    with pytest.raises(ValueError):
        check_int_range("x", True, 0, 5)
    with pytest.raises(ValueError):
        check_int_range("x", 6, 0, 5)
