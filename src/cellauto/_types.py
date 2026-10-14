from typing import Literal

import numpy as np
import numpy.typing as npt

Boundary = Literal["wrap", "dead"]
BinaryArray = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float32]
