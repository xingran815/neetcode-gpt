import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        zmax = np.max(z)
        total_sum = np.sum(np.exp(z - zmax))
        return np.round(np.exp(z - zmax) / total_sum, 4)
