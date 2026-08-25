import numpy as np
from typing import List, Callable

class ARCOperations:
    """Primitif DSL untuk tugas ARC."""

    @staticmethod
    def get_all_operations() -> List[Callable[[np.ndarray], np.ndarray]]:
        return [
            ARCOperations.identity,
            ARCOperations.fill_non_zero,
            ARCOperations.rotate_90,
            ARCOperations.flip_horizontal,
            ARCOperations.flip_vertical,
        ]

    # Alias untuk kompatibilitas
    @staticmethod
    def get_all_primitives() -> List[Callable[[np.ndarray], np.ndarray]]:
        return ARCOperations.get_all_operations()

    @staticmethod
    def identity(grid: np.ndarray) -> np.ndarray:
        return np.copy(grid)

    @staticmethod
    def fill_non_zero(grid: np.ndarray) -> np.ndarray:
        result = np.copy(grid)
        non_zero_vals = result[result > 0]
        if len(non_zero_vals) > 0:
            val = non_zero_vals[0]
            result[result == 0] = val
        return result

    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid)

    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)

    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)
