import numpy as np
from typing import List, Callable

class ARCOperations:
    """Library Primitif DSL yang Diperluas untuk Tugas ARC-AGI."""

    @staticmethod
    def get_all_operations() -> List[Callable[[np.ndarray], np.ndarray]]:
        return [
            ARCOperations.identity,
            ARCOperations.fill_non_zero,
            ARCOperations.rotate_90,
            ARCOperations.rotate_180,
            ARCOperations.rotate_270,
            ARCOperations.flip_horizontal,
            ARCOperations.flip_vertical,
            ARCOperations.recolor_most_frequent,
            ARCOperations.crop_non_zero,
        ]

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
        return np.rot90(grid, k=1)

    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)

    @staticmethod
    def rotate_270(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=3)

    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)

    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)

    @staticmethod
    def recolor_most_frequent(grid: np.ndarray) -> np.ndarray:
        """Mengganti warna background (0) dengan warna objek yang paling sering muncul."""
        result = np.copy(grid)
        non_zero = result[result > 0]
        if len(non_zero) > 0:
            counts = np.bincount(non_zero)
            most_freq = np.argmax(counts)
            result[result == 0] = most_freq
        return result

    @staticmethod
    def crop_non_zero(grid: np.ndarray) -> np.ndarray:
        """Memotong (crop) grid hanya pada area yang memiliki objek non-nol."""
        coords = np.argwhere(grid > 0)
        if len(coords) == 0:
            return np.copy(grid)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return grid[y_min:y_max+1, x_min:x_max+1]
