```python
"""
ARC Domain Specific Language (DSL)

Kumpulan primitive transformation yang digunakan oleh Active Search.

Desain:
- Setiap operation menerima np.ndarray.
- Setiap operation mengembalikan np.ndarray baru.
- Operation tidak memodifikasi input.
- Metadata operation tersedia untuk search/debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


Grid = np.ndarray
Operation = Callable[[Grid], Grid]


@dataclass(frozen=True)
class OperationSpec:
    """Deskripsi sebuah primitive operation."""

    name: str
    function: Operation
    cost: float = 1.0

    def __call__(self, grid: Grid) -> Grid:
        return self.function(grid)


class ARCOperations:
    """ARC DSL primitives."""

    # ------------------------------------------------------------------
    # BASIC
    # ------------------------------------------------------------------

    @staticmethod
    def identity(grid: Grid) -> Grid:
        return np.array(grid, copy=True)

    # ------------------------------------------------------------------
    # GEOMETRIC
    # ------------------------------------------------------------------

    @staticmethod
    def rotate_90(grid: Grid) -> Grid:
        return np.rot90(grid, 1).copy()

    @staticmethod
    def rotate_180(grid: Grid) -> Grid:
        return np.rot90(grid, 2).copy()

    @staticmethod
    def rotate_270(grid: Grid) -> Grid:
        return np.rot90(grid, 3).copy()

    @staticmethod
    def flip_horizontal(grid: Grid) -> Grid:
        # kiri <-> kanan
        return np.fliplr(grid).copy()

    @staticmethod
    def flip_vertical(grid: Grid) -> Grid:
        # atas <-> bawah
        return np.flipud(grid).copy()

    # ------------------------------------------------------------------
    # COLOR / VALUE
    # ------------------------------------------------------------------

    @staticmethod
    def replace_2_with_3(grid: Grid) -> Grid:
        result = np.array(grid, copy=True)
        result[result == 2] = 3
        return result

    @staticmethod
    def replace_3_with_2(grid: Grid) -> Grid:
        result = np.array(grid, copy=True)
        result[result == 3] = 2
        return result

    @staticmethod
    def fill_non_zero(grid: Grid) -> Grid:
        """
        Isi seluruh background (0) dengan satu warna foreground.

        Warna dipilih deterministically:
        - ambil warna non-zero yang paling sering muncul;
        - jika frekuensi sama, pilih nilai warna terkecil.
        """
        result = np.array(grid, copy=True)

        values = result[result != 0]

        if values.size == 0:
            return result

        unique, counts = np.unique(values, return_counts=True)

        max_count = counts.max()
        candidates = unique[counts == max_count]

        fill_value = candidates.min()

        result[result == 0] = fill_value
        return result

    @staticmethod
    def recolor_most_frequent(grid: Grid) -> Grid:
        """
        Recolor background (0) menggunakan warna foreground
        yang paling sering muncul.
        """
        return ARCOperations.fill_non_zero(grid)

    # ------------------------------------------------------------------
    # OBJECT / CROP
    # ------------------------------------------------------------------

    @staticmethod
    def crop_non_zero(grid: Grid) -> Grid:
        """
        Crop bounding box dari seluruh cell non-zero.
        """
        coords = np.argwhere(grid != 0)

        if coords.size == 0:
            return np.array(grid, copy=True)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return np.array(
            grid[y_min : y_max + 1, x_min : x_max + 1],
            copy=True,
        )

    # ------------------------------------------------------------------
    # ADDITIONAL SAFE PRIMITIVES
    # ------------------------------------------------------------------

    @staticmethod
    def transpose(grid: Grid) -> Grid:
        return np.transpose(grid).copy()

    @staticmethod
    def invert_colors(grid: Grid) -> Grid:
        """
        ARC biasanya menggunakan 0 sebagai background.
        Operation ini hanya menukar nilai non-zero:
        min foreground <-> max foreground.
        """
        result = np.array(grid, copy=True)

        values = result[result != 0]

        if values.size == 0:
            return result

        low = values.min()
        high = values.max()

        if low == high:
            return result

        result[result == low] = high
        result[result == high] = low

        return result

    # ------------------------------------------------------------------
    # OPERATION REGISTRY
    # ------------------------------------------------------------------

    @classmethod
    def get_operation_specs(cls) -> List[OperationSpec]:
        """
        Return seluruh primitive dalam urutan deterministic.

        Urutan penting karena Active Search menggunakan deterministic
        traversal sehingga eksperimen dapat direproduksi.
        """
        return [
            OperationSpec("identity", cls.identity),
            OperationSpec("rotate_90", cls.rotate_90),
            OperationSpec("rotate_180", cls.rotate_180),
            OperationSpec("rotate_270", cls.rotate_270),
            OperationSpec("flip_horizontal", cls.flip_horizontal),
            OperationSpec("flip_vertical", cls.flip_vertical),
            OperationSpec("transpose", cls.transpose),
            OperationSpec("fill_non_zero", cls.fill_non_zero),
            OperationSpec(
                "recolor_most_frequent",
                cls.recolor_most_frequent,
            ),
            OperationSpec("crop_non_zero", cls.crop_non_zero),
            OperationSpec("replace_2_with_3", cls.replace_2_with_3),
            OperationSpec("replace_3_with_2", cls.replace_3_with_2),
            OperationSpec("invert_colors", cls.invert_colors),
        ]

    @classmethod
    def get_all_operations(cls) -> List[Operation]:
        """
        Backward-compatible API.

        main.py / solver lama mungkin masih memanggil:
            ARCOperations.get_all_operations()
        """
        return [
            spec.function
            for spec in cls.get_operation_specs()
        ]
```
