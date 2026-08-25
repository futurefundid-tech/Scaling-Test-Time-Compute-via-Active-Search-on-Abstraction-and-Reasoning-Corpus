import numpy as np

class ARCOperations:
    """Kumpulan transformasi DSL dasar untuk grid ARC."""
    
    # --- 1. Transformasi Geometri ---
    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=1)

    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)

    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)

    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)

    # --- 2. Manipulasi Warna ---
    @staticmethod
    def replace_color(grid: np.ndarray, color_from: int = 1, color_to: int = 2) -> np.ndarray:
        """Mengganti satu warna spesifik menjadi warna lain."""
        new_grid = grid.copy()
        new_grid[grid == color_from] = color_to
        return new_grid

    @staticmethod
    def invert_colors(grid: np.ndarray) -> np.ndarray:
        """Peta pembalikan warna dasar ARC (0-9)."""
        return np.where(grid > 0, 9 - grid, 0)

    # --- 3. Manipulasi Ukuran & Objek ---
    @staticmethod
    def crop_bounding_box(grid: np.ndarray) -> np.ndarray:
        """Memotong grid untuk mengambil area non-background (non-zero)."""
        non_zero_coords = np.argwhere(grid > 0)
        if non_zero_coords.size == 0:
            return grid
        
        r_min, c_min = non_zero_coords.min(axis=0)
        r_max, c_max = non_zero_coords.max(axis=0)
        return grid[r_min:r_max+1, c_min:c_max+1]
