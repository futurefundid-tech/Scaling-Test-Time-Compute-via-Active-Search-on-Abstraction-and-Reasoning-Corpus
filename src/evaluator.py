from typing import Callable, List, Tuple
import numpy as np

class TaskEvaluator:
    """Modul untuk menilai kesesuaian (fitness) fungsi DSL terhadap contoh latihan ARC."""

    @staticmethod
    def evaluate_candidate(candidate_fn: Callable, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Mengembalikan skor akurasi (0.0 - 1.0) dari kandidat fungsi pada set latihan."""
        correct_count = 0
        total_examples = len(train_examples)

        if total_examples == 0:
            return 0.0

        for input_grid, target_grid in train_examples:
            try:
                predicted_grid = candidate_fn(input_grid)
                if np.array_equal(predicted_grid, target_grid):
                    correct_count += 1
            except Exception:
                # Jika transformasi gagal (misal ukuran tidak cocok), lewati
                pass

        return correct_count / total_examples
