import numpy as np
from typing import List, Tuple, Callable

class TaskEvaluator:
    @staticmethod
    def evaluate_candidate(candidate_fn: Callable, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Menghitung akurasi kandidat fungsi terhadap contoh data latih."""
        if not train_examples:
            return 0.0

        correct = 0
        for train_in, train_out in train_examples:
            try:
                pred = candidate_fn(train_in)
                if pred is not None and np.array_equal(pred, train_out):
                    correct += 1
            except Exception:
                return 0.0

        return correct / len(train_examples)
