```python
import time
from typing import List, Tuple, Optional, Callable
import numpy as np

from src.dsl import ARCOperations
from src.evaluator import TaskEvaluator


class ActiveSearchSolver:
    """
    Active Search solver untuk ARC.

    Strategi:
    1. Cari solusi dari primitive operations terlebih dahulu.
    2. Jika tidak ditemukan, cari kombinasi 2 operations.
    3. Kandidat hanya diterima jika cocok dengan SEMUA training examples.
    4. Hindari kandidat None / exception.
    5. Timeout tetap digunakan untuk mencegah search tidak berujung.
    """

    def __init__(self, timeout_seconds: int = 20, max_depth: int = 2):
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth
        self.operations = ARCOperations.get_all_operations()

    @staticmethod
    def compose(f1: Callable, f2: Callable) -> Callable:
        """
        Menghasilkan fungsi:
            f(g) = f2(f1(g))
        """
        def combined(grid):
            intermediate = f1(grid)
            if intermediate is None:
                return None
            return f2(intermediate)

        return combined

    @staticmethod
    def is_valid_solution(
        candidate_fn: Callable,
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """
        Candidate harus menghasilkan output yang identik
        dengan SEMUA training examples.
        """
        if not train_examples:
            return False

        for train_input, train_output in train_examples:
            try:
                prediction = candidate_fn(train_input)

                if prediction is None:
                    return False

                if not np.array_equal(prediction, train_output):
                    return False

            except Exception:
                return False

        return True

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> Optional[np.ndarray]:

        start_time = time.time()

        if not train_examples:
            return None

        # ==========================================================
        # DEPTH 1
        # ==========================================================
        # Prioritaskan exact solution.
        for op in self.operations:

            if time.time() - start_time > self.timeout_seconds:
                break

            if self.is_valid_solution(op, train_examples):
                try:
                    result = op(test_input)

                    if result is not None:
                        return result

                except Exception:
                    continue

        # ==========================================================
        # DEPTH 2
        # ==========================================================
        if self.max_depth >= 2:

            for op1 in self.operations:

                if time.time() - start_time > self.timeout_seconds:
                    return None

                for op2 in self.operations:

                    if time.time() - start_time > self.timeout_seconds:
                        return None

                    combined_fn = self.compose(op1, op2)

                    if self.is_valid_solution(
                        combined_fn,
                        train_examples
                    ):
                        try:
                            result = combined_fn(test_input)

                            if result is not None:
                                return result

                        except Exception:
                            continue

        return None
```
