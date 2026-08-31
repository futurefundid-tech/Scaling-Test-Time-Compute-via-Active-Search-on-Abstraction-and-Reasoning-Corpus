import time
from typing import List, Tuple, Optional, Callable
import numpy as np
from src.dsl import ARCOperations

class ActiveSearchSolver:
    def __init__(self, timeout_seconds: int = 20, max_depth: int = 2):
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth
        self.operations = ARCOperations.get_all_operations()

    def _evaluate(self, fn: Callable, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        correct = 0
        for train_in, train_out in train_examples:
            try:
                pred = fn(train_in)
                if pred is not None and np.array_equal(pred, train_out):
                    correct += 1
            except Exception:
                return 0.0
        return correct / len(train_examples)

    def solve(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray) -> Optional[np.ndarray]:
        start_time = time.time()
        best_score = 0.0
        best_fn = None

        # Search Depth 1
        for op in self.operations:
            if time.time() - start_time > self.timeout_seconds:
                break
            score = self._evaluate(op, train_examples)
            if score > best_score:
                best_score = score
                best_fn = op

        # Search Depth 2
        if self.max_depth >= 2:
            for op1 in self.operations:
                for op2 in self.operations:
                    if time.time() - start_time > self.timeout_seconds:
                        break
                    combined_fn = lambda g, f1=op1, f2=op2: f2(f1(g))
                    score = self._evaluate(combined_fn, train_examples)
                    if score > best_score:
                        best_score = score
                        best_fn = combined_fn

        if best_fn is not None:
            try:
                return best_fn(test_input)
            except Exception:
                return None
        return None
