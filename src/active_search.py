import time
from typing import List, Tuple, Optional, Callable
import numpy as np
from src.dsl import ARCOperations
from src.evaluator import TaskEvaluator

class ActiveSearchSolver:
    def __init__(self, timeout_seconds: int = 20, max_depth: int = 2):
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth
        self.operations = ARCOperations.get_all_operations()

    def solve(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray) -> Optional[np.ndarray]:
        start_time = time.time()
        best_score = 0.0
        best_fn = None

        def search(current_fn: Callable, depth: int):
            nonlocal best_score, best_fn
            if time.time() - start_time > self.timeout_seconds:
                return

            score = TaskEvaluator.evaluate_candidate(current_fn, train_examples)
            if score > best_score:
                best_score = score
                best_fn = current_fn

            if score == 1.0 or depth >= self.max_depth:
                return

            for op in self.operations:
                combined_fn = lambda g, f1=current_fn, f2=op: f2(f1(g))
                search(combined_fn, depth + 1)

        for op in self.operations:
            search(op, 1)
            if best_score == 1.0:
                break

        if best_fn is not None:
            try:
                return best_fn(test_input)
            except Exception:
                return None
        return None
