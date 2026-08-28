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

        def compose(f1: Callable, f2: Callable) -> Callable:
            return lambda g: f2(f1(g))

        # 1. Depth 1 Evaluation
        for op in self.operations:
            if time.time() - start_time > self.timeout_seconds:
                break
            score = TaskEvaluator.evaluate_candidate(op, train_examples)
            if score > best_score:
                best_score = score
                best_fn = op
            if best_score == 1.0:
                break

        # 2. Depth 2 Evaluation
        if best_score < 1.0 and self.max_depth >= 2:
            for op1 in self.operations:
                for op2 in self.operations:
                    if time.time() - start_time > self.timeout_seconds:
                        break
                    combined_fn = compose(op1, op2)
                    score = TaskEvaluator.evaluate_candidate(combined_fn, train_examples)
                    if score > best_score:
                        best_score = score
                        best_fn = combined_fn
                    if best_score == 1.0:
                        break
                if best_score == 1.0:
                    break

        if best_fn is not None:
            try:
                return best_fn(test_input)
            except Exception:
                return None
        return None
