```python
"""
ARC Candidate Evaluator

Evaluator bertanggung jawab untuk:
- menjalankan candidate program;
- memeriksa output;
- menghitung exact-match score;
- menghitung cell accuracy;
- memberikan diagnostic result.

Evaluator TIDAK melakukan search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


Grid = np.ndarray
Operation = Callable[[Grid], Grid]
TrainExample = Tuple[Grid, Grid]


@dataclass
class EvaluationResult:
    """Hasil evaluasi sebuah candidate."""

    score: float
    cell_accuracy: float
    exact: bool
    valid: bool
    correct_examples: int
    total_examples: int
    errors: int = 0

    @property
    def solved(self) -> bool:
        return self.exact and self.valid


class TaskEvaluator:
    """Evaluator untuk ARC training examples."""

    # ---------------------------------------------------------------
    # BASIC UTILITIES
    # ---------------------------------------------------------------

    @staticmethod
    def _safe_execute(
        candidate_fn: Operation,
        grid: Grid,
    ) -> Optional[Grid]:
        """Execute candidate dengan proteksi exception."""
        try:
            result = candidate_fn(grid)

            if result is None:
                return None

            result = np.asarray(result)

            if result.ndim != 2:
                return None

            if not np.issubdtype(result.dtype, np.integer):
                # ARC grid seharusnya integer.
                if np.all(np.equal(result, result.astype(int))):
                    result = result.astype(int)
                else:
                    return None

            return result

        except Exception:
            return None

    @staticmethod
    def exact_match(
        prediction: Optional[Grid],
        target: Grid,
    ) -> bool:
        if prediction is None:
            return False

        prediction = np.asarray(prediction)
        target = np.asarray(target)

        return (
            prediction.shape == target.shape
            and np.array_equal(prediction, target)
        )

    @staticmethod
    def cell_accuracy(
        prediction: Optional[Grid],
        target: Grid,
    ) -> float:
        """
        Cell-level accuracy.

        Jika shape berbeda, gunakan area overlap sebagai diagnostic,
        bukan sebagai exact solution.
        """
        if prediction is None:
            return 0.0

        prediction = np.asarray(prediction)
        target = np.asarray(target)

        if prediction.shape == target.shape:
            if target.size == 0:
                return 1.0

            return float(
                np.mean(prediction == target)
            )

        if prediction.ndim != 2 or target.ndim != 2:
            return 0.0

        rows = min(
            prediction.shape[0],
            target.shape[0],
        )
        cols = min(
            prediction.shape[1],
            target.shape[1],
        )

        if rows == 0 or cols == 0:
            return 0.0

        overlap = (
            prediction[:rows, :cols]
            == target[:rows, :cols]
        )

        return float(np.mean(overlap))

    # ---------------------------------------------------------------
    # FULL EVALUATION
    # ---------------------------------------------------------------

    @classmethod
    def evaluate(
        cls,
        candidate_fn: Operation,
        train_examples: List[TrainExample],
    ) -> EvaluationResult:

        if not train_examples:
            return EvaluationResult(
                score=0.0,
                cell_accuracy=0.0,
                exact=False,
                valid=False,
                correct_examples=0,
                total_examples=0,
            )

        exact_count = 0
        cell_scores = []
        errors = 0

        for train_input, train_output in train_examples:

            prediction = cls._safe_execute(
                candidate_fn,
                train_input,
            )

            if prediction is None:
                errors += 1
                cell_scores.append(0.0)
                continue

            if cls.exact_match(
                prediction,
                train_output,
            ):
                exact_count += 1

            cell_scores.append(
                cls.cell_accuracy(
                    prediction,
                    train_output,
                )
            )

        total = len(train_examples)

        score = exact_count / total

        average_cell_accuracy = (
            sum(cell_scores) / len(cell_scores)
            if cell_scores
            else 0.0
        )

        return EvaluationResult(
            score=score,
            cell_accuracy=average_cell_accuracy,
            exact=(exact_count == total),
            valid=(errors == 0),
            correct_examples=exact_count,
            total_examples=total,
            errors=errors,
        )

    # ---------------------------------------------------------------
    # BACKWARD COMPATIBILITY
    # ---------------------------------------------------------------

    @classmethod
    def evaluate_candidate(
        cls,
        candidate_fn: Operation,
        train_examples: List[TrainExample],
    ) -> float:
        """
        API lama.

        Tetap mengembalikan float agar kompatibel dengan code lama.
        """
        result = cls.evaluate(
            candidate_fn,
            train_examples,
        )

        return result.score

    @classmethod
    def is_solution(
        cls,
        candidate_fn: Operation,
        train_examples: List[TrainExample],
    ) -> bool:
        """True hanya jika SEMUA training examples exact match."""
        result = cls.evaluate(
            candidate_fn,
            train_examples,
        )

        return result.solved

    @classmethod
    def predict(
        cls,
        candidate_fn: Operation,
        test_input: Grid,
    ) -> Optional[Grid]:
        """Jalankan candidate ke test input."""
        return cls._safe_execute(
            candidate_fn,
            test_input,
        )
```
