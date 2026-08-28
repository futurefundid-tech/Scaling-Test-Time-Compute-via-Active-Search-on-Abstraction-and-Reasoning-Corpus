```python
"""
Active Search Solver for ARC.

Search strategy:
    Depth 1
        primitive operations

    Depth 2+
        composition of previously generated programs

Candidate ranking:
    1. exact training score
    2. cell accuracy
    3. lower program cost
    4. shorter program

Tujuan utama:
    menemukan program yang benar pada SELURUH training examples,
    kemudian menjalankannya pada test input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.dsl import ARCOperations, OperationSpec
from src.evaluator import TaskEvaluator


Grid = np.ndarray
TrainExample = Tuple[Grid, Grid]
Operation = Callable[[Grid], Grid]


@dataclass
class SearchNode:
    """Satu candidate program dalam search tree."""

    function: Operation
    name: str
    depth: int
    cost: float
    cell_accuracy: float = 0.0
    exact_score: float = 0.0
    exact: bool = False


class ActiveSearchSolver:
    """
    Active Search solver.

    Parameters
    ----------
    timeout_seconds:
        Batas waktu total search.

    max_depth:
        Kedalaman maksimum program.

    beam_width:
        Jumlah kandidat terbaik yang diteruskan ke depth berikutnya.

    stop_on_exact:
        Stop ketika seluruh training examples solved.
    """

    def __init__(
        self,
        timeout_seconds: int = 20,
        max_depth: int = 3,
        beam_width: int = 100,
        stop_on_exact: bool = True,
    ):
        self.timeout_seconds = max(
            1,
            int(timeout_seconds),
        )

        self.max_depth = max(
            1,
            int(max_depth),
        )

        self.beam_width = max(
            1,
            int(beam_width),
        )

        self.stop_on_exact = stop_on_exact

        self.operation_specs = (
            ARCOperations.get_operation_specs()
        )

        # Backward compatibility.
        self.operations = [
            spec.function
            for spec in self.operation_specs
        ]

        self.last_search_stats = {}

    # ================================================================
    # COMPOSITION
    # ================================================================

    @staticmethod
    def compose(
        first: Operation,
        second: Operation,
    ) -> Operation:
        """
        Return:
            second(first(grid))
        """

        def composed(grid: Grid) -> Grid:
            intermediate = first(grid)

            if intermediate is None:
                return None

            return second(intermediate)

        return composed

    # ================================================================
    # CANONICAL SIGNATURE
    # ================================================================

    @staticmethod
    def _grid_signature(grid: Optional[Grid]):
        """
        Signature output grid.

        Digunakan untuk deduplication candidate.
        """
        if grid is None:
            return None

        arr = np.asarray(grid)

        return (
            arr.shape,
            str(arr.dtype),
            arr.tobytes(),
        )

    @classmethod
    def _candidate_signature(
        cls,
        candidate: Operation,
        train_examples: List[TrainExample],
    ):
        """
        Signature candidate berdasarkan seluruh output training.

        Dua program yang menghasilkan output identik terhadap seluruh
        training set dianggap equivalent untuk tujuan search.
        """
        outputs = []

        for train_input, _ in train_examples:
            try:
                result = candidate(train_input)

                if result is None:
                    return None

                outputs.append(
                    cls._grid_signature(result)
                )

            except Exception:
                return None

        return tuple(outputs)

    # ================================================================
    # EVALUATION
    # ================================================================

    @staticmethod
    def _evaluate_node(
        node: SearchNode,
        train_examples: List[TrainExample],
    ) -> SearchNode:

        result = TaskEvaluator.evaluate(
            node.function,
            train_examples,
        )

        node.exact_score = result.score
        node.cell_accuracy = result.cell_accuracy
        node.exact = result.solved

        return node

    @staticmethod
    def _rank_key(node: SearchNode):
        """
        Higher is better.

        Exact score adalah prioritas utama.
        Cell accuracy menjadi tie-breaker.
        Cost/program length dipakai sebagai simplicity prior.
        """
        return (
            node.exact_score,
            node.cell_accuracy,
            -node.cost,
            -node.depth,
        )

    # ================================================================
    # DEPTH 1
    # ================================================================

    def _initial_nodes(
        self,
        train_examples: List[TrainExample],
    ) -> List[SearchNode]:

        nodes = []

        for spec in self.operation_specs:

            node = SearchNode(
                function=spec.function,
                name=spec.name,
                depth=1,
                cost=spec.cost,
            )

            self._evaluate_node(
                node,
                train_examples,
            )

            nodes.append(node)

        return nodes

    # ================================================================
    # EXPAND
    # ================================================================

    def _expand(
        self,
        parents: List[SearchNode],
        train_examples: List[TrainExample],
    ) -> List[SearchNode]:

        children = []

        for parent in parents:

            for spec in self.operation_specs:

                function = self.compose(
                    parent.function,
                    spec.function,
                )

                node = SearchNode(
                    function=function,
                    name=f"{parent.name} -> {spec.name}",
                    depth=parent.depth + 1,
                    cost=parent.cost + spec.cost,
                )

                self._evaluate_node(
                    node,
                    train_examples,
                )

                children.append(node)

        return children

    # ================================================================
    # DEDUPLICATION
    # ================================================================

    def _deduplicate(
        self,
        nodes: List[SearchNode],
        train_examples: List[TrainExample],
    ) -> List[SearchNode]:

        best_by_signature = {}

        for node in nodes:

            signature = self._candidate_signature(
                node.function,
                train_examples,
            )

            if signature is None:
                continue

            previous = best_by_signature.get(
                signature
            )

            if previous is None:
                best_by_signature[signature] = node
                continue

            if self._rank_key(node) > self._rank_key(previous):
                best_by_signature[signature] = node

        return list(
            best_by_signature.values()
        )

    # ================================================================
    # SEARCH
    # ================================================================

    def search(
        self,
        train_examples: List[TrainExample],
    ) -> Optional[SearchNode]:

        start_time = time.monotonic()

        self.last_search_stats = {
            "depth_reached": 0,
            "generated": 0,
            "evaluated": 0,
            "deduplicated": 0,
            "timed_out": False,
            "found_exact": False,
        }

        if not train_examples:
            return None

        # ------------------------------------------------------------
        # DEPTH 1
        # ------------------------------------------------------------

        nodes = self._initial_nodes(
            train_examples
        )

        self.last_search_stats["generated"] += len(nodes)
        self.last_search_stats["evaluated"] += len(nodes)
        self.last_search_stats["depth_reached"] = 1

        nodes = self._deduplicate(
            nodes,
            train_examples,
        )

        self.last_search_stats["deduplicated"] += len(nodes)

        nodes.sort(
            key=self._rank_key,
            reverse=True,
        )

        best = nodes[0] if nodes else None

        if best is not None and best.exact:
            self.last_search_stats["found_exact"] = True
            return best

        if self.max_depth == 1:
            return best

        # ------------------------------------------------------------
        # DEPTH 2+
        # ------------------------------------------------------------

        frontier = nodes[: self.beam_width]

        for depth in range(2, self.max_depth + 1):

            if (
                time.monotonic() - start_time
                >= self.timeout_seconds
            ):
                self.last_search_stats["timed_out"] = True
                break

            children = self._expand(
                frontier,
                train_examples,
            )

            self.last_search_stats["generated"] += len(
                children
            )

            self.last_search_stats["evaluated"] += len(
                children
            )

            self.last_search_stats["depth_reached"] = depth

            if not children:
                break

            children = self._deduplicate(
                children,
                train_examples,
            )

            self.last_search_stats["deduplicated"] += len(
                children
            )

            children.sort(
                key=self._rank_key,
                reverse=True,
            )

            if children:
                candidate = children[0]

                if (
                    best is None
                    or self._rank_key(candidate)
                    > self._rank_key(best)
                ):
                    best = candidate

                if candidate.exact:
                    self.last_search_stats[
                        "found_exact"
                    ] = True

                    if self.stop_on_exact:
                        return candidate

            frontier = children[
                : self.beam_width
            ]

            if not frontier:
                break

        return best

    # ================================================================
    # SOLVE
    # ================================================================

    def solve(
        self,
        train_examples: List[TrainExample],
        test_input: Grid,
    ) -> Optional[Grid]:

        node = self.search(
            train_examples
        )

        if node is None:
            return None

        prediction = TaskEvaluator.predict(
            node.function,
            test_input,
        )

        return prediction

    # ================================================================
    # DEBUGGING / EXPLANATION
    # ================================================================

    def solve_with_details(
        self,
        train_examples: List[TrainExample],
        test_input: Grid,
    ):
        """
        Sama seperti solve(), tetapi mengembalikan metadata search.

        Berguna untuk debugging dan eksperimen paper.
        """

        node = self.search(
            train_examples
        )

        if node is None:
            return {
                "prediction": None,
                "program": None,
                "depth": None,
                "score": 0.0,
                "cell_accuracy": 0.0,
                "stats": self.last_search_stats,
            }

        prediction = TaskEvaluator.predict(
            node.function,
            test_input,
        )

        return {
            "prediction": prediction,
            "program": node.name,
            "depth": node.depth,
            "score": node.exact_score,
            "cell_accuracy": node.cell_accuracy,
            "exact": node.exact,
            "cost": node.cost,
            "stats": self.last_search_stats,
        }
```
