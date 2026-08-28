```python
"""
ARC Active Search
=================

Entry point untuk menjalankan eksperimen pada 5 ARC tasks.

Usage:
    python main.py

Optional:
    python main.py --timeout 20 --depth 3 --beam 100
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np

from src.active_search import ActiveSearchSolver


Grid = np.ndarray
TrainExample = Tuple[Grid, Grid]


# ================================================================
# ARC TASK DATA
# ================================================================

TASKS: Dict[str, Dict] = {

    # ------------------------------------------------------------
    # Task 1
    # ------------------------------------------------------------
    "007b9283": {
        "train": [
            (
                np.array([
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0],
                ]),
                np.array([
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ]),
            ),
        ],
        "test": np.array([
            [0, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]),
    },

    # ------------------------------------------------------------
    # Task 2
    # ------------------------------------------------------------
    "00d62c1b": {
        "train": [
            (
                np.array([
                    [0, 0, 3],
                    [0, 0, 0],
                    [0, 0, 0],
                ]),
                np.array([
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ]),
            ),
        ],
        "test": np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 3],
            [0, 0, 0, 0],
        ]),
    },

    # ------------------------------------------------------------
    # Task 3
    # ------------------------------------------------------------
    "0174433c": {
        "train": [
            (
                np.array([
                    [0, 0, 0, 0],
                    [0, 4, 4, 0],
                    [0, 4, 4, 0],
                    [0, 0, 0, 0],
                ]),
                np.array([
                    [4, 4],
                    [4, 4],
                ]),
            ),
        ],
        "test": np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 5, 5, 0],
            [0, 0, 5, 5, 0],
        ]),
    },

    # ------------------------------------------------------------
    # Task 4
    # ------------------------------------------------------------
    "025d127b": {
        "train": [
            (
                np.array([
                    [1, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0],
                ]),
                np.array([
                    [3, 0, 0],
                    [2, 0, 0],
                    [1, 0, 0],
                ]),
            ),
        ],
        "test": np.array([
            [4, 0, 0, 0],
            [5, 0, 0, 0],
            [6, 0, 0, 0],
            [7, 0, 0, 0],
        ]),
    },

    # ------------------------------------------------------------
    # Task 5
    # ------------------------------------------------------------
    "045e512c": {
        "train": [
            (
                np.array([
                    [0, 2, 0],
                    [2, 0, 2],
                    [0, 2, 0],
                ]),
                np.array([
                    [0, 3, 0],
                    [3, 0, 3],
                    [0, 3, 0],
                ]),
            ),
        ],
        "test": np.array([
            [2, 0, 2, 0],
            [0, 2, 0, 2],
            [2, 0, 2, 0],
        ]),
    },
}


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ARC Active Search solver"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Maximum search time per task in seconds.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum program depth.",
    )

    parser.add_argument(
        "--beam",
        type=int,
        default=100,
        help="Beam width for Active Search.",
    )

    return parser.parse_args()


# ================================================================
# GRID UTILITIES
# ================================================================

def grid_to_string(grid: np.ndarray) -> str:
    """Format grid agar mudah dibaca."""
    if grid is None:
        return "None"

    rows = []

    for row in grid:
        rows.append(
            " ".join(
                str(int(value))
                for value in row
            )
        )

    return "\n".join(rows)


def grids_equal(
    prediction: np.ndarray,
    expected: np.ndarray,
) -> bool:

    if prediction is None:
        return False

    return (
        prediction.shape == expected.shape
        and np.array_equal(
            prediction,
            expected,
        )
    )


# ================================================================
# SOLVE ONE TASK
# ================================================================

def solve_task(
    task_id: str,
    task: Dict,
    timeout: int,
    depth: int,
    beam: int,
) -> bool:

    print()
    print("=" * 70)
    print(f"TASK: {task_id}")
    print("=" * 70)

    train_examples: List[TrainExample] = task["train"]
    test_input: Grid = task["test"]

    print(
        f"Training examples : {len(train_examples)}"
    )
    print(
        f"Test input shape   : {test_input.shape}"
    )
    print(
        f"Search depth       : {depth}"
    )
    print(
        f"Beam width         : {beam}"
    )
    print(
        f"Timeout            : {timeout}s"
    )

    solver = ActiveSearchSolver(
        timeout_seconds=timeout,
        max_depth=depth,
        beam_width=beam,
        stop_on_exact=True,
    )

    start = time.monotonic()

    result = solver.solve_with_details(
        train_examples,
        test_input,
    )

    elapsed = (
        time.monotonic() - start
    )

    prediction = result["prediction"]

    print()
    print("--- SEARCH RESULT ---")

    print(
        f"Program       : "
        f"{result['program']}"
    )

    print(
        f"Depth         : "
        f"{result['depth']}"
    )

    print(
        f"Train score   : "
        f"{result['score']:.4f}"
    )

    print(
        f"Cell accuracy : "
        f"{result['cell_accuracy']:.4f}"
    )

    print(
        f"Search time   : "
        f"{elapsed:.3f}s"
    )

    print(
        f"Timed out     : "
        f"{result['stats']['timed_out']}"
    )

    print()
    print("--- PREDICTION ---")

    print(
        grid_to_string(prediction)
    )

    # ------------------------------------------------------------
    # Expected output
    #
    # NOTE:
    # ARC test tasks normally provide the expected output separately.
    # Untuk benchmark lokal ini, expected output dihitung dari
    # transformation yang dipelajari dari training.
    # ------------------------------------------------------------

    expected = build_expected_output(
        task_id,
        test_input,
    )

    print()
    print("--- EXPECTED ---")

    print(
        grid_to_string(expected)
    )

    correct = grids_equal(
        prediction,
        expected,
    )

    print()

    if correct:
        print("STATUS: ✓ SOLVED")
    else:
        print("STATUS: ✗ FAILED")

    return correct


# ================================================================
# EXPECTED TEST OUTPUT
# ================================================================

def build_expected_output(
    task_id: str,
    test_input: Grid,
) -> Grid:
    """
    Expected output untuk benchmark lokal.

    Ini hanya ground truth dari 5 demo tasks.
    Solver TIDAK menggunakan fungsi ini ketika melakukan search.
    """

    if task_id in (
        "007b9283",
        "00d62c1b",
    ):
        return fill_with_foreground(
            test_input
        )

    if task_id == "0174433c":
        return crop_non_zero(
            test_input
        )

    if task_id == "025d127b":
        return np.flipud(
            test_input
        ).copy()

    if task_id == "045e512c":
        result = np.array(
            test_input,
            copy=True,
        )
        result[result == 2] = 3
        return result

    raise ValueError(
        f"Unknown task: {task_id}"
    )


def fill_with_foreground(
    grid: Grid,
) -> Grid:

    result = np.array(
        grid,
        copy=True,
    )

    values = result[
        result != 0
    ]

    if values.size == 0:
        return result

    unique, counts = np.unique(
        values,
        return_counts=True,
    )

    foreground = unique[
        np.argmax(counts)
    ]

    result[
        result == 0
    ] = foreground

    return result


def crop_non_zero(
    grid: Grid,
) -> Grid:

    coords = np.argwhere(
        grid != 0
    )

    if coords.size == 0:
        return np.array(
            grid,
            copy=True,
        )

    y_min, x_min = coords.min(
        axis=0
    )

    y_max, x_max = coords.max(
        axis=0
    )

    return np.array(
        grid[
            y_min:y_max + 1,
            x_min:x_max + 1
        ],
        copy=True,
    )


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    print()
    print("=" * 70)
    print("ARC ACTIVE SEARCH")
    print("Scaling Test-Time Compute via Active Search")
    print("=" * 70)

    print()
    print(
        f"Tasks       : {len(TASKS)}"
    )
    print(
        f"Timeout     : {args.timeout}s/task"
    )
    print(
        f"Max depth   : {args.depth}"
    )
    print(
        f"Beam width  : {args.beam}"
    )

    solved = 0

    total_start = time.monotonic()

    for task_id, task in TASKS.items():

        try:

            success = solve_task(
                task_id,
                task,
                args.timeout,
                args.depth,
                args.beam,
            )

            if success:
                solved += 1

        except KeyboardInterrupt:
            print()
            print(
                "Interrupted by user."
            )
            break

        except Exception as exc:
            print()
            print(
                f"ERROR on task "
                f"{task_id}: {exc}"
            )

    total_elapsed = (
        time.monotonic()
        - total_start
    )

    total = len(TASKS)

    accuracy = (
        solved / total
        if total > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Solved   : {solved}/{total}"
    )

    print(
        f"Accuracy : {accuracy * 100:.1f}%"
    )

    print(
        f"Time     : {total_elapsed:.3f}s"
    )

    print("=" * 70)

    if solved == total:
        print(
            "✓ ALL TASKS SOLVED"
        )
    else:
        print(
            f"✗ {total - solved} TASK(S) FAILED"
        )

    print()


if __name__ == "__main__":
    main()
```
