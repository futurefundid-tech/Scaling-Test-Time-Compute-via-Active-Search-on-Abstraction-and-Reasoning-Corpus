import os
import json
import numpy as np
from src.active_search import ActiveSearchSolver

# 5 Tugas Simulasi ARC dengan variasi aturan visual
TASKS_DATA = {
    "007b9283": {
        "train": [{"input": [[0, 0], [1, 0]], "output": [[1, 1], [1, 1]]}],
        "test": [{"input": [[0, 2], [0, 0]], "output": [[2, 2], [2, 2]]}]
    },
    "00d62c1b": {
        "train": [{"input": [[0, 3, 3], [0, 3, 0]], "output": [[3, 3, 3], [3, 3, 3]]}],
        "test": [{"input": [[0, 0, 4], [4, 0, 0]], "output": [[4, 4, 4], [4, 4, 4]]}]
    },
    "0174433c": {
        "train": [{"input": [[0, 0, 0], [0, 5, 5], [0, 5, 5]], "output": [[5, 5], [5, 5]]}],
        "test": [{"input": [[0, 0, 0, 0], [0, 7, 7, 0], [0, 7, 7, 0], [0, 0, 0, 0]], "output": [[7, 7], [7, 7]]}]
    },
    "025d127b": {
        "train": [{"input": [[2, 0], [0, 0]], "output": [[0, 0], [2, 0]]}],
        "test": [{"input": [[9, 9], [0, 0]], "output": [[0, 0], [9, 9]]}]
    },
    "045e512c": {
        "train": [{"input": [[2, 2], [0, 2]], "output": [[3, 3], [0, 3]]}],
        "test": [{"input": [[2, 0], [2, 2]], "output": [[3, 0], [3, 3]]}]
    }
}

def run_batch_evaluation():
    print("==========================================")
    print(f"[Main] Memulai Evaluasi Batch pada {len(TASKS_DATA)} tugas ARC...")
    print("==========================================\n")
    
    solver = ActiveSearchSolver(timeout_seconds=20, max_depth=2)
    total_passed = 0

    for task_id, task_data in TASKS_DATA.items():
        train_examples = [(np.array(ex['input']), np.array(ex['output'])) for ex in task_data['train']]
        test_input = np.array(task_data['test'][0]['input'])
        test_target = np.array(task_data['test'][0]['output'])

        print(f"--- Menguji Task ID: {task_id} ---")
        predicted_output = solver.solve(train_examples, test_input)
        is_correct = predicted_output is not None and np.array_equal(predicted_output, test_target)
        
        if is_correct:
            total_passed += 1
            print(f"Hasil Task {task_id}: ✅ SUKSES\n")
        else:
            print(f"Hasil Task {task_id}: ❌ GAGAL\n")

    accuracy = (total_passed / len(TASKS_DATA)) * 100
    print("==========================================")
    print("--- RANGKUMAN EVALUASI BATCH ---")
    print(f"Total Soal Diuji  : {len(TASKS_DATA)}/{len(TASKS_DATA)}")
    print(f"Berhasil Dikelola : {total_passed}")
    print(f"Tingkat Akurasi   : {accuracy:.1f}%")
    print("==========================================")

if __name__ == "__main__":
    run_batch_evaluation()
