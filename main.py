import os
import json
import glob
import numpy as np
from src.active_search import ActiveSearchSolver

def run_batch_evaluation(task_dir: str = "ARC-AGI/data/training", max_tasks: int = 5):
    """Mengevaluasi Active Search Solver pada beberapa tugas ARC secara otomatis."""
    json_files = sorted(glob.glob(os.path.join(task_dir, "*.json")))[:max_tasks]
    
    if not json_files:
        print(f"[Main] Tidak ada file JSON ditemukan di {task_dir}.")
        return

    print(f"[Main] Memulai Evaluasi Batch pada {len(json_files)} tugas ARC...\n")
    solver = ActiveSearchSolver(timeout_seconds=15, max_depth=2)
    
    results = {}
    total_passed = 0

    for filepath in json_files:
        task_name = os.path.basename(filepath).replace(".json", "")
        with open(filepath, "r") as f:
            task_data = json.load(f)

        train_examples = [(np.array(ex['input']), np.array(ex['output'])) for ex in task_data['train']]
        test_input = np.array(task_data['test'][0]['input'])
        test_target = np.array(task_data['test'][0]['output'])

        predicted_output = solver.solve(train_examples, test_input)
        is_correct = predicted_output is not None and np.array_equal(predicted_output, test_target)
        
        if is_correct:
            total_passed += 1
            results[task_name] = "SUKSES"
        else:
            results[task_name] = "GAGAL"

        print(f"Task {task_name}: {'✅ SUKSES' if is_correct else '❌ GAGAL'}\n")

    accuracy = (total_passed / len(json_files)) * 100
    print(f"--- RANGKUMAN EVALUASI BATCH ---")
    print(f"Total Tugas Diuji : {len(json_files)}")
    print(f"Berhasil Dikelola : {total_passed}")
    print(f"Tingkat Akurasi   : {accuracy:.1f}%\n")

if __name__ == "__main__":
    run_batch_evaluation()
