import os
import json
import urllib.request
import numpy as np
from src.active_search import ActiveSearchSolver

# 5 ID soal ARC asli untuk pengujian batch
SAMPLE_TASKS = ["007b9283", "00d62c1b", "0174433c", "025d127b", "045e512c"]

def load_task(task_id: str) -> dict:
    """Memuat file JSON soal ARC baik dari disk lokal maupun unduhan langsung."""
    local_path = f"ARC-AGI/data/training/{task_id}.json"
    if os.path.exists(local_path):
        with open(local_path, "r") as f:
            return json.load(f)
    
    # Download otomatis dari repository resmi jika file lokal tidak ada
    url = f"https://raw.githubusercontent.com/fchollet/ARC-AGI/main/data/training/{task_id}.json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"[Warning] Gagal memuat task {task_id}: {e}")
        return None

def run_batch_evaluation():
    print(f"[Main] Memulai Evaluasi Batch pada {len(SAMPLE_TASKS)} tugas ARC...\n")
    solver = ActiveSearchSolver(timeout_seconds=10, max_depth=2)
    
    total_passed = 0
    valid_tasks = 0

    for task_id in SAMPLE_TASKS:
        task_data = load_task(task_id)
        if task_data is None:
            continue
            
        valid_tasks += 1
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

    accuracy = (total_passed / valid_tasks * 100) if valid_tasks > 0 else 0
    print(f"==========================================")
    print(f"--- RANGKUMAN EVALUASI BATCH ---")
    print(f"Total Soal Diuji  : {valid_tasks}")
    print(f"Berhasil Dikelola : {total_passed}")
    print(f"Tingkat Akurasi   : {accuracy:.1f}%")
    print(f"==========================================")

if __name__ == "__main__":
    run_batch_evaluation()
