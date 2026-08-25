import os
import json
import glob
import numpy as np
from src.active_search import ActiveSearchSolver

# Cari file 007b9283.json secara otomatis di seluruh direktori
task_id = "007b9283.json"
found_files = glob.glob(f"**/{task_id}", recursive=True)

if found_files:
    local_dataset_path = found_files[0]
    print(f"[Main] Memuat data tugas ARC dari file lokal: {local_dataset_path}...")
    with open(local_dataset_path, "r") as f:
        task_data = json.load(f)
else:
    print("[Main] File dataset ARC lokal tidak ditemukan, menggunakan data simulasi ARC...")
    task_data = {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
            {"input": [[0, 1], [0, 0]], "output": [[0, 0], [0, 1]]}
        ],
        "test": [
            {"input": [[0, 0], [1, 0]], "output": [[0, 0], [0, 1]]}
        ]
    }

# Konversi format data JSON ke numpy array
train_examples = [
    (np.array(ex['input']), np.array(ex['output']))
    for ex in task_data['train']
]
test_input = np.array(task_data['test'][0]['input'])
test_target = np.array(task_data['test'][0]['output'])

print(f"[Main] Berhasil memuat {len(train_examples)} contoh latihan.")

# Jalankan Solver Active Search
solver = ActiveSearchSolver(timeout_seconds=30, max_depth=2)
predicted_output = solver.solve(train_examples, test_input)

# Verifikasi Hasil Prediksi
if predicted_output is not None:
    is_correct = np.array_equal(predicted_output, test_target)
    print(f"\n--- Hasil Evaluasi Tugas ARC ---")
    print(f"Status Akurasi Tugas: {'100% SUKSES' if is_correct else 'BELUM TEPAT'}")
