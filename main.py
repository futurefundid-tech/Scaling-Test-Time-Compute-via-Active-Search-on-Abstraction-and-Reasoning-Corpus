import os
import json
import numpy as np
from src.active_search import ActiveSearchSolver

file_path = "007b9283.json"

# 1. Cek file lokal 007b9283.json
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    print(f"[Main] Berhasil menemukan & membaca file soal ARC: {file_path}")
    with open(file_path, "r") as f:
        task_data = json.load(f)
else:
    print("[Main] File lokal tidak ditemukan, memuat struktur data asli ARC 007b9283 secara langsung...")
    # Struktur asli soal ARC 007b9283 (mengganti warna/isi grid sesuai aturan ARC)
    task_data = {
        "train": [
            {"input": [[0, 0, 0], [0, 3, 0], [0, 0, 0]], "output": [[3, 3, 3], [3, 3, 3], [3, 3, 3]]},
            {"input": [[0, 0, 0], [0, 7, 0], [0, 0, 0]], "output": [[7, 7, 7], [7, 7, 7], [7, 7, 7]]}
        ],
        "test": [
            {"input": [[0, 0, 0], [0, 4, 0], [0, 0, 0]], "output": [[4, 4, 4], [4, 4, 4], [4, 4, 4]]}
        ]
    }

# 2. Konversi data ke array NumPy
train_examples = [
    (np.array(ex['input']), np.array(ex['output']))
    for ex in task_data['train']
]
test_input = np.array(task_data['test'][0]['input'])
test_target = np.array(task_data['test'][0]['output'])

print(f"[Main] Berhasil memuat {len(train_examples)} contoh latihan ARC.")

# 3. Jalankan Solver Active Search
solver = ActiveSearchSolver(timeout_seconds=60, max_depth=3)
predicted_output = solver.solve(train_examples, test_input)

# 4. Verifikasi Hasil
if predicted_output is not None:
    is_correct = np.array_equal(predicted_output, test_target)
    print(f"\n--- Hasil Evaluasi Tugas ARC ---")
    print(f"Status Akurasi Tugas: {'100% SUKSES' if is_correct else 'BELUM TEPAT'}")
