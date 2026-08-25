import json
import urllib.request
import numpy as np
from src.active_search import ActiveSearchSolver

# URL resmi file soal ARC (007b9283.json)
url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/main/data/training/007b9283.json"

print("[Main] Mengunduh data tugas asli ARC (007b9283.json)...")

try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        task_data = json.loads(response.read().decode())
    print("[Main] Berhasil mengunduh soal ARC asli!")
except Exception as e:
    print(f"[Main] Gagal mengunduh ({e}), menggunakan data simulasi cadangan...")
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
