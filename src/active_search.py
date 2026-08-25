import time
from typing import List, Tuple, Optional
import numpy as np
from src.dsl import ARCOperations
from src.evaluator import TaskEvaluator

class ActiveSearchSolver:
    def __init__(self, timeout_seconds: int = 60, max_depth: int = 2):
        self.timeout = timeout_seconds
        self.max_depth = max_depth
        
        # Daftar primitif DSL yang siap diuji
        self.primitive_ops = [
            ARCOperations.rotate_90,
            ARCOperations.rotate_180,
            ARCOperations.flip_horizontal,
            ARCOperations.flip_vertical,
            ARCOperations.invert_colors,
            ARCOperations.crop_bounding_box,
            lambda g: ARCOperations.replace_color(g, color_from=1, color_to=2),
            lambda g: ARCOperations.replace_color(g, color_from=0, color_to=3)
        ]

    def solve(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray) -> Optional[np.ndarray]:
        """Menjalankan Active Search dengan mengalokasikan alokasi waktu komputasi."""
        start_time = time.time()
        best_score = 0.0
        best_fn = None

        print(f"[ActiveSearch] Memulai pencarian (Timeout: {self.timeout}s, Max Depth: {self.max_depth})...")

        # Depth 1 Search
        for op in self.primitive_ops:
            if time.time() - start_time > self.timeout:
                break
                
            score = TaskEvaluator.evaluate_candidate(op, train_examples)
            if score > best_score:
                best_score = score
                best_fn = op
                
            if best_score == 1.0:
                print(f"[ActiveSearch] Solusi tingkat 1 ditemukan dalam {time.time() - start_time:.4f} detik!")
                return best_fn(test_input)

        # Depth 2 Search (Kombinasi 2 Fungsi)
        if best_score < 1.0 and self.max_depth >= 2:
            for op1 in self.primitive_ops:
                for op2 in self.primitive_ops:
                    if time.time() - start_time > self.timeout:
                        break
                    
                    chained_fn = lambda g, f1=op1, f2=op2: f2(f1(g))
                    score = TaskEvaluator.evaluate_candidate(chained_fn, train_examples)
                    
                    if score > best_score:
                        best_score = score
                        best_fn = chained_fn
                        
                    if best_score == 1.0:
                        print(f"[ActiveSearch] Solusi kombinasi ditemukan dalam {time.time() - start_time:.4f} detik!")
                        return best_fn(test_input)

        if best_fn is not None:
            print(f"[ActiveSearch] Selesai. Skor terbaik: {best_score:.2f}")
            return best_fn(test_input)
            
        print("[ActiveSearch] Tidak ditemukan solusi yang cocok sempurna.")
        return None
