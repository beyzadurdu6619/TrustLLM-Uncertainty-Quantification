# src/test_benchmarks.py

BENCHMARK_SUITE = [
    # --- KATEGORİ 1: NESNEL / TEK KELİMELİ (Objective Single-Token) ---
    {"prompt": "capital of France", "expected": "Paris", "type": "objective"},
    {"prompt": "capital of Germany", "expected": "Berlin", "type": "objective"},
    
    # --- KATEGORİ 2: NESNEL / ÇOKLU KELİMELİ (Objective Multi-Token) ---
    {"prompt": "capital of India", "expected": "New Delhi", "type": "objective"},
    {"prompt": "capital of USA", "expected": "Washington D.C.", "type": "objective"},
    
    # --- KATEGORİ 3: ÖZNEL / GÖRECELİ (Subjective / Opinion) ---
    {"prompt": "best movie in world", "expected": "SUBJECTIVE_REFUSAL", "type": "subjective"},
    {"prompt": "most beautiful city in europe", "expected": "SUBJECTIVE_REFUSAL", "type": "subjective"},
    {"prompt": "funniest comedian", "expected": "SUBJECTIVE_REFUSAL", "type": "subjective"},

    # --- KATEGORİ 4: YANLIŞ ÖNERMELİ / MUĞLAK (Counterfactual / Ambiguous) ---
    {"prompt": "capital of Atlantis", "expected": "LOW_CONFIDENCE_REFUSAL", "type": "ambiguous"},
]