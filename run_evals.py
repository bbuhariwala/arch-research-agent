"""
Run eval pipeline against all test cases.
Usage: python run_evals.py
"""
from src.evals import run_evals

if __name__ == "__main__":
    results = run_evals()
    
    print(f"\nEval run complete.")
    print(f"Overall score: {results['overall_average']}/5.0")