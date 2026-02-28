import sys
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PRX_ROOT, 'tsim', 'src'))

import subprocess
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Find any cached/compiled output files
result = subprocess.run(
    ['find', _PRX_ROOT, '-name', '*.pkl', '-o', '-name', '*.npz', '-o', '-name', '*.json'],
    capture_output=True, text=True
)
print("Cached files found:")
print(result.stdout)

# Look at the first 100 lines of tsim_cutting.py to understand find_stab_cutting signature
print("\n" + "="*60)
print("find_stab_cutting signature and docstring:")
print("="*60)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tsim_cutting.py') as f:
    lines = f.readlines()

# Find find_stab_cutting definition and print surrounding lines
for i, line in enumerate(lines):
    if 'def find_stab_cutting' in line:
        # Print this function until next def or 50 lines
        for j in range(i, min(i+50, len(lines))):
            print(f"{j+1:4d}: {lines[j]}", end='')
            if j > i and lines[j].strip().startswith('def '):
                break
        print("\n---")
