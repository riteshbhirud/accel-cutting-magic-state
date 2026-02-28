import sys
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, '/Users/ritesh/Downloads/prx/accel-cutting-magic-state')
sys.path.insert(0, '/Users/ritesh/Downloads/prx/tsim/src')

import subprocess

# Find any cached/compiled output files
result = subprocess.run(
    ['find', '/Users/ritesh/Downloads/prx/', '-name', '*.pkl', '-o', '-name', '*.npz', '-o', '-name', '*.json'],
    capture_output=True, text=True
)
print("Cached files found:")
print(result.stdout)

# Look at the first 100 lines of tsim_cutting.py to understand find_stab_cutting signature
print("\n" + "="*60)
print("find_stab_cutting signature and docstring:")
print("="*60)

with open('/Users/ritesh/Downloads/prx/accel-cutting-magic-state/tsim_cutting.py') as f:
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
