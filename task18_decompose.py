import sys
import stim
import numpy as np
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PRX_ROOT, 'tsim', 'src'))

# Build T-gate circuit
D3_CLIFFORD = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", "for_perfectionist_decoding", "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"

with open(D3_CLIFFORD) as f:
    content = f.read()

# Replace S->T, S_DAG->T_DAG
import re
t_content = re.sub(r'\bS_DAG\b', 'T_DAG', content)
t_content = re.sub(r'\bS\b', 'T', t_content)

# Write to temp file
with open('/tmp/d3_t_gate.stim', 'w') as f:
    f.write(t_content)

print("T-gate circuit written to /tmp/d3_t_gate.stim")

# Now use tsim's existing compilation pipeline to decompose into Clifford terms
from tsim_cutting import find_stab_cutting
from stab_rank_cut import decompose

# Try to compile the T-gate circuit through the existing pipeline
try:
    result = find_stab_cutting('/tmp/d3_t_gate.stim')
    print(f"find_stab_cutting succeeded")
    print(f"Type: {type(result)}")
    print(f"Keys/attrs: {dir(result)}")
except Exception as e:
    print(f"find_stab_cutting failed: {e}")
    import traceback
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    traceback.print_exc()
