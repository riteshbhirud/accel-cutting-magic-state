import sys
import re
import numpy as np
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PRX_ROOT, 'tsim', 'src'))

import tsim
import pyzx_param as zx
from tsim_cutting import find_stab_cutting

D3_CLIFFORD = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", "for_perfectionist_decoding", "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
with open(D3_CLIFFORD) as f:
    content = f.read()
t_content = re.sub(r'\bS_DAG\b', 'T_DAG', content)
t_content = re.sub(r'\bS\b', 'T', t_content)

tsim_circuit = tsim.Circuit(t_content)
half_graph = tsim_circuit.get_graph()
clifford_terms = find_stab_cutting(half_graph, max_cut_iterations=20)
print(f"Got {len(clifford_terms)} Clifford terms")

# Inspect the scalar on the ORIGINAL half graph before decomposition
print(f"\nOriginal half_graph scalar: {half_graph.scalar}")
print(f"Original half_graph scalar type: {type(half_graph.scalar)}")

# Inspect first Clifford term in detail
term0 = clifford_terms[0]
term0.auto_detect_io()

print(f"\nTerm 0 scalar: {term0.scalar}")
print(f"Term 0 scalar type: {type(term0.scalar)}")

print(f"Term 0 scalar.to_number(): ", end="")
try:
    val = term0.scalar.to_number()
    print(val)
except Exception as e:
    print(f"failed: {e}")

# Get the scalar value as complex number
try:
    scalar_val = complex(term0.scalar.to_number())
    print(f"Term 0 scalar as complex: {scalar_val}")
except Exception as e:
    print(f"scalar conversion failed: {e}")
    # Try alternative
    print(f"scalar dir: {[a for a in dir(term0.scalar) if not a.startswith('_')]}")

# Get tensor WITH scalar
tensor_raw = zx.tensorfy(term0)
print(f"\nTensor without scalar: max={np.abs(tensor_raw).max():.6e}")

# Try pyzx's full tensor evaluation that includes scalar
try:
    from pyzx_param.tensor import tensorfy
    import inspect
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    print(f"\ntensorfy signature: {inspect.signature(tensorfy)}")
except Exception as e:
    print(f"tensorfy inspection failed: {e}")

# Try with preserve_scalar=True
try:
    tensor_with_scalar = zx.tensorfy(term0, preserve_scalar=True)
    print(f"Tensor WITH scalar: max={np.abs(tensor_with_scalar).max():.6e}")
except Exception as e:
    print(f"tensorfy(preserve_scalar=True) failed: {e}")

# Check boundary vertex types and ordering
print(f"\nBoundary vertices in term0:")
boundaries = [v for v in term0.vertices() if term0.type(v).name == 'BOUNDARY']
print(f"Count: {len(boundaries)}")
for v in boundaries[:10]:
    print(f"  vertex {v}: type={term0.type(v)}, row={term0.row(v):.2f}, qubit={term0.qubit(v):.2f}")

# Check inputs vs outputs
inputs = term0.inputs()
outputs = term0.outputs()
print(f"\nInputs: {list(inputs)}")
print(f"Outputs: {list(outputs)}")
print(f"Input count: {len(inputs)}, Output count: {len(outputs)}")

# Also check a few more terms' scalars
print("\nScalar values for first 5 terms:")
for i in range(min(5, len(clifford_terms))):
    t = clifford_terms[i]
    try:
        sv = t.scalar.to_number()
        print(f"  Term {i}: scalar = {sv}")
    except Exception as e:
        print(f"  Term {i}: scalar error: {e}")
