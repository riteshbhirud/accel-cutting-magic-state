import stim
import numpy as np
import re
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

D3_CLIFFORD = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", "for_perfectionist_decoding", "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"

def replace_s_with_t(s):
    """Replace S→T and S_DAG→T_DAG in a circuit string."""
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

# Load Clifford proxy circuit
clifford_str = open(D3_CLIFFORD).read()

# Apply replace_s_with_t to get actual T-gate circuit string
t_gate_str = replace_s_with_t(clifford_str)

# Verify T gates are present
t_count = len(re.findall(r'^\s*T\s', t_gate_str, re.MULTILINE))
tdag_count = len(re.findall(r'^\s*T_DAG\s', t_gate_str, re.MULTILINE))
s_count = len(re.findall(r'^\s*S\s', t_gate_str, re.MULTILINE))
sdag_count = len(re.findall(r'^\s*S_DAG\s', t_gate_str, re.MULTILINE))
print(f"T gates: {t_count}, T_DAG gates: {tdag_count}")
print(f"S gates: {s_count}, S_DAG gates: {sdag_count}")

# Try to parse as stim circuit — this will FAIL because stim doesn't support T gates
print("\nAttempting stim.Circuit(t_gate_str)...")
try:
    t_gate_circuit = stim.Circuit(t_gate_str)
    print("SUCCESS: stim accepted T-gate circuit")
except Exception as e:
    print(f"FAILED: {e}")
    print("\nstim does not support T gates natively.")
    print("The T-gate circuit cannot be run through stim's TableauSimulator.")
    print("This is WHY we need tsim — stim is a Clifford simulator.")

# Show the lines that changed
print("\nOriginal S-gate lines:")
for line in clifford_str.split('\n'):
    if re.match(r'^\s*S[\s_]', line):
        print(f"  {line.strip()}")

print("\nReplaced T-gate lines:")
for line in t_gate_str.split('\n'):
    if re.match(r'^\s*T[\s_]', line):
        print(f"  {line.strip()}")
