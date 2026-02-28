"""Task 1: d=3 validation via stim sampling of Clifford variants.

Build pure Clifford circuits (no T gates) and sample with stim.
Compare with tsim's existing d=3 results.
"""
import sys, os, re
from pathlib import Path
from copy import deepcopy
from collections import Counter
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim
import pyzx_param as zx
from stab_rank_cut import tcount
import stim

# Load d=3 circuit
CIRCUIT_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=15,b=Y,r=4,d1=3.stim")
circuit_str = CIRCUIT_PATH.read_text()

# Strip noise
lines = circuit_str.split('\n')
noiseless = []
for line in lines:
    s = line.strip()
    if any(s.startswith(p) for p in ['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    noiseless.append(line)

# Block identification from previous run:
# Line 36: S_DAG 3 (injection)
# Lines 78-100: S_DAG/S on qubits [0,3,7,9,10,12,13] (cultivation)

injection_line = 36
cult_start, cult_end = 78, 100
cult_qubits = [0, 3, 7, 9, 10, 12, 13]

# Apply S→T
def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

t_lines = replace_s_with_t('\n'.join(noiseless)).split('\n')

def make_xsd_lines(data_qubits):
    return [
        f"S_DAG {' '.join(str(q) for q in data_qubits)}",
        "TICK",
        f"X {' '.join(str(q) for q in data_qubits)}",
        "TICK",
    ]

def build_variant(t_lines, cult_mode, inj_mode):
    """Build variant replacing cultivation block and injection."""
    result = []
    skip = set()

    # Handle injection
    if inj_mode != "keep":
        skip.add(injection_line)

    # Handle cultivation block
    if cult_mode != "keep":
        for i in range(cult_start, cult_end + 1):
            skip.add(i)

    for i, line in enumerate(t_lines):
        if i in skip:
            if i == injection_line and inj_mode != "keep":
                if inj_mode == "identity":
                    result.append("TICK")
                elif inj_mode == "xsd":
                    result.extend(make_xsd_lines([3]))
            elif i == cult_start and cult_mode != "keep":
                if cult_mode == "identity":
                    result.append("TICK")
                elif cult_mode == "xsd":
                    result.extend(make_xsd_lines(cult_qubits))
            continue
        result.append(line)

    return '\n'.join(result)


# ============================================================================
# Build all 4 variants (injection × cultivation)
# ============================================================================
print("="*60)
print("Building 4 Clifford variants for d=3")
print("="*60)

variants = [
    ("I⊗I", "identity", "identity"),
    ("I⊗XS†", "identity", "xsd"),
    ("XS†⊗I", "xsd", "identity"),
    ("XS†⊗XS†", "xsd", "xsd"),
]

variant_texts = {}
for name, cult_m, inj_m in variants:
    text = build_variant(t_lines, cult_m, inj_m)

    # Verify no T gates remain
    has_t = any(
        l.strip().startswith('T_DAG') or
        (l.strip().startswith('T ') and not l.strip().startswith('TICK'))
        for l in text.split('\n')
    )

    # Also check ZX T-count
    circuit = tsim.Circuit(text)
    g = circuit.get_graph()
    g2 = deepcopy(g)
    zx.full_reduce(g2, paramSafe=True)
    tc = tcount(g2)

    print(f"  {name}: has_T_gates={has_t}, ZX T-count={tc}")

    if has_t:
        # Remove any remaining T/T_DAG by converting back to S/S_DAG
        text_clean = text.replace('T_DAG ', 'S_DAG ').replace('\nT ', '\nS ')
        has_t2 = any(
            l.strip().startswith('T_DAG') or
            (l.strip().startswith('T ') and not l.strip().startswith('TICK'))
            for l in text_clean.split('\n')
        )
        print(f"    After T→S conversion: has_T={has_t2}")
        variant_texts[name] = text_clean
    else:
        variant_texts[name] = text


# ============================================================================
# Sample from each variant using stim
# ============================================================================
print(f"\n{'='*60}")
print("Sampling from Clifford variants using stim")
print(f"{'='*60}")

n_samples = 200000

variant_samples = {}
for name in ["I⊗I", "I⊗XS†", "XS†⊗I", "XS†⊗XS†"]:
    text = variant_texts[name]
    try:
        stim_circ = stim.Circuit(text)
        sampler = stim_circ.compile_sampler()
        samples = sampler.sample(n_samples)

        outcomes = [''.join(str(int(b)) for b in row) for row in samples]
        counter = Counter(outcomes)

        variant_samples[name] = counter
        print(f"\n  {name}:")
        print(f"    Unique outcomes: {len(counter)}")
        print(f"    Top 5:")
        for outcome, count in counter.most_common(5):
            print(f"      {outcome}: {count/n_samples:.4f}")
    except Exception as e:
        print(f"  {name}: Error - {e}")


# ============================================================================
# Also sample from the original (noiseless Clifford part only, no T)
# The original noiseless circuit WITHOUT S→T conversion should be Clifford
# ============================================================================
print(f"\n{'='*60}")
print("Original noiseless circuit (S gates, not T)")
print(f"{'='*60}")

# The noiseless circuit with S/S_DAG (original, not T-substituted) IS the actual circuit
# S gates are Clifford! The S→T substitution makes it non-Clifford
# But wait - the S gates in the cultivation protocol ARE the T gates
# S is just how Gidney encodes T in the stim file (S[T] notation)

# Actually, the stim file uses S/S_DAG as Clifford gates
# The tsim framework CONVERTS them to T/T_DAG for non-Clifford simulation
# The original circuit with S/S_DAG is NOT the same physics as with T/T_DAG

# For the validation, I need to compare with the T-gate version
# But stim can't handle T gates directly
# So I need tsim for the original and stim for the Clifford variants

# Let me try tsim's sampling pipeline
print("\nUsing tsim's sampling pipeline for original circuit...")
try:
    from tsim.core.graph import prepare_graph, connected_components
    from tsim.compile.stabrank import find_stab

    orig_circuit = tsim.Circuit(replace_s_with_t('\n'.join(noiseless)))
    prepared = prepare_graph(orig_circuit, sample_detectors=True)

    components = connected_components(prepared.graph)
    print(f"  Components: {len(components)}")
    for i, cc in enumerate(components):
        g_cc = cc.graph
        tc_cc = tcount(g_cc)
        n_out = len(cc.output_indices)
        n_vert = len(list(g_cc.vertices()))
        print(f"    Component {i}: {n_vert} verts, {n_out} outputs, T-count={tc_cc}")

    # Find the non-trivial component
    nontrivial = [cc for cc in components if tcount(cc.graph) > 0]
    print(f"\n  Non-trivial components: {len(nontrivial)}")

    for cc in nontrivial:
        g_cc = deepcopy(cc.graph)
        zx.full_reduce(g_cc, paramSafe=True)
        tc = tcount(g_cc)
        print(f"    T-count after reduce: {tc}")

        if tc > 0:
            # Try decomposing
            print(f"    Decomposing with find_stab...")
            stab_terms = find_stab(deepcopy(cc.graph))
            print(f"    Stabilizer terms: {len(stab_terms)}")
            for j, term in enumerate(stab_terms):
                print(f"      Term {j}: {len(list(term.vertices()))} verts, "
                      f"T-count={tcount(term)}")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    traceback.print_exc()

# ============================================================================
# Compare: are the Clifford variant distributions the same?
# ============================================================================
print(f"\n{'='*60}")
print("Distribution comparison between variants")
print(f"{'='*60}")

# All valid outcomes from all variants
all_outcomes = set()
for counter in variant_samples.values():
    all_outcomes.update(counter.keys())

print(f"Total unique outcomes across all variants: {len(all_outcomes)}")

# Print distribution table for top outcomes
outcomes_sorted = sorted(all_outcomes)

print(f"\n{'Outcome':>20}", end="")
for name in ["I⊗I", "I⊗XS†", "XS†⊗I", "XS†⊗XS†"]:
    print(f"  {name:>10}", end="")
print()

# Get top outcomes by total probability across variants
total_counts = Counter()
for counter in variant_samples.values():
    total_counts.update(counter)

for outcome, _ in total_counts.most_common(20):
    print(f"{outcome:>20}", end="")
    for name in ["I⊗I", "I⊗XS†", "XS†⊗I", "XS†⊗XS†"]:
        c = variant_samples.get(name, Counter())
        p = c[outcome] / n_samples
        print(f"  {p:>10.4f}", end="")
    print()
