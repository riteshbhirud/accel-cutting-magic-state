# Approach 2: Closed Pauli Web Post-Selection for d=5 Cultivation

## Background

The d=5 Gidney cultivation circuit has 89 detectors: 69 "working" detectors from
the injection+cultivation stages, and 20 "projection" detectors from the
double-checking sub-circuit. The existing tsim pipeline (T=30) handles only the
69 working detectors. Adding any projection detector inflates T from 30 to 68
because meas[73] (qubit 30 in the final MX, positioned after the T gate at
line 430) forces the double-checking T gates into the sampling graph.

Approach 2 avoids this by computing projection detector parities **classically**
from the error configuration, before any stabilizer decomposition. This is the
"closed Pauli web post-selection" approach from Wan & Zhong Section 8.2.

---

## Q1: What is a "closed Pauli web" and how does it correspond to a stim DETECTOR?

### Definition

A closed Pauli web (Wan & Zhong Section 7, referencing Bombin et al. [11] and
Rodatz et al. [14]) is a set of edges in the ZX-diagram that forms a closed
structure — meaning any Pauli error on an edge in the web flips the web's
parity, while errors outside the web leave it unchanged. The parity is +1
(un-violated) when an even number of errors land on the web's edges, and -1
(violated) when an odd number do.

### Correspondence to stim DETECTOR

Each stim DETECTOR is a product of measurement outcomes whose expected value is
deterministic (+1) in the absence of errors. The DETECTOR fires (returns -1)
when an odd number of its contributing measurements have been flipped by errors.

The exact correspondence:

| Concept | Stim language | ZX-diagram language |
|---|---|---|
| Definition | `DETECTOR rec[-k1] rec[-k2] ...` | Set of edges forming a closed web |
| Parity computation | XOR of measurement outcomes at rec[] indices | XOR of error indicators on web edges |
| No-error value | All measurements match expectation → parity 0 | No errors on web → parity +1 |
| Error detection | Odd flips among contributing measurements → fires | Odd errors on web edges → violated |

**Key insight**: The detector parity depends **only on the error configuration**,
not on the quantum state or measurement randomness. This is because detectors
are designed as stabilizer checks — products of measurements that are
deterministic regardless of the (non-Clifford) quantum state. Pauli errors
either commute or anti-commute with the measured operators, and the
anti-commuting errors flip the measurement outcomes deterministically.

This means we can compute all 89 detector parities from the error configuration
alone, without evaluating any stabilizer decomposition or Born-rule amplitudes.

### For d=5

The 89 detectors from Gidney's circuit translate to 89 closed Pauli webs. Six
examples for d=5 are illustrated in Appendix G (Figures 16-18: detectors 81, 48,
55). The full set is available in the supplementary tikz folder `d5_det_webs`.

---

## Q2: How are Pauli errors initialized on ZX diagram edges?

### Error model

Wan & Zhong Section 5, equation (13): the noise model is circuit-level uniform
depolarizing applied to every edge:

    E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

Each Pauli X/Y/Z error occurs with probability p/3 per edge. In the ZX-diagram,
errors are represented as coloured half-edges on edges (equation 14). The paper
uses red for X, green for Z, and both for Y. Errors between CNOTs are also
valid locations (equation 15).

### Data structure in tsim

In tsim, the error representation has three layers:

**Layer 1: Noise channels → e-parameters**

Each noise location in the circuit produces a `Channel` object with:
- `probs: np.ndarray` — probability of each outcome (identity, X, Y, Z, or
  multi-qubit Paulis)
- `unique_col_ids: tuple[int, ...]` — which e-parameter columns this channel
  writes to

For the d=5 circuit: 3522 channels producing 7787 e-parameter bits total:
- 211 channels × 1 e-bit (2 outcomes: identity or single Pauli)
- 2834 channels × 2 e-bits (4 outcomes: I, X, Y, Z)
- 477 channels × 4 e-bits (16 outcomes: two-qubit depolarizing)

**Layer 2: e-parameters → f-parameters via error_transform**

The `error_transform` matrix (shape `(num_f, num_e)`, binary over GF(2)) maps
the raw e-parameter vector to a reduced f-parameter vector via Gaussian
elimination (`transform_error_basis()` in `graph.py:253`):

    f = error_transform @ e  (mod 2)

For the 69-detector circuit: `error_transform` has shape (101, 7787) — 101
f-parameters from 7787 e-bits.

For the 89-detector circuit (noiseless projection block): shape (154, 7787) —
154 f-parameters from the same 7787 e-bits. The extra 53 f-parameters are
"activated" by the double-checking structure requiring more of the e-parameter
space.

**Layer 3: f-parameters → ZX-diagram spider phases**

f-parameters appear as binary phase variables on spiders in the ZX-diagram.
Each spider's phase is: `(constant + Σ_j param_bits[j] · p[j]) mod 8`, where
`p = [f, m]` concatenates f-parameters with m-parameters (output bits). This
is stored in the `CompiledScalarGraphs` structure's `a_param_bits`,
`b_param_bits`, `c_param_bits_a/b`, `d_param_bits_a/b` arrays.

### Sampling

The `ChannelSampler` draws e-parameters from each channel's probability
distribution, then applies `error_transform` to produce f-parameters. At
p=0.001, approximately 63.9% of shots have all-zero f-parameters (noiseless),
which are served from a precomputed cache (Section 9.4.6).

---

## Q3: How is the parity of a closed Pauli web computed from an error configuration?

### Algorithm

For a given error configuration (e-parameter vector `e ∈ {0,1}^7787`):

1. **Compute f-parameters**: `f = error_transform @ e mod 2`, giving
   `f ∈ {0,1}^{num_f}`.

2. **For each detector d**: `parity_d = Σ_j W[d,j] · e[j] mod 2`, where
   `W ∈ {0,1}^{89 × 7787}` is the **web matrix** — a binary matrix where
   `W[d,j] = 1` iff e-parameter j is in the closed Pauli web of detector d.

3. **Post-select**: If `parity_d = 1` for any d ∈ {0, ..., 88} → discard shot.

This is a GF(2) matrix-vector product. The entire post-selection check is:

    reject = any( (W @ e) mod 2 )

### Constructing the web matrix W

The web matrix W can be constructed by either of two approaches:

**Approach A: From stim's detector error model (DEM)**

The DEM lists error mechanisms and which detectors each flips. Each error
mechanism corresponds to a specific Pauli error at a specific circuit location,
which maps to a specific e-parameter pattern. By collecting all (error
mechanism → detector) relationships and mapping error mechanisms to e-parameter
indices, we build W.

Concretely:
1. Build the full 89-detector stim circuit (T→S substituted, noiseless
   projection block)
2. Extract its DEM via `circuit.detector_error_model(decompose_errors=True)`
3. For each DEM instruction `error(p) D[d1] D[d2] ...`: the error corresponds
   to a specific circuit location → a specific set of e-parameter indices
4. Set `W[d, e_idx] = 1` for each (detector d, e-parameter index e_idx) pair

**Approach B: Empirical GF(2) solve**

1. Sample N shots from the full 89-detector stim circuit (single sampler,
   extracting both raw measurements and detector outcomes)
2. For each shot, also record the e-parameter vector (from channel sampling)
3. Solve `W` such that `detector_parities = W @ e_params mod 2` for all shots
4. This is a GF(2) linear system solvable by Gaussian elimination

Approach B is simpler and was partially validated in Task 88 (where we computed
all 89 detector parities from raw measurements with perfect accuracy).

**Approach C: From tsim's error_transform directly**

For the 69-detector circuit, tsim's `SamplingGraph` already encodes how
f-parameters affect detector parities through the ZX-diagram structure. The
output indices tell us which m-parameter positions correspond to detectors.
For the projection detectors, we need the 89-detector circuit's error_transform
(154, 7787). The additional rows encode the linear dependencies needed for the
20 projection detector parities.

The practical implementation: use the 89-detector error_transform to map e→f,
then identify which f-parameters correspond to which detectors via the circuit's
output structure. This gives us the projection detector parities as specific
rows of `(error_transform_89 @ e) mod 2`.

### Validation

Compare against stim's detector sampler on 100K+ shots. For each shot:
- Sample raw measurements from a single stim sampler (seeded)
- Compute detector parities from raw measurements (using rec[] indices from
  Task 88)
- Compute detector parities from e-parameters via W matrix
- Verify exact match

---

## Q4: What is the magic cat state decomposition for d=5?

### The decomposition (Appendix C, equation 29)

The basic flag-free double-checking circuit decomposes as:

    double_check ∝ I^⊗n + (XS†)^⊗n

where n is the number of input/output qubits. For d=5, n=19, giving **exactly 2
Clifford terms**.

### Derivation

The double-checking circuit consists of n controlled-H_XY gates applied to
qubits on an n-GHZ state as control and qubits on the injected colour code as
target (equation 26). The key steps:

1. Identify the two Z-spiders inside the cyan dotted circles in the circuit
   diagram (equation 26)
2. Cut both via the cutting identity (equation 18) → 2² = 4 terms (equation 27)
3. Simplify and collect terms → 2 terms (equation 28)
4. The result: I^⊗n + (XS†)^⊗n (equation 29)

### Explicit terms for d=5

- **Term 1 (I^⊗19)**: Identity on all 19 data qubits. The double-checking
  measurements see the state unchanged. All Clifford.

- **Term 2 ((XS†)^⊗19)**: Apply XS† = X · S† to each of the 19 data qubits.
  Since XS† is Clifford, this is a Clifford operation. Recall:
  `I + XS† = [[1, -i], [1, 1]]`.

### Context: full d=5 term count

| Circuit part | T spiders | Decomposition method | Terms |
|---|---|---|---|
| d=3 cultivation (injection + first cultivation) | 15 | Spider cutting + BSS | ~4 avg |
| d=5 double-checking | 38 | Magic cat (I^⊗n + (XS†)^⊗n) | 2 |
| **Full d=5** | **53** | **Product** | **~8 avg** |

This matches Figure 4 in the paper: average ~8 terms at error rates
O(10^-4) to O(10^-3).

### Important note on the Gidney circuit

The actual Gidney circuit (from stim files) has additional flag structures
beyond the basic flag-free double-checking. The paper notes (end of Appendix C):
"The difficulty in finding stabiliser decomposition for such circuits from [1]
may be due to the presence of additional flag-like structures." In our
implementation, we use the flag-free decomposition (equation 29) and handle
flags separately or verify that the Gidney circuit's double-checking matches
the flag-free structure.

From our investigation (Tasks 38-92), the d=5 Gidney circuit's double-checking
structure is:
- Line 364: T_DAG on 19 data qubits
- Lines 365-394: CX network
- Line 395: MX on qubit 25 (single midpoint measurement)
- Lines 396-429: CX network (conjugate)
- Line 430: T on 19 data qubits
- Line 433: MX on 19 qubits (final measurement → 20 projection detectors)

The T_DAG...CX...MX...CX...T structure is exactly the double-checking pattern
that decomposes as I^⊗n + (XS†)^⊗n.

---

## Q5: How are Clifford terms evaluated with Pauli errors applied?

### What "apply error to ZX diagram" means

From Sections 5 and 9.4: applying a Pauli error to a ZX-diagram edge means
inserting a coloured half-edge (spider with phase π for Z-type, or colour-
swapped spider for X-type) at that edge location. In the parametric setting:

1. Each noise channel's outcome is encoded as binary e-parameters
2. The error_transform maps e-parameters to f-parameters
3. f-parameters appear as **additive phase contributions** on spiders in the
   ZX-diagram: a spider's phase becomes `α + Σ_j param_bits[j] · f[j] · π`

The Pauli error doesn't change the graph topology — it modifies spider phases.
After `full_reduce(paramSafe=True)`, the reduced graph still has the same
Clifford structure but with parametrically-shifted phases.

### Evaluation of each Clifford term

Each fully-reduced Clifford graph g is compiled into a `CompiledScalarGraphs`
structure (Section 9.4.1). Its scalar amplitude factorizes as:

    s_g(p) = ω^{φ_g} · λ_g · 2^{r_g} · Π_t A_t · B · C · Π_t D_t

where ω = e^{iπ/4} and p = [f, m]:

- **A-terms** (node terms): `A_t = 1 + ω^{α_t}` where
  `α_t = (4·r_t + c_t) mod 8` and `r_t = Σ_j a_param_bits[t,j] · p[j]` is a
  binary row sum.

- **B-terms** (half-pi terms): `B = ω^β` where
  `β = Σ_t (r_t · τ_t) mod 8`, a collective phase from π/4-multiple phases.

- **C-terms** (pi-pair terms): `C = (-1)^{Σ_t (ψ_t · φ_t) mod 2}`, a
  collective sign from pairs of π-phase contributions. The ψ_t and φ_t are
  binary row sums with additive constant bits.

- **D-terms** (phase-pair terms):
  `D_t = 1 + ω^{α_t} + ω^{β_t} - ω^{(α_t + β_t) mod 8}`, with two
  independent row sums.

All row sums are GF(2) dot products: `r = Σ_j B[g,t,j] · p[j]`, computed via
BLAS matrix-multiply (Section 9.4.3, equation 23):

    (G·T, P) × (P, n) → mod 2 → (n, G, T)

### Split f/m evaluation (Section 9.4.5)

The parameter vector splits as p = [f, m]. Since m has only M = 2^N possible
values (N = number of output bits), the row sums decompose:

    r = (r_f + r_m) mod 2

where r_f depends on noise (computed per-batch via 6 matmuls) and r_m depends
on output combo (precomputed once at compile time as an (M, G, T) tensor).

Per-combo evaluation reduces to element-wise `r = (r_f + r_m[c]) mod 2`
followed by lookup-table evaluation and product accumulation (Algorithm 3,
lines 13-28).

### For the double-checking's 2 terms

**[CORRECTION — see Blocking Architecture Questions below]**

The original assumption that the double-checking could be evaluated independently
is WRONG. The d=3 pipeline processes the full circuit (cultivation + projection)
as ONE connected ZX graph. The same must apply to d=5.

The 2 Clifford terms from the magic cat decomposition are applied WITHIN the
full ZX graph, not as a separate evaluation. Each term produces a modified
version of the full graph where the double-checking T spiders have been replaced
by Clifford operations. Spider cutting then handles the remaining cultivation
T spiders.

---

## Q6: What is the full per-shot pipeline from error sampling to probability output?

**[REVISED — accounts for one-graph architecture]**

### Pipeline overview

```
COMPILE TIME:
  1. Build full d=5 circuit (noisy cultivation + noiseless double-checking)
  2. Convert to ZX graph
  3. Apply magic cat decomposition to double-checking T spiders → 2 branches
  4. For each branch: apply spider cutting to cultivation T spiders → ~4 terms
  5. Total: ~8 Clifford ZX-diagrams
  6. For each Clifford term: full_reduce → connected_components → determine N
  7. Compile into CompiledScalarGraphs (A/B/C/D terms, split f/m row sums)

PER-BATCH (B shots):
  1. SAMPLE ERRORS
     ├─ Draw e-parameters from channel probability distributions
     └─ Compute f = error_transform @ e mod 2

  2. CLOSED PAULI WEB POST-SELECTION (new for d=5)
     ├─ For each of 89 detectors: parity_d = W[d,:] @ e mod 2
     ├─ If any parity_d = 1 → reject shot
     └─ ~35-50% of shots survive at p=0.001

  3. NOISELESS FAST PATH
     ├─ If f = 0 → use cached noiseless distribution P(m | f=0)
     └─ ~64% of shots at p=0.001

  4. FULL AMPLITUDE EVALUATION (one connected graph, ~8 terms)
     ├─ f-only row sums: 6 matmuls per sub-component
     ├─ For each of M = 2^N output combos:
     │   ├─ r = (r_f + r_m[c]) mod 2
     │   ├─ Evaluate A,B,C,D terms via lookup table
     │   └─ Sum over G graphs per sub-component
     ├─ Product over K sub-components
     └─ N and M are determined empirically (see Blocking Q2 below)

  5. SAMPLE OUTPUT
     ├─ P(m | f) from amplitude evaluation
     ├─ Draw detector outcomes + observable from P(m | f)
     └─ Output: (det[0..88], obs[0])
```

### Step-by-step detail

**Step 1: Sample errors**

Using tsim's `ChannelSampler`:
- For each noise channel: draw from categorical distribution
- Each outcome sets specific e-parameter bits
- Apply `error_transform` to get f-parameters

With sparse geometric channel sampling (Section 10.1): only fire events are
visited, reducing random draws from ~260M to ~500K per batch at p = 5×10^-4.

**Step 2: Closed Pauli web post-selection**

Compute `reject = any((W @ e) mod 2)` where W is the (89, num_e) web matrix.

This is the key innovation. Instead of adding projection detectors to the
sampling graph (inflating T from 30 to 68), we evaluate their parities
classically from the error vector. Negligible cost (single GF(2) matmul).

Construction of W: empirical GF(2) solve from 200K stim samples (Approach B).
Each row of W encodes which e-parameter bits contribute to that detector's
parity. Validated against stim's detector sampler.

**Step 3: Noiseless fast path**

If all f-parameters are zero: sample from cached noiseless distribution.
At p=0.001, ~64% of shots take this path.

**Step 4: Full amplitude evaluation**

The full d=5 circuit is ONE connected ZX graph, processed identically to d=3
but with magic cat decomposition applied first. After magic cat (2 branches) +
spider cutting (~4 terms per branch) + full_reduce, we get ~8 Clifford terms.

The N free output bits in the main component determine M = 2^N enumeration
cost. For d=3, N=5, M=32. For d=5, N is unknown (see Blocking Q2).

The evaluation uses the same split f/m pipeline as d=3:
- f-only row sums: BLAS matmul
- m-contribution: precomputed at compile time
- Per-combo: element-wise (r_f + r_m[c]) mod 2 → LUT → product accumulation
- Sub-component factorization: product over K sub-components

**Step 5: Sample output**

Draw detector+observable outcomes from P(m | f). The output includes all 89
detector bits (with the 20 projection detectors already known to be 0 from
post-selection) and the observable bit.

### Expected performance

From the paper (Figures 4-5):
- Average ~8 terms per shot at p ∈ [10^-4, 10^-3]
- With closed Pauli web post-selection: average stays ~8, max ~16
- Post-selection survival: ~65-90% at p ∈ [0.0005, 0.001]

Throughput depends critically on N (free output bits):
- If N ≈ 5 (same as d=3): ~2M shots/s (factor of 2 overhead from 2× more terms)
- If N ≈ 10: ~200K shots/s (M=1024, 32× more combos than d=3)
- If N ≈ 20: ~2K shots/s (M=1M, requires different approach)

---

## Resolved Open Questions

**Q1 — Web matrix construction:** Use Approach B (empirical GF(2) solve). Task
88 proved all 89 detector parities can be computed from raw measurements with
perfect accuracy (1.000000 match on 100K shots). Implementation: use
`compile_sampler()` with fixed seed, compute detector parities from rec[] index
lists (saved in `d5_proj_det_indices.json`), and solve: for each of 7787
e-parameter bits, which detectors does it flip? Use 200K shots.

**Q2 — Flag structures:** Do not assume the Gidney circuit matches the flag-free
form. Verify explicitly in Phase 3: build the noiseless double-checking
sub-circuit (lines 364-433, T→S), run through
`compile_detector_sampler_subcomp_enum_general`, count resulting Clifford terms.
If exactly 2 → flag-free, proceed. If more → flags present, adjust
decomposition.

**Q3 — Double-checking output legs:** After post-selecting all 20 projection
detectors to parity 0, the 19 final MX measurements are NOT all determined.
They are free m-parameters encoding which sector of the logical code space the
state landed in. The observable = XOR of last 10 of these 19 measurements
(indices 83-92). The actual number of free m-parameters in the main component
is determined by the ZX-calculus reduction, not by the raw measurement count.
**See Blocking Q2 below for how this is resolved.**

**Q4 — Factored evaluation:** The factorization S(f,m) = S_cultivation × S_check
is WRONG. The d=3 pipeline processes the full circuit (injection + projection)
as ONE connected ZX graph (confirmed from `run.py` lines 101-102:
`circ = c_injection + c_projection`). The cultivation output state flows
directly into the double-checking on the same qubit register. **See Blocking Q1
below for full analysis.**

**Q5 — Observable:** The observable uses the last 10 of the 19 final MX
measurements (indices 83-92). These are free m-parameters after post-selection.
The observable bit is their XOR. Falls out naturally from m-parameter sampling.

---

## Blocking Architecture Questions

### Question 1: Is d=3 one connected graph or two?

**Answer: ONE connected graph. The factorization assumption is wrong.**

Evidence from `d_3_circuit_definitions.py` and `run.py`:

The d=3 circuit is defined as two separate strings:
- `circuit_source_injection_T` (lines 2-132): injection + cultivation, 18 qubits,
  T_DAG (line 55), T_DAG on 7 qubits (line 99), T on 7 qubits (line 121)
- `circuit_source_projection_proj` (lines 135-150): noiseless projection,
  T on qubit 6 (line 138), 7 measurements, 6 detectors, 1 observable

They are **concatenated into a single circuit** before compilation:

```python
# run.py lines 87-105
c_projection = tsim.Circuit(circuit_source_projection_proj)
c_injection = tsim.Circuit(noisy_injection_str)
circ = c_injection + c_projection                    # ONE circuit
sampler = compile_detector_sampler_subcomp_enum_general(circ, ...)
```

The concatenated circuit passes through `compile_detector_sampler_subcomp_enum_general`
as a single `tsim.Circuit` object. Inside tsim:
1. `circuit_to_zx(circ)` → one ZX diagram
2. `connected_components()` → multiple components (main + single-output)
3. For the main component: plug N=5 outputs, `full_reduce` → K=2 sub-components

The projection's T gate on qubit 6 is part of the same ZX graph as the
cultivation's T gates. The pipeline does NOT factor at the measurement boundary.

From the paper (Section 9.3):
> "After stabiliser decomposition and full_reduce, the fully-plugged ZX-graph of
> the d=3 cultivation circuit disconnects into K independent sub-components"

And (Section 9.1):
> "A noiseless projection circuit is used to calculate logical error rates at the
> end of the circuit."

**Implication for d=5:** The full d=5 circuit (noisy cultivation + noiseless
double-checking) must be processed as ONE connected ZX graph. The
double-checking's 38 T gates (T_DAG on 19 qubits + T on 19 qubits) are part
of the same graph as the cultivation's 15 T gates. Total: ~53 T spiders.

The d=3 circuit has ~16 T gates → cutting produces 32 Clifford terms per
sub-component → tractable.

The d=5 circuit has ~53 T gates → cutting alone is intractable → magic cat
decomposition must be applied first to the 38 double-checking T spiders,
reducing them to 2 Clifford branches within the same graph. Then cutting handles
the remaining ~15 T spiders → ~4 terms per branch → ~8 total.

**Architecture conclusion:** d=5 must follow the SAME single-graph architecture
as d=3. The magic cat decomposition is applied WITHIN the ZX graph, not as a
separate post-processing step.

### Question 2: How many free output bits does d=5 have?

**Answer: The paper does not compute this. It is explicitly "left for future
work." The answer must be determined empirically.**

From Section 11:
> "Extending these simulations to the d=5 cultivation circuit, potentially with
> GPU-accelerated tsim, is left for future work."

The paper's d=5 analysis (Sections 4-7) operates ONLY at the circuit-level
decomposition layer — counting stabilizer terms. It never builds the full d=5
sampling graph, never determines N, and never runs the sampling pipeline for d=5.

**What we know about d=3:**
- Full circuit: 18 qubits, ~28 measurements, 19 detectors + 1 observable
- Sampling graph main component: N=5 free outputs → M=32
- 28 single-output components evaluated autoregressively
- Paper: "N=5, M=32, K=2, batch size 65,536"

**Why the 2^19 concern is likely overstated:**

The 19 final MX measurements in the double-checking do NOT necessarily contribute
N=19 free outputs to the main component. Evidence:

1. **d=3 precedent:** The d=3 projection has 7 measurements (MX 0 11 8, M 9 3 14,
   MX 6) and 6 detectors + 1 observable. Despite these 7 measurements, the main
   component has only N=5 free outputs. Many measurements became single-output
   components (28 total).

2. **ZX-calculus reduction:** `full_reduce(paramSafe=True)` aggressively
   simplifies the graph, fusing spiders, removing identities, and absorbing
   boundary vertices. For the Clifford double-checking terms, this
   simplification is maximally effective.

3. **Detector structure:** Most of the 20 projection detectors are simple
   single-measurement parity checks (from Task 88: proj_det[1-6] and
   proj_det[8-19] are single measurements). These will likely become
   single-output components, not part of the main component.

4. **Magic cat decomposition effect:** After replacing the 38 double-checking
   T spiders with Clifford operations (I^⊗19 or (XS†)^⊗19), the
   double-checking becomes entirely Clifford. This means `full_reduce` can
   simplify it completely, potentially absorbing many of the 19 MX outputs.

**How the paper handles large N (Section 8.1, Algorithm 5):**

For circuits where enumeration is infeasible, the paper describes a time-sliced
"sample measurements" approach:
1. Contract the errored ZX-diagram up to the first measurement layer
2. Remove measurement spiders → more open legs
3. Perform stabilizer decomposition on the sub-diagram
4. Sample measurement outcomes from the decomposed sum
5. Plug sampled outcomes back, contract to next measurement layer
6. Repeat

This is slower (multiple decomposition rounds per shot) but handles arbitrary N.

**Architecture conclusion:**

The approach is:
1. Build full d=5 ZX graph with magic cat decomposition applied
2. Apply spider cutting to remaining cultivation T spiders
3. For each Clifford term: `full_reduce`, `connected_components`, measure N
4. **If N ≤ ~20:** use enumeration (existing pipeline, possibly slower)
5. **If N > 20:** use autoregressive sampling (tsim already implements this for
   single-output components) or the time-sliced approach

**The critical empirical experiment** (first action of Phase 3): build the full
d=5 ZX graph, apply magic cat decomposition + cutting + full_reduce, and report
N for the main component of each Clifford term. This determines the entire
downstream architecture.

**Recommended implementation order:**
1. Phase 1: Save circuit utilities (no architecture dependency)
2. Phase 2: Build Pauli web evaluator (no architecture dependency)
3. Phase 3: Build magic cat decomposition AND empirically determine N
4. Phases 4-5: Design evaluator/sampler based on empirical N
